import argparse
import os
import time
import numpy as np
import yaml
from pathlib import Path

import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from config import parser
from dreamer import Dreamer
from environments.env import Env, EnvBatcher
from environments.memory import ExperienceReplay
from agent.models import (
    bottle,
    Encoder,
    ObservationModel,
    RewardModel,
    TransitionModel,
    ValueModel,
    ActorModel,
    ViolationModel,
)
from agent.planner import MPCPlanner
from shields.bps import BoundedPrescienceShield, ShieldBatcher
from utils import (
    lineplot,
    write_video,
    imagine_ahead,
    lambda_return,
    FreezeParameters,
    ActivateParameters,
)
from torch.utils.tensorboard import SummaryWriter

# Hyper parameters
args = parser.parse_args()

with open(args.path, "r") as fp:
    defaults = yaml.safe_load(fp)
    parser.set_defaults(**defaults)
    args = parser.parse_args()

args.overshooting_distance = min(
    args.chunk_size, args.overshooting_distance
)  # Overshooting distance cannot be greater than chunk size

print(" " * 10 + "Options")
for k, v in vars(args).items():
    print(" " * 10 + k + ": " + str(v))


# Setup
results_dir = os.path.join("results", "{}_{}".format(args.env, args.id))
os.makedirs(results_dir, exist_ok=True)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available() and not args.disable_cuda:
    print("using CUDA")
    args.device = torch.device("cuda")
    torch.cuda.manual_seed(args.seed)
else:
    print("using CPU")
    args.device = torch.device("cpu")

torch.set_default_device(args.device)
torch.set_default_dtype(torch.float32)

metrics = {
    "steps": [],
    "episodes": [],
    "train_rewards": [],
    "test_episodes": [],
    "test_rewards": [],
    "observation_loss": [],
    "reward_loss": [],
    "kl_loss": [],
    "actor_loss": [],
    "value_loss": [],
    "violation_loss": [],
    "violation_count": [],
}

summary_name = results_dir + "/{}_{}_log"
writer = SummaryWriter(summary_name.format(args.env, args.id))

# Initialise training environment and experience replay memory
env = Env(
    args.env,
    args.max_episode_length,
    args.action_repeat,
    args.bit_depth,
    args.render,
)

if not args.test:
    D = ExperienceReplay(
        args.experience_size,
        env.observation_size,
        env.action_size,
        args.bit_depth,
        args.device,
    )
    # Initialise dataset D with S random seed episodes
    for s in range(1, args.seed_episodes + 1):
        violation_count = 0
        observation, _ = env.reset()
        observation = torch.tensor(observation)
        done, t = False, 0
        while not done:
            action = env.sample_random_action()
            next_observation, reward, done, _, info = env.step(action)
            next_observation = torch.tensor(next_observation)

            violation = info["violation"]
            if violation:
                violation_count += 1
            D.append(observation, action, reward, violation, done)
            observation = next_observation
            t += 1
        metrics["violation_count"].append((s, violation_count))
        metrics["steps"].append(
            t * args.action_repeat
            + (0 if len(metrics["steps"]) == 0 else metrics["steps"][-1])
        )
        metrics["episodes"].append(s)


# Initialise model parameters randomly
agent = Dreamer(args, env)
agent.load_models()

# Global prior N(0, I)
global_prior = Normal(
    torch.zeros(args.batch_size, args.state_size),
    torch.ones(args.batch_size, args.state_size),
)

# Allowed deviation in KL divergence
free_nats = torch.full((1,), args.free_nats)


def class_weighted_bce_loss(pred, target, positive_weight, negative_weight):
    # Calculate class-weighted BCE loss
    return negative_weight * (target - 1) * torch.clamp(
        torch.log(1 - pred), -100, 0
    ) - positive_weight * target * torch.clamp(torch.log(pred), -100, 0)


def update_belief_and_act(
    args,
    env,
    planner,
    transition_model,
    violation_model,
    observation_model,
    encoder,
    belief,
    posterior_state,
    action,
    observation,
    violation,
    shield,
    explore=False,
):
    # Infer belief over current state q(s_t|o≤t,a<t) from the history
    belief, _, _, _, posterior_state, _, _ = transition_model(
        posterior_state,
        action.unsqueeze(dim=0),
        belief,
        encoder(observation).unsqueeze(dim=0),
    )  # Action and observation need extra time dimension
    imagd_violation = torch.argmax(
        bottle(violation_model, (belief, posterior_state)).squeeze()
    )

    # if not isinstance(violation, torch.Tensor):
    #     if violation == 1 and imagd_violation > 0.8:
    #         print("correctly pred violation")
    #     elif violation == 1 and imagd_violation < 0.8:
    #         print("missed violation")
    #     elif violation == 0 and imagd_violation > 0.8:
    #         print("incorrectly pred violation")

    belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(
        dim=0
    )  # Remove time dimension from belief/state

    if args.algo == "dreamer":
        action = planner.get_action(belief, posterior_state, det=not (explore)).to(
            args.device
        )
    else:
        action = planner(
            belief, posterior_state
        )  # Get action from planner(q(s_t|o≤t,a<t), p)
    if explore:
        action = torch.clamp(
            Normal(action.float(), args.action_noise).rsample(), -1, 1
        ).to(
            args.device
        )  # Add gaussian exploration noise on top of the sampled action
    shield_action, shield_interfered = shield.step(
        belief,
        posterior_state,
        action,
        # observation_model,
        planner,
        # observation,
        # encoder,
    )

    print(belief[0, 0], posterior_state[0, 0])

    action = shield_action
    next_observation, reward, done, _, info = env.step(
        action.cpu() if isinstance(env, EnvBatcher) else action[0].cpu()
    )  # action[0].cpu())  # Perform environment step (action repeats handled internally)
    next_observation = torch.tensor(next_observation).to(torch.float32)
    violation = info["violation"]

    return belief, posterior_state, action, next_observation, reward, violation, done


# Testing only
if args.test:
    # Set models to eval mode
    agent.set_eval()
    shield = agent.build_shield()

    with torch.no_grad():
        rewards = []
        violations = []
        for _ in tqdm(range(args.test_episodes)):
            total_reward = 0
            total_violations = 0
            observation, _ = env.reset()
            observation = torch.tensor(observation).to(torch.float32)

            violation = torch.zeros((1, 1))
            belief, posterior_state, action = (
                torch.zeros(1, args.belief_size, device=args.device),
                torch.zeros(1, args.state_size, device=args.device),
                torch.zeros(1, env.action_size, device=args.device),
            )

            pbar = tqdm(range(args.max_episode_length // args.action_repeat))
            for t in pbar:
                belief, posterior_state, action = agent.policy(
                    belief, posterior_state, action, observation, explore=False
                )

                shield_action, shield_interfered = shield.step(
                    belief,
                    posterior_state,
                    action,
                    agent.actor_model,
                )
                action = shield_action

                # Perform environment step (action repeats handled internally)
                observation, reward, done, _, info = env.step(action[0].cpu())
                observation = torch.tensor(observation).to(torch.float32)
                violation = info["violation"]

                total_reward += reward
                if violation:
                    total_violations += 1
                if args.render:
                    env.render()
                if done:
                    pbar.close()
                    break
            rewards.append(total_reward)
            violations.append(total_violations)

    print("Average Reward:", np.mean(rewards), " std ", np.std(rewards))
    print("Average Violations:", np.mean(violations), " std ", np.std(violations))
    env.close()
    quit()

# Close training environment
env.close()
