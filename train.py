import argparse
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from agent.models import (
    ActorModel,
    Encoder,
    ObservationModel,
    RewardModel,
    TransitionModel,
    ValueModel,
    ViolationModel,
    bottle,
)
from agent.planner import MPCPlanner
from config import parser
from dreamer import Dreamer
from environments.env import Env, EnvBatcher
from environments.memory import ExperienceReplay
from shields.bps import BoundedPrescienceShield, ShieldBatcher
from utils import (
    ActivateParameters,
    FreezeParameters,
    imagine_ahead,
    lambda_return,
    lineplot,
    write_video,
)

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

metrics = defaultdict(list)
summary_name = results_dir + "/{}_{}_log"
writer = SummaryWriter(summary_name.format(args.env, args.id))

# Initialise training environment and experience replay memory
env = Env(
    args.env,
    args.max_episode_length,
    args.action_repeat,
    args.bit_depth,
)

if not args.test:
    D = ExperienceReplay(
        args.experience_size,
        env.observation_size,
        env.action_size,
        env.violation_size,
        args.bit_depth,
        args.device,
    )
    # Initialise dataset D with S random seed episodes
    for s in range(1, args.seed_episodes + 1):
        violation_count = 0
        observation, _ = env.reset()
        observation = torch.tensor(observation, dtype=torch.float32)

        done, t = False, 0
        while not done:
            action = env.sample_random_action()
            next_observation, reward, done, _, info = env.step(action)
            next_observation = torch.tensor(next_observation, dtype=torch.float32)

            violation = info["violation"]
            violation_count += violation.sum()

            D.append(observation.cpu(), action.cpu(), reward, violation, done)
            observation = next_observation
            t += 1

        metrics["violation_count"].append((s, violation_count))
        metrics["steps"].append(
            t * args.action_repeat
            + (0 if len(metrics["steps"]) == 0 else metrics["steps"][-1])
        )
        metrics["episodes"].append(s)


# Initialise model parameters randomly
agent = Dreamer(args, env, results_dir, metrics, writer)
if args.models != "":
    agent.load_models()

# Training (and testing)
for episode in (
    trainbar := tqdm(
        range(metrics["episodes"][-1] + 1, args.episodes + 1),
        total=args.episodes,
        initial=metrics["episodes"][-1] + 1,
    )
):
    # Model fitting
    losses = []
    agent.set_train()

    for s in (pbar := tqdm(range(args.collect_interval))):
        # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
        # Transitions start at time t = 0
        observations, actions, rewards, violations, nonterminals = D.sample(
            args.batch_size, args.chunk_size
        )
        pbar.set_description(f"Episode {episode} - training models ")
        loss = agent.train(observations, actions, rewards, violations, nonterminals)

        pbar.set_postfix(obs=loss[0], kl=loss[2], actor=loss[3], value=loss[4])
        losses.append(loss)
    pbar.close()

    # Update and plot loss metrics
    losses = tuple(zip(*losses))
    metrics["observation_loss"].append(losses[0])
    metrics["reward_loss"].append(losses[1])
    metrics["kl_loss"].append(losses[2])
    metrics["actor_loss"].append(losses[3])
    metrics["value_loss"].append(losses[4])
    metrics["violation_loss"].append(losses[5])

    with torch.no_grad():
        agent.set_eval()
        shield = agent.build_shield()
        violations, total_reward = 0, 0

        observation, _ = env.reset()
        observation = torch.tensor(observation, dtype=torch.float32)

        belief, posterior_state = agent.build_initial_params(1)
        action, violation = (
            torch.zeros(1, env.action_size),
            torch.zeros(1, env.violation_size),
        )

        for s in (dbar := tqdm(range(args.max_episode_length // args.action_repeat))):
            dbar.set_description(f"Simulating - Step {s} ")
            belief, posterior_state, action = agent.policy(
                belief, posterior_state, action, observation, explore=True
            )

            shield_action, shield_interfered = shield.step(
                belief,
                posterior_state,
                action,
                agent.actor_model,
            )
            # action = shield_action

            # Perform environment step (action repeats handled internally)
            next_observation, reward, done, _, info = env.step(action.cpu())
            next_observation = torch.tensor(next_observation, dtype=torch.float32)
            violation = info["violation"]

            D.append(observation.cpu(), action.cpu(), reward, violation, done)

            total_reward += reward
            observation = next_observation

            violations += violation.sum()
            if args.render:
                env.render()
            if done:
                dbar.close()
                break

        # Update and plot train reward metrics
        metrics["steps"].append(s + metrics["steps"][-1])
        metrics["episodes"].append(episode)
        metrics["train_rewards"].append(total_reward)
        metrics["violation_count"].append((episode, violations))

    # Test
    if episode % args.test_interval == 0:
        # Initialise parallelised test environments
        agent.set_eval()
        shield = agent.build_shield()

        test_env = Env(
            args.env,
            args.max_episode_length,
            args.action_repeat,
            args.bit_depth,
            args.render,
        )

        with torch.no_grad():
            for t in (tbar := tqdm(range(args.test_episodes))):
                total_violations, total_reward = 0, 0

                observation, _ = test_env.reset()
                observation = torch.tensor(observation, dtype=torch.float32)

                belief, posterior_state = agent.build_initial_params(1)
                action, violation = (
                    torch.zeros(1, test_env.action_size),
                    torch.zeros(1, test_env.violation_size),
                )

                for s in range(args.max_episode_length):
                    tbar.set_description(f"Testing {t} - Step {s} ")
                    belief, posterior_state, action = agent.policy(
                        belief, posterior_state, action, observation, explore=False
                    )

                    shield_action, shield_interfered = shield.step(
                        belief,
                        posterior_state,
                        action,
                        agent.actor_model,
                    )
                    # action = shield_action

                    # Perform environment step (action repeats handled internally)
                    observation, reward, done, _, info = test_env.step(action.cpu())
                    observation = torch.tensor(observation, dtype=torch.float32)

                    total_reward += reward
                    total_violations += info["violation"].sum()

                    if args.render:
                        test_env.render()
                    if done:
                        break
                tqdm.write(
                    "test episode: {}, reward: {}, violations: {}".format(
                        t, total_reward, total_violations
                    )
                )

            test_env.close()
            tbar.close()
            tqdm.write(
                "episodes: {}, total_steps: {}, train_reward: {}, violations: {} ".format(
                    metrics["episodes"][-1],
                    metrics["steps"][-1],
                    metrics["train_rewards"][-1],
                    metrics["violation_count"][-1][1],
                )
            )
        torch.save(metrics, os.path.join(results_dir, "metrics.pth"))

    agent.write_summaries()

    trainbar.set_postfix(
        episode=metrics["episodes"][-1],
        steps=metrics["steps"][-1],
        train_rewards=metrics["train_rewards"][-1],
        violations=metrics["violation_count"][-1][1],
    )
    # Checkpoint models
    if episode % args.checkpoint_interval == 0:
        agent.checkpoint(os.path.join(results_dir, "models_%d.pth" % episode))

        if args.checkpoint_experience:
            # Warning: will fail with MemoryError with large memory sizes
            torch.save(D, os.path.join(results_dir, "experience.pth"))

agent.checkpoint(os.path.join(results_dir, "models_final.pth"))
# Close training environment
env.close()
