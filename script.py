import argparse
from collections import defaultdict
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

metrics = defaultdict(list)
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


def class_weighted_bce_loss(pred, target, positive_weight, negative_weight):
    # Calculate class-weighted BCE loss
    return negative_weight * (target - 1) * torch.clamp(
        torch.log(1 - pred), -100, 0
    ) - positive_weight * target * torch.clamp(torch.log(pred), -100, 0)


# Testing only
if args.test:
    # Set models to eval mode
    agent.set_eval()
    shield = agent.build_shield()

    with torch.no_grad():
        rewards = []
        violations = []
        for t in (pbar := tqdm(range(0, args.test_episodes))):
            total_reward, total_violations = 0, 0

            observation, _ = env.reset()
            observation = torch.tensor(observation).to(torch.float32)

            violation = torch.zeros((1, 1))
            action = torch.zeros(1, env.action_size)

            belief, posterior_state = agent.build_initial_params(1)

            for s in range(args.max_episode_length):
                pbar.set_description(f"Testing - Step {s} / Episode {t} ")
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

                pbar.set_postfix(
                    r=total_reward, v=total_violations, interfered=shield_interfered
                )
                if done:
                    break
            rewards.append(total_reward)
            violations.append(total_violations)

        pbar.close()

    print("Average Reward:", np.mean(rewards), " std ", np.std(rewards))
    print("Average Violations:", np.mean(violations), " std ", np.std(violations))
    env.close()
    quit()

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

    for l in [
        "observation_loss",
        "reward_loss",
        "kl_loss",
        "actor_loss",
        "value_loss",
        "violation_loss",
    ]:
        lineplot(
            metrics["episodes"][-len(metrics[l]) :],
            metrics[l],
            l,
            results_dir,
        )

    with torch.no_grad():
        shield = agent.build_shield()
        violations, total_reward = 0, 0

        observation, _ = env.reset()
        observation = torch.tensor(observation, dtype=torch.float32)

        belief, posterior_state = agent.build_initial_params(1)
        action, violation = (
            torch.zeros(1, env.action_size),
            torch.zeros(1, 1),
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
            action = shield_action

            # Perform environment step (action repeats handled internally)
            next_observation, reward, done, _, info = env.step(action[0].cpu())
            next_observation = torch.tensor(observation, dtype=torch.float32)
            violation = info["violation"]

            D.append(observation.cpu(), action.cpu(), reward, violation, done)

            total_reward += reward
            observation = next_observation
            if violation:
                violations += 1
            if args.render:
                env.render()
            if done:
                dbar.close()
                break

        # Update and plot train reward metrics
        metrics["steps"].append(t + metrics["steps"][-1])
        metrics["episodes"].append(episode)
        metrics["train_rewards"].append(total_reward)
        metrics["violation_count"].append((episode, violations))
        lineplot(
            metrics["episodes"][-len(metrics["train_rewards"]) :],
            metrics["train_rewards"],
            "train_rewards",
            results_dir,
        )
        lineplot(
            [x for x, y in metrics["violation_count"]],
            [y for x, y in metrics["violation_count"]],
            "violation_count",
            results_dir,
        )

    # Test
    if episode % args.test_interval == 0:
        # Initialise parallelised test environments
        agent.set_eval()
        shield = agent.build_shield()

        with torch.no_grad():
            for test_t in (tbar := tqdm(range(args.test_episodes))):
                total_reward, total_violations = 0, 0

                observation, _ = env.reset()
                observation = torch.tensor(observation, dtype=torch.float32)

                violation = torch.zeros((1, 1))
                action = torch.zeros(1, env.action_size)

                belief, posterior_state = agent.build_initial_params(1)
                pbar.set_description(f"Testing - Episode {test_t} ")

                for test_s in range(args.max_episode_length):
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
                        tbar.set_postfix(r=total_reward, v=total_violations)
                        break
            tbar.close()
            print(
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


# Close training environment
env.close()
