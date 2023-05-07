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

# Initialise training environment and experience replay memory
env = Env(
    args.env,
    args.max_episode_length,
    args.action_repeat,
    args.bit_depth,
    args.render,
)

# Initialise model parameters randomly
agent = Dreamer(args, env, results_dir, metrics, None)
agent.load_models()

# Set models to eval mode
agent.set_eval()
shield = agent.build_shield()

with torch.no_grad():
    rewards = []
    violations = []
    for t in (pbar := tqdm(range(0, args.test_episodes))):
        total_reward, total_violations = 0, 0

        observation, _ = env.reset()
        observation = torch.tensor(observation, dtype=torch.float32)

        violation = torch.zeros((1, env.violation_size))
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
            # action = shield_action
            print(action)

            # Perform environment step (action repeats handled internally)
            observation, reward, done, _, info = env.step(action.cpu())
            observation = torch.tensor(observation, dtype=torch.float32)

            violation = info["violation"]

            total_reward += reward
            total_violations += violation.sum()
            if args.render:
                env.render()

            pbar.set_postfix(
                r=total_reward,
                v=total_violations,
                interfered=shield_interfered,
                action=action.argmax().item(),
                shield=shield_action.argmax().item(),
            )
            if done:
                break
        rewards.append(total_reward)
        violations.append(total_violations)

    pbar.close()

print("Average Reward:", np.mean(rewards), " std ", np.std(rewards))
print("Average Violations:", np.mean(violations), " std ", np.std(violations))
env.close()
