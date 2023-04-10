import cv2
import numpy as np
import torch
import random

from environments.minigrids.lavagap.lavagap import LavaGapMinigrid

from .grid import GridEnv

GYM_ENVS = ['Pendulum-v0', 'MountainCarContinuous-v0', 'Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2',
            'HumanoidStandup-v2', 'InvertedDoublePendulum-v2', 'InvertedPendulum-v2', 'Reacher-v2', 'Swimmer-v2', 'Walker2d-v2']
MINIGRID_ENVS = ['MiniGrid-LavaGapS5-v0']
CONTROL_SUITE_ENVS = ['cartpole-balance', 'cartpole-swingup', 'reacher-easy', 'finger-spin', 'cheetah-run', 'ball_in_cup-catch',
                      'walker-walk', 'reacher-hard', 'walker-run', 'humanoid-stand', 'humanoid-walk', 'fish-swim', 'acrobot-swingup']
CONTROL_SUITE_ACTION_REPEATS = {'cartpole': 8, 'reacher': 4, 'finger': 2,
                                'cheetah': 4, 'ball_in_cup': 6, 'walker': 2, 'humanoid': 2, 'fish': 2, 'acrobot': 4}


def Env(env, seed, max_episode_length, action_repeat, bit_depth, render=False):
    render_mode = 'human' if render else None
    return LavaGapMinigrid(seed, max_episode_length, action_repeat, bit_depth, render_mode=render_mode)

# Wrapper for batching environments together
class EnvBatcher():
    def __init__(self, env_class, env_args, env_kwargs, n):
        self.n = n
        self.envs = [env_class(*env_args, **env_kwargs) for _ in range(n)]
        self.dones = [True] * n

    # Resets every environment and returns observation
    def reset(self):
        observations = [env.reset() for env in self.envs]
        self.dones = [False] * self.n
        return torch.cat(observations)

     # Steps/resets every environment and returns (observation, reward, done)
    def step(self, actions):
        # Done mask to blank out observations and zero rewards for previously terminated environments
        done_mask = torch.nonzero(torch.tensor(self.dones))[:, 0]
        observations, rewards, violations, dones = zip(
            *[env.step(action) for env, action in zip(self.envs, actions)])
        # Env should remain terminated if previously terminated
        dones = [d or prev_d for d, prev_d in zip(dones, self.dones)]
        self.dones = dones
        observations, rewards, violations, dones = torch.cat(observations), torch.tensor(
            rewards, dtype=torch.float32), torch.tensor(violations, dtype=torch.float32),  torch.tensor(dones, dtype=torch.uint8)
        observations[done_mask] = 0
        rewards[done_mask] = 0
        violations[done_mask] = 0
        return observations, rewards, violations, dones

    def close(self):
        [env.close() for env in self.envs]
