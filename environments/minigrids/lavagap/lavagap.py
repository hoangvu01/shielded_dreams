import random
from enum import IntEnum

import gymnasium
import numpy as np
import torch
import torch.nn.functional as F
from minigrid.wrappers import ImgObsWrapper


class LavaGapMinigrid():
    def __init__(self, seed=0, max_episode_length=100, action_repeat=1, bit_depth=1, render_mode='rgb_array'):
        self.max_episode_length = max_episode_length
        
        # Environment config
        _env = gymnasium.make("MiniGrid-LavaGapS5-v0", render_mode=render_mode, max_episode_steps=max_episode_length)
        _env = ImgObsWrapper(_env) # Remove 'mission' field
        self._env = _env

        self._t = 0

    def reset(self):
        self._env.reset()
        self._t = 0

        obs, info = self._env.reset()
        obs = torch.tensor(obs).flatten()
        return obs

    def step(self, action):
        action = action.argmax()
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        
        obs = torch.tensor(obs).flatten()
        violation = 0
        return obs, reward, violation, done 

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    @property
    def observation_size(self):
        return np.prod(self._env.observation_space.shape).item()

    @property
    def action_size(self):
        return self._env.action_space.n.item()

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        a = random.randint(0, self.action_size - 1)
        return F.one_hot(torch.tensor([a]), num_classes=self.action_size)
    
if __name__ == '__main__':
    e = LavaGapMinigrid(render_mode='human')
    e.reset()
    e.render()
    while True:
        try:
            cmd = input('Continue?')
            e.step(e.sample_random_action())
            e.render()
        except EOFError:
            break
