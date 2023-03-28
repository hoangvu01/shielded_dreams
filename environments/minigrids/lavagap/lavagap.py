import random
from enum import IntEnum

import gymnasium
import numpy as np
import torch
import torch.nn.functional as F
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper


class LavaGapMinigrid:
    def __init__(
        self,
        seed=0,
        max_episode_length=100,
        action_repeat=1,
        bit_depth=1,
        render_mode="rgb_array",
        screen_size=620,
    ):
        self.max_episode_length = max_episode_length

        # Environment config
        _env = gymnasium.make(
            "MiniGrid-LavaGapS5-v0",
            render_mode=render_mode,
            max_episode_steps=max_episode_length,
            screen_size=screen_size,
        )
        self._env = _env
        self._env_partial = ImgObsWrapper(_env)  # Without 'mission' field
        self._env_full = FullyObsWrapper(_env)  # Fully observable grid

        self._t = 0

    def reset(self):
        self._env.reset()
        self._t = 0

        obs, info = self._env.reset()
        partial_obs = self._env_partial.observation(obs)
        partial_obs = torch.tensor(partial_obs).to(torch.float32).view(1, -1)

        return partial_obs

    def step(self, action):
        prev_grid = self._env_full.observation({})["image"][:, :, 0]

        action = action.argmax()

        obs, reward, terminated, truncated, info = self._env.step(action)
        partial_obs = self._env_partial.observation(obs)
        partial_obs = torch.tensor(partial_obs).to(torch.float32).view(1, -1)

        done = terminated or truncated

        x, y = self._env.agent_pos
        cur_cell = prev_grid[x, y].item()

        violation = 0
        if cur_cell == 9 or cur_cell == 2:
            violation = 1

        return partial_obs, reward, violation, done

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    @property
    def observation_size(self):
        return np.prod(self._env_partial.observation_space.shape).item()

    @property
    def action_size(self):
        return self._env.action_space.n.item()

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        a = random.randint(0, self.action_size - 1)
        return F.one_hot(torch.tensor([a]), num_classes=self.action_size)


if __name__ == "__main__":
    e = LavaGapMinigrid(render_mode="human")
    e.reset()
    e.render()
    while True:
        try:
            cmd = input("Move? (0, 1, 2) ")
            move = torch.zeros(e.observation_size)
            move[int(cmd)] = 1
            e.step(move)
            e.render()
        except EOFError:
            break
