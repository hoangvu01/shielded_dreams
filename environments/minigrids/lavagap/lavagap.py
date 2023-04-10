import random

import gymnasium
import numpy as np
import torch
import torch.nn.functional as F
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
from minigrid.core.world_object import Lava, Wall, Goal


class LavaGapMinigrid:
    def __init__(
        self,
        seed=0,
        max_episode_length=100,
        action_repeat=1,
        bit_depth=1,
        render_mode="human",
        screen_size=300,
    ):
        self.max_episode_length = max_episode_length

        # Environment config
        _env = gymnasium.make(
            "MiniGrid-LavaGapS5-v0",
            obstacle_type=Lava,
            render_mode=render_mode,
            max_episode_steps=max_episode_length,
            screen_size=screen_size,
        )
        self._env = _env
        self._env_partial = ImgObsWrapper(_env)  # Without 'mission' field
        self._env_full = FullyObsWrapper(_env)  # Fully observable grid

        self._t = 0
        self.wall_hits = 0
        self.lava_hits = 0

    def reset(self):
        self._env.reset()
        self._t = 0
        self.wall_hits = 0
        self.lava_hits = 0

        obs, info = self._env.reset()
        partial_obs = self._env_partial.observation(obs)
        partial_obs = torch.tensor(partial_obs).to(torch.float32).view(1, -1)

        return partial_obs

    def step(self, action):
        action = action.argmax()

        # Next cell / position
        cur_cell = self._env.grid.get(*self._env.agent_pos)
        fwd_cell = self._env.grid.get(*self._env.front_pos)

        obs, env_reward, terminated, truncated, info = self._env.step(action)
        partial_obs = self._env_partial.observation(obs)

        x, y = self._env_partial.agent_pos
        rx, ry = self._env.relative_coords(x, y)

        violation = 0

        done = False
        reward = -0.01

        if action == 2:
            hit_wall = isinstance(fwd_cell, Wall)
            hit_lava = isinstance(fwd_cell, Lava)

            if hit_lava:
                self.lava_hits += 1
                violation = 1
            elif hit_wall:
                self.wall_hits += 1
                violation = 1

            # Replaces current position with object
            if fwd_cell is not None:
                partial_obs[rx, ry] = fwd_cell.encode()[0]

            if isinstance(fwd_cell, Goal):
                done = True
                reward = 1

        elif cur_cell is not None:
            partial_obs[rx, ry] = cur_cell.encode()[0]

        flattened_obs = torch.tensor(partial_obs).to(torch.float32).view(1, -1)
        return flattened_obs, reward, violation, done

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    @property
    def observation_size(self):
        return np.prod(self._env_partial.observation_space.shape).item()

    @property
    def action_size(self):
        # return self._env.action_space.n.item()
        return 3

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
            state, reward, violation, done = e.step(move)
            print(reward, violation)
            e.render()
            if done:
                e.reset()
        except EOFError:
            break
