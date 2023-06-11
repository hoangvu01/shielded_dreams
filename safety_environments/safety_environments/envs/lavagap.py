import random

import gymnasium
import numpy as np
import torch
import torch.nn.functional as F
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
from minigrid.core.world_object import Lava, Goal, Wall


class LavaGapMinigrid(gymnasium.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        max_episode_steps=100,
        action_repeat=1,
        bit_depth=1,
        render_mode="human",
        screen_size=300,
        grid_size=5,
        lava_death=False,
        test=False,
    ):
        super().__init__()

        # Environment config
        _env = gymnasium.make(
            f"MiniGrid-LavaGapS{grid_size}-v0",
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            screen_size=screen_size,
        )

        self.action_space = gymnasium.spaces.Discrete(3)
        self.observation_space = gymnasium.spaces.Dict(
            {
                "direction": _env.observation_space["direction"],
                "mission": _env.observation_space["mission"],
                "image": gymnasium.spaces.Box(
                    low=0,
                    high=255,
                    shape=(1, np.prod(_env.observation_space["image"].shape).item()),
                    dtype=np.uint8,
                ),
            }
        )

        self._env = _env
        self._env_partial = ImgObsWrapper(_env)  # Without 'mission' field
        self._env_full = FullyObsWrapper(_env)  # Fully observable grid

        self._lava_death = lava_death

        self._t = 0
        self.wall_hits = 0
        self.lava_hits = 0
        self.max_episode_steps = max_episode_steps

        self.prev_step = []
        self.test = test

    def reset(self, seed=None, options=None):
        self._t = 0
        self.wall_hits = 0
        self.lava_hits = 0

        obs, info = self._env.reset(seed=seed, options=options)
        partial_obs = self._env_partial.observation(obs).reshape(1, -1)
        obs["image"] = partial_obs
        return obs, info

    def step(self, action):
        # Next cell / position
        cur_cell = self._env.grid.get(*self._env.agent_pos)
        fwd_cell = self._env.grid.get(*self._env.front_pos)

        obs, env_reward, _, truncated, info = self._env.step(action)
        partial_obs = self._env_partial.observation(obs)

        x, y = self._env_partial.agent_pos
        rx, ry = self._env.relative_coords(x, y)

        reward = 0
        terminated, hit_wall, hit_lava, idle = False, False, False, False
        if isinstance(self._env.grid.get(*self._env.agent_pos), Lava):
            self.lava_hits += 1
            reward -= 0.1
            hit_lava = True

        self.prev_step.append(action)
        self.prev_step = self.prev_step[-4:]
        if len(self.prev_step) == 4 and 2 not in self.prev_step:
            idle = True

        if action == 2:
            if isinstance(fwd_cell, Wall):
                hit_wall = True
                reward -= 0.1
                self.wall_hits += 1

            # Replaces current position with object
            if fwd_cell is not None:
                partial_obs[rx, ry] = fwd_cell.encode()[0]

            if isinstance(fwd_cell, Goal):
                terminated = True
                reward = 1
        elif cur_cell is not None:
            partial_obs[rx, ry] = cur_cell.encode()[0]

        flattened_obs = partial_obs.reshape(1, -1)
        obs["image"] = flattened_obs

        info["violation"] = np.array(
            [hit_wall, hit_lava], dtype=np.int32
        ).reshape(1, -1)

        if self.test: 
            reward = env_reward
        return obs, reward, terminated, truncated, info

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    @property
    def observation_size(self):
        return np.prod(self.observation_space["image"].shape).item()

    @property
    def action_size(self):
        return self.action_space.n.item()

    @property
    def violation_size(self):
        return 2

    @property
    def violation_keys(self):
        return ["hit_wall", "hit_lava"]

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        a = random.randint(0, self.action_size - 1)
        return F.one_hot(torch.tensor([a]), num_classes=self.action_size)


if __name__ == "__main__":
    e = LavaGapMinigrid(render_mode="human", grid_size=7, screen_size=400)
    e.reset()
    e.render()
    while True:
        try:
            cmd = int(input("Move? (0, 1, 2) "))
            state, reward, done, _, info = e.step(cmd)
            print(state, reward, info["violation"])
            e.render()
            if done:
                e.reset()
        except EOFError:
            break
