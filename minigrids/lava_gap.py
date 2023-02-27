from enum import IntEnum
from random import Random
from typing import Tuple

import gymnasium
import numpy as np

class LavaGapMiniGridAction(IntEnum):
    LEFT = 0
    RIGHT = 1
    FORWARD = 2

class LavaGapMiniGrid:
    actions = LavaGapMiniGridAction
    def __init__(self, size=(10, 10), fixed_seed=None, render_size=(256, 256)):
        self.size = size
        # Rendering
        self._viewer = None
        self._render_size = render_size
        # Generate world
        self.fixed_seed = fixed_seed
        self._rng = Random(fixed_seed)
        self._env = gymnasium.make('MiniGrid-LavaGapS5-v0', render_mode='rgb_array')

    def reset(self):
        self._rng = Random(self.fixed_seed)
        self._env.reset()

    def step(self, action: LavaGapMiniGridAction) -> Tuple[float, int]:
        return self._env.step(action)

    def render(self) -> np.ndarray:
        return self._env.render()

    def close(self):
        self._env.close()

if __name__ == '__main__':
    g = LavaGapMiniGrid()
    g.reset()
    g.render()
    while True:
        action = input(f'Choose an action: {list(LavaGapMiniGridAction)}\n')
        if action == 'exit':
            exit()
        
        try:
            action = LavaGapMiniGridAction[action.upper()]
            reward = g.step(action)
            g.render()
            print(f'Received reward of {reward}')
        except Exception as e:
            print(e)
