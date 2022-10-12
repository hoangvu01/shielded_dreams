import torch
import random

from enum import Enum
from typing import Tuple

class CarAction(Enum):
    ACCELERATE = 0
    BRAKE = 1
    NOP = 2


class CarEnvironment:
    """
    A simple symbolic environment.
    """

    def __init__(self, length=10, p_sticky=0.1):
        self._length = length
        self._starting_state = length
        self._state = self._starting_state
        self._speed = 0
        self._friction = 1
        self._p_sticky = p_sticky
        self._got_reward = False
        self._prev_action = 0

    def reset(self):
        self._state = self._starting_state
        self._got_reward = False
        self._prev_action = 0
        print('RESET', self._p_sticky)

    def step(self, action) -> Tuple[float, int]:
        action = action / 10
        print(action, self._state, self._speed)
        if self._state <= 0:
            self._state = 0
            self._speed = 0
            return 0, 1, False
        else:
            if random.random() < self._p_sticky:
                action = self._prev_action
            acceleration = action if action > 0 else 1.5 * action
            self._speed = max(0, self._speed + acceleration)
            self._state = max(0, self._state - self._speed)
            self._prev_action = action
            # if self._state <= 3:
            #     # if not self._got_reward:
            #     #     self._got_reward = True
            #     #     return 500, 0, False
            #     # else:
            #     return 1, 0, False
            return float((1 - (self._state / self._length))), 0, False


    def render(self):
        return torch.tensor([[self._state / self._length, self._speed]]).float()

    def close(self):
        pass


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    g = CarEnvironment()
    g.render()
    while True:
        action = input('Pls give an action: ')
        if action == 'exit':
            exit()
        reward = g.step(CarAction(int(action)))
        print(f'Received reward of {reward}')
        print(g.render())
