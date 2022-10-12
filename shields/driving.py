import random
import numpy as np
import torch
from .shield import Shield

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

class DrivingShield(Shield):
    def __init__(self, env):
        self._env = env._env
        self._discrete_actions = [-1, 1]
        self._lookahead = 6

    def step(self, action):
        if len(action.shape) == 2 and action.shape[0] > 1:
            raise Exception('Maybe try using a ShieldBatcher')
        had_to_interfere = False
        state, speed = self._env._state, self._env._speed
        if self.__is_unsafe__(action, state, speed, self._lookahead):
            action = torch.tensor([[-1]])
            had_to_interfere = True
        return action, had_to_interfere

    def __is_unsafe__(self, action, state, speed, depth):
        if depth == 0:
            return False
        discrete_action = find_nearest(self._discrete_actions, action.cpu().item()) / 10
        acceleration = discrete_action if discrete_action > 0 else 1.5 * discrete_action
        next_speed = max(0, speed + acceleration)
        next_state = max(0, state - next_speed)
        next_state_safe = next_state > 0
        exists_safe_path = False
        for da in self._discrete_actions:
            exists_safe_path = exists_safe_path or not self.__is_unsafe__(torch.tensor(da), next_state, next_speed, depth-1)
        return (not next_state_safe or not exists_safe_path)



class ShieldBatcher(Shield):

    def __init__(self, shield_class, envs):
        self._envs = envs.envs
        self._shields = [shield_class(env) for env in self._envs]

    def step(self, actions):
        safe_actions = []
        had_to_interfere = [0 for i in range(len(actions))]
        for i, action in enumerate(actions.split(1)):
            safe_action, interfered = self._shields[i].step(action)
            safe_actions.append(safe_action.cpu())
            had_to_interfere[i] = 1 if interfered else 0
        return torch.cat(safe_actions, 0), had_to_interfere

if __name__ == '__main__':
    class test:
        def __init__(self):
            self._speed = 1.1033
            self._state = 1.2752

    class testwrap:
        def __init__(self):
            self._env = test()

    ds = DrivingShield(testwrap())
    print(ds.step(torch.tensor([0.0385])))