import random
import torch
from .shield import Shield


class GridWorldShield(Shield):
    def __init__(self, env):
        self._env = env._env
        self._unsafe_positions = self._env._unsafe_positions
        self._enemies = self._env._enemies

    def step(self, action):
        if len(action.shape) == 2 and action.shape[0] > 1:
            raise Exception('Maybe try using a ShieldBatcher')
        had_to_interfere = False
        if self.__is_unsafe__(action):
            action = self.__find_next_best_safe_action__(action)
            had_to_interfere = True
        return action, had_to_interfere

    def __find_random_safe_action__(self, action):
        action_count = max(action.shape)
        idx = random.randint(0, action_count - 1)
        new_action = torch.tensor(
            [1 if i == idx else -1 for i in range(action_count)]).reshape(action.shape)
        while self.__is_unsafe__(new_action):
            idx = random.randint(0, action_count - 1)
            new_action = torch.tensor(
                [1 if i == idx else -1 for i in range(action_count)]).reshape(action.shape)
        return new_action.float()

    def __find_next_best_safe_action__(self, action, epsilon=0.01):
        dims = len(action.shape)
        # Add epsilon so even the "not preferred" options have a chance
        new_action = action.clone().squeeze().cpu() + epsilon
        idx = new_action.argmax()
        new_action = new_action.clamp(-1, 1)
        # since the normal range is -1 to 1, setting the entry to -1 => it will not be selected
        new_action[idx] = -1
        if max(new_action) == -1:
            # we are dealing with a one-hot vector - just choose a random vector
            return self.__find_random_safe_action__(action)
        # Otherwise, find the next best option
        while self.__is_unsafe__(new_action):
            idx = new_action.argmax()
            new_action[idx] = -1
            if max(new_action) == -1:
                # This should never happen theoretically
                raise Exception('No safe actions')
        new_action = new_action.clamp(-1, 1)
        return new_action if dims == 1 else new_action.unsqueeze(0)

    def __is_unsafe__(self, action):
        x, y = self._env._agent_position
        new_x, new_y = x, y
        action_idx = action.cpu().argmax().item()
        if action_idx == 0:
            pass
        elif action_idx == 1:
            new_y += 1 if y < self._env.size[1] else 0
        elif action_idx == 2:
            new_y -= 1 if y > 0 else 0
        elif action_idx == 3:
            new_x -= 1 if x > 0 else 0
        elif action_idx == 4:
            new_x += 1 if x < self._env.size[0] else 0
        return (new_x, new_y) in self._unsafe_positions


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
