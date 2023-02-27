import torch
import random

from .utils import images_to_observation

class GridEnv():
    def __init__(self, env, symbolic, seed, max_episode_length, action_repeat, bit_depth):
        from gridworld.gridworld import GridWorld, GridWorldAction

        self.max_episode_length = max_episode_length
        self.bit_depth = bit_depth
        self._env = GridWorld((3, 3), fixed_seed=seed, no_enemies=0)
        self._t = 0

        self._actions = list(GridWorldAction)
        # A list of one-hot tensors representing actions
        self._action_tensors = [
            torch.tensor(
                [1 if i == j else -1 for i in range(len(self._actions))])
            for j in range(len(self._actions))
        ]  # Can't use a dictionary unfortunately because tensor equality is screwed

    def reset(self):
        self._env.reset()
        self._t = 0
        return images_to_observation(self._env.render(), self.bit_depth)

    def step(self, action):
        decoded_action = self.__tensor_to_action__(action)
        # Do action
        reward, violation = self._env.step(decoded_action)
        observation = images_to_observation(
            self._env.render(), self.bit_depth)
        self._t += 1
        done = self._t == self.max_episode_length
        if done:
            print("DONE")
        return observation, reward, violation, done

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    @property
    def observation_size(self):
        return (3, 64, 64)

    @property
    def action_size(self):
        return len(self._actions)

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        return random.choice(self._action_tensors)

    # Convert a tensor into a GridWorldAction
    def __tensor_to_action__(self, t):
        # Assume action to be in 1-hot representation
        idx = t.cpu().argmax()
        # Find GridWorldAction
        return self._actions[idx.item()]

