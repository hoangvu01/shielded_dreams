import torch

class DrivingEnv():
    def __init__(self, env, symbolic, seed, max_episode_length, action_repeat, bit_depth, stickiness):
        from driving.driving import CarEnvironment, CarAction
        self.max_episode_length = max_episode_length
        self._env = CarEnvironment(p_sticky=stickiness)
        self._t = 0
        self._actions = list(CarAction)
        # A list of one-hot tensors representing actions
        self._action_tensors = [
            torch.tensor(
                [1 if i == j else -1 for i in range(len(self._actions))])
            for j in range(len(self._actions))
        ]  # Can't use a dictionary unfortunately because tensor equality is screwed

    def reset(self):
        self._env.reset()
        self._t = 0
        return self._env.render()

    def step(self, action):
        # decoded_action = self.__tensor_to_action__(action)
        # Do action
        reward, violation, done = self._env.step(action)
        observation = torch.tensor(self._env.render(), dtype=torch.float32)
        self._t += 1
        done = self._t == self.max_episode_length or done
        return observation, reward, violation, done

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    @property
    def observation_size(self):
        return 2

    @property
    def action_size(self):
        return 1

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        return torch.tensor(random.random() * 2 - 1)

    # Convert a tensor into a MountainOfDeathAction
    def __tensor_to_action__(self, t):
        # Assume action to be in 1-hot representation
        idx = t.cpu().argmax()
        # Find MountainOfDeathAction
        return self._actions[idx.item()]



