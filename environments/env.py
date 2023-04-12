import torch
import gymnasium
from minigrid.wrappers import ImgObsWrapper

from safety_environments import MINIGRID_ENVS
from safety_environments.wrappers import ActionWeights


def Env(env, max_episode_steps, action_repeat, bit_depth, render=False):
    assert env in MINIGRID_ENVS

    render_mode = "human" if render else None
    env = gymnasium.make(
        env,
        action_repeat=action_repeat,
        bit_depth=bit_depth,
        max_episode_steps=max_episode_steps,
        render_mode=render_mode,
    )
    env = ActionWeights(env)
    env = ImgObsWrapper(env)
    return env


# Wrapper for batching environments together
class EnvBatcher:
    def __init__(self, env_class, env_args, env_kwargs, n):
        self.n = n
        self.envs = [env_class(*env_args, **env_kwargs) for _ in range(n)]
        self.dones = [True] * n

    # Resets every environment and returns observation
    def reset(self):
        observations = [
            torch.tensor(env.reset()[0]).to(torch.float32) for env in self.envs
        ]
        self.dones = [False] * self.n
        return torch.cat(observations)

    # Steps/resets every environment and returns (observation, reward, done)
    def step(self, actions):
        # Done mask to blank out observations and zero rewards for previously terminated environments
        done_mask = torch.nonzero(torch.tensor(self.dones))[:, 0]

        observations, rewards, dones, _, infos = zip(
            *[env.step(action) for env, action in zip(self.envs, actions)]
        )

        violations = [info["violation"] for info in infos]
        # Env should remain terminated if previously terminated
        dones = [d or prev_d for d, prev_d in zip(dones, self.dones)]
        self.dones = dones
        observations, rewards, violations, dones = (
            torch.cat(observations),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(violations, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.uint8),
        )
        observations[done_mask] = 0
        rewards[done_mask] = 0
        violations[done_mask] = 0
        return observations, rewards, violations, dones

    def close(self):
        [env.close() for env in self.envs]
