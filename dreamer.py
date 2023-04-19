from collections import defaultdict
import os
import time
import numpy as np
import yaml
from pathlib import Path

import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from config import parser
from environments.env import Env, EnvBatcher
from environments.memory import ExperienceReplay
from agent.models import (
    bottle,
    Encoder,
    ObservationModel,
    RewardModel,
    TransitionModel,
    ValueModel,
    ActorModel,
    ViolationModel,
)
from shields.bps import BoundedPrescienceShield, ShieldBatcher
from utils import (
    lineplot,
    write_video,
    imagine_ahead,
    lambda_return,
    FreezeParameters,
    ActivateParameters,
)


class Dreamer:
    def __init__(self, config, env, results_dir, metrics, writer) -> None:
        self._c = config
        self._env = env

        self._metrics = metrics
        self._results_dir = results_dir
        self._writer = writer
        self._losses = []

        self._build_models()

    def build_initial_params(self, batch_size):
        belief = torch.zeros(batch_size, self._c.belief_size)
        state = torch.zeros(batch_size, self._c.state_size)

        return belief, state

    def load_models(self, models_path=""):
        p = self._c.models if models_path == "" else models_path

        if not os.path.exists(p):
            raise Exception("Model path does not exist")

        model_dicts = torch.load(p)
        self.transition_model.load_state_dict(
            model_dicts["transition_model"], strict=True
        )
        self.observation_model.load_state_dict(model_dicts["observation_model"])
        self.reward_model.load_state_dict(model_dicts["reward_model"])
        self.violation_model.load_state_dict(model_dicts["violation_model"])
        self.encoder.load_state_dict(model_dicts["encoder"])
        self.actor_model.load_state_dict(model_dicts["actor_model"])
        self.value_model.load_state_dict(model_dicts["value_model"])

        return model_dicts

    def _build_models(self) -> None:
        self.transition_model = TransitionModel(
            self._c.belief_size,
            self._c.state_size,
            self._env.action_size,
            self._c.hidden_size,
            self._c.embedding_size,
            self._c.dense_activation_function,
        )
        self.observation_model = ObservationModel(
            self._env.observation_size,
            self._c.belief_size,
            self._c.state_size,
            self._c.embedding_size,
            self._c.cnn_activation_function,
        )
        self.reward_model = RewardModel(
            self._c.belief_size,
            self._c.state_size,
            self._c.hidden_size,
            self._c.dense_activation_function,
        )
        self.violation_model = ViolationModel(
            self._c.belief_size,
            self._c.state_size,
            self._c.hidden_size,
            self._c.dense_activation_function,
        )
        self.encoder = Encoder(
            self._env.observation_size,
            self._c.embedding_size,
            self._c.cnn_activation_function,
        )
        self.actor_model = ActorModel(
            self._c.belief_size,
            self._c.state_size,
            self._c.hidden_size,
            self._env.action_size,
            self._c.dense_activation_function,
        )
        self.value_model = ValueModel(
            self._c.belief_size,
            self._c.state_size,
            self._c.hidden_size,
            self._c.dense_activation_function,
        )

        if self._c.learning_rate_schedule != 0:
            lr = 0
        else:
            lr = self._c.model_learning_rate

        self.model_opt = optim.Adam(
            list(self.transition_model.parameters())
            + list(self.observation_model.parameters())
            + list(self.reward_model.parameters())
            + list(self.violation_model.parameters())
            + list(self.encoder.parameters()),
            lr=self._c.model_learning_rate,
            eps=self._c.adam_epsilon,
        )

        self.actor_opt = optim.Adam(
            self.actor_model.parameters(),
            lr=self._c.actor_learning_rate,
            eps=self._c.adam_epsilon,
        )

        self.value_opt = optim.Adam(
            self.value_model.parameters(),
            lr=self._c.value_learning_rate,
            eps=self._c.adam_epsilon,
        )

        self.model_modules = (
            self.transition_model.modules
            + self.encoder.modules
            + self.observation_model.modules
            + self.reward_model.modules
            + self.violation_model.modules
        )

        self.model_params = (
            list(self.transition_model.parameters())
            + list(self.observation_model.parameters())
            + list(self.reward_model.parameters())
            + list(self.violation_model.parameters())
            + list(self.encoder.parameters())
        )

    def set_eval(self):
        self.transition_model.eval()
        self.reward_model.eval()
        self.observation_model.eval()
        self.violation_model.eval()
        self.encoder.eval()
        self.actor_model.eval()
        self.value_model.eval()

    def set_train(self):
        self.transition_model.train()
        self.reward_model.train()
        self.observation_model.train()
        self.violation_model.train()
        self.encoder.train()
        self.actor_model.train()
        self.value_model.train()

    def write_summaries(self):
        s = self._metrics["steps"][-1]

        self._writer.add_scalar(
            "train/episode_reward",
            self._metrics["train_rewards"][-1],
            s * self._c.action_repeat,
        )
        self._writer.add_scalar("train_reward", self._metrics["train_rewards"][-1], s)

        for metric in [
            "observation_loss",
            "reward_loss",
            "kl_loss",
            "actor_loss",
            "value_loss",
            "violation_loss",
        ]:
            agg = torch.mean(torch.tensor(self._metrics[metric][-1]))
            self._writer.add_scalar(metric, agg.item(), s)

    def checkpoint(self, path):
        torch.save(
            {
                "transition_model": self.transition_model.state_dict(),
                "observation_model": self.observation_model.state_dict(),
                "reward_model": self.reward_model.state_dict(),
                "violation_model": self.violation_model.state_dict(),
                "encoder": self.encoder.state_dict(),
                "actor_model": self.actor_model.state_dict(),
                "value_model": self.value_model.state_dict(),
                "model_optimizer": self.model_opt.state_dict(),
                "actor_optimizer": self.actor_opt.state_dict(),
                "value_optimizer": self.value_opt.state_dict(),
            },
            path,
        )

    def build_shield(self):
        shield = BoundedPrescienceShield(
            self.transition_model,
            self.violation_model,
            violation_threshold=self._c.violation_threshold,
            paths_to_sample=self._c.paths_to_sample,
        )
        return shield

    def train(self, obs, actions, rewards, violations, nonterminals):
        init_belief, init_state = self.build_initial_params(self._c.batch_size)
        embedded_observations = bottle(self.encoder, (obs[1:],))
        # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
        (
            beliefs,
            prior_states,
            prior_means,
            prior_std_devs,
            posterior_states,
            posterior_means,
            posterior_std_devs,
        ) = self.transition_model(
            init_state,
            actions[:-1],
            init_belief,
            embedded_observations,
            nonterminals[:-1],
        )

        # Calculate observation likelihood, reward likelihood and KL losses (for t = 0 only for latent overshooting); sum over final dims, average over batch and time (original implementation, though paper seems to miss 1/T scaling?)
        observation_loss = (
            F.mse_loss(
                bottle(self.observation_model, (beliefs, posterior_states)),
                obs[1:],
                reduction="none",
            )
            .sum(dim=2)
            .mean(dim=(0, 1))
        )

        reward_loss = F.mse_loss(
            bottle(self.reward_model, (beliefs, posterior_states)),
            rewards[:-1],
            reduction="none",
        ).mean(dim=(0, 1))

        violation_loss = F.cross_entropy(
            bottle(
                self.violation_model,
                (
                    beliefs,
                    posterior_states,
                ),
            ).reshape(len(violations[:-1]) * len(violations), 2),
            violations[:-1].reshape(len(violations[:-1]) * len(violations)),
            reduction="none",
        ).mean()

        kl_loss = 0.8 * kl_divergence(
            Normal(posterior_means.detach(), posterior_std_devs.detach()),
            Normal(prior_means, prior_std_devs),
        ).sum(dim=2)
        kl_loss += 0.2 * kl_divergence(
            Normal(posterior_means, posterior_std_devs),
            Normal(prior_means.detach(), prior_std_devs.detach()),
        ).sum(dim=2)
        kl_loss = kl_loss.mean(dim=(0, 1))
        # kl_loss = torch.max(div, free_nats).mean(dim=(0, 1))  # Note that normalisation by overshooting distance and weighting by overshooting distance cancel out

        # Global prior N(0, I)
        global_prior = Normal(
            torch.zeros(self._c.batch_size, self._c.state_size),
            torch.ones(self._c.batch_size, self._c.state_size),
        )
        if self._c.global_kl_beta != 0:
            kl_loss += self._c.global_kl_beta * kl_divergence(
                Normal(posterior_means, posterior_std_devs), global_prior
            ).sum(dim=2).mean(dim=(0, 1))

        model_loss = observation_loss + reward_loss + kl_loss + violation_loss
        # Update model parameters
        self.model_opt.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(self.model_params, self._c.grad_clip_norm, norm_type=2)
        self.model_opt.step()

        # Dreamer implementation: actor loss calculation and optimization
        with torch.no_grad():
            actor_states = posterior_states.detach()
            actor_beliefs = beliefs.detach()
        with FreezeParameters(self.model_modules):
            imagination_traj = imagine_ahead(
                actor_states,
                actor_beliefs,
                self.actor_model,
                self.transition_model,
                self._c.planning_horizon,
            )
        (
            imged_beliefs,
            imged_prior_states,
            imged_prior_means,
            imged_prior_std_devs,
        ) = imagination_traj
        with FreezeParameters(self.model_modules + self.value_model.modules):
            imged_reward = bottle(
                self.reward_model, (imged_beliefs, imged_prior_states)
            )
            value_pred = bottle(self.value_model, (imged_beliefs, imged_prior_states))
        returns = lambda_return(
            imged_reward,
            value_pred,
            bootstrap=value_pred[-1],
            discount=self._c.discount,
            lambda_=self._c.disclam,
        )
        actor_loss = -torch.mean(returns)
        # Update model parameters
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            self.actor_model.parameters(), self._c.grad_clip_norm, norm_type=2
        )
        self.actor_opt.step()

        # Dreamer implementation: value loss calculation and optimization
        with torch.no_grad():
            value_beliefs = imged_beliefs.detach()
            value_prior_states = imged_prior_states.detach()
            target_return = returns.detach()
        # detach the input tensor from the transition network.
        value_dist = Normal(
            bottle(self.value_model, (value_beliefs, value_prior_states)), 1
        )
        value_loss = -value_dist.log_prob(target_return).mean(dim=(0, 1))
        # Update model parameters
        self.value_opt.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(
            self.value_model.parameters(), self._c.grad_clip_norm, norm_type=2
        )
        self.value_opt.step()

        return (
            observation_loss.item(),
            reward_loss.item(),
            kl_loss.item(),
            actor_loss.item(),
            value_loss.item(),
            violation_loss.item(),
        )

    def policy(
        self,
        belief,
        state,
        action,
        observation,
        explore=False,
    ):
        # Infer belief over current state q(s_t|o≤t,a<t) from the history
        # Action and observation need extra time dimension
        belief, _, _, _, posterior_state, _, _ = self.transition_model(
            state,
            action.unsqueeze(dim=0),
            belief,
            self.encoder(observation).unsqueeze(dim=0),
        )

        imagd_violation = torch.argmax(
            bottle(self.violation_model, (belief, posterior_state)).squeeze()
        )

        # Remove time dimension
        belief = belief.squeeze(dim=0)
        posterior_state = posterior_state.squeeze(0)

        action = self.actor_model.get_action(belief, posterior_state, det=not (explore))

        if explore:
            # Add gaussian exploration noise on top of the sampled action
            action = torch.clamp(
                Normal(action.float(), self._c.action_noise).rsample(), -1, 1
            )

        return belief, posterior_state, action
