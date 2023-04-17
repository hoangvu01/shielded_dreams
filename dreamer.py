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
    def __init__(self, config, env) -> None:
        self._c = config
        self._metrics = {}
        self._env = env

        self._build_models()

    def load_models(self, models_path=""):
        p = self._c.models if models_path == "" else models_path

        if not os.path.exists(p):
            raise Exception("Model path does not exist")

        print(p)

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
            lr=lr,
            eps=self._c.adam_epsilon,
        )

        self.actor_opt = optim.Adam(
            self.actor_model.parameters(),
            lr=lr,
            eps=self._c.adam_epsilon,
        )

        self.value_opt = optim.Adam(
            self.value_model.parameters(),
            lr=lr,
            eps=self._c.adam_epsilon,
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

    def build_shield(self):
        shield = BoundedPrescienceShield(
            self.transition_model,
            self.violation_model,
            violation_threshold=self._c.violation_threshold,
            paths_to_sample=self._c.paths_to_sample,
        )
        return shield

    def train(self, data):
        pass

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
