import math
from collections import defaultdict
from agent.models import (
    APModel,
    ActorModel,
    ModelGroup,
    SymbolicObservationModel,
    TransitionModel,
    ValueModel,
)
import torch
import torch.nn.functional as F
from .shield import Shield


class Node:
    def __init__(
        self,
        belief,
        state,
        models,
        action_size=3,
    ) -> None:
        self.belief = belief
        self.state = state

        self.transition_model = models.transition_model
        self.ap_model = models.ap_model
        self.actor_model = models.actor_model
        self.value_model = models.value_model
        self.observation_model = models.observation_model
        self.reward_model = models.reward_model
        self.models = models

        self.expanded = False
        self.children = []
        self.children_uncertainty = []

        self.N_sa = torch.zeros(1, action_size)
        self.Q_sa = torch.zeros(1, action_size)
        self.uncertainty = torch.zeros(1, action_size)

    def increment_action(self, a):
        self.N_sa[0, a] += 1

    def update_action(self, a, value, violation, uncertainty, reward):
        # print(a, value, violation, uncertainty)
        self.Q_sa[0, a] += violation + reward
        self.uncertainty[0, a] += uncertainty

    def is_terminal(self):
        reach_goal = (
            self.ap_model.forward_goal(self.belief, self.state).squeeze().item()
        )
        return reach_goal > 0.5

    def find_random_child(self, sensitivity=1, exploration_weight=1):
        p_sa, _ = self.actor_model.get_action(self.belief, self.state)
        p_sa = torch.softmax(p_sa, dim=1)

        if len(self.children) == 0:
            v_sa = torch.zeros_like(p_sa)
        else:
            v_sa = torch.tensor([n.value() for n in self.children]).reshape(1, -1)

        q_sa = torch.nan_to_num(self.Q_sa / self.N_sa) + v_sa
        q_sa = torch.softmax(q_sa * sensitivity, dim=1)

        N = torch.log(2 + torch.sum(self.N_sa)).item()
        u_sa = exploration_weight * p_sa * torch.sqrt(1.5 * N / (self.N_sa + 1))

        puct = q_sa + u_sa

        actions = puct
        action = F.one_hot(actions.argmax(), num_classes=actions.size(1))
        belief, state, _, _ = self.transition_model(
            self.state, action.reshape(1, 1, -1), self.belief
        )

        _, action_std = self.actor_model(self.belief, self.state)
        action_std = action_std.squeeze(0)

        return (
            action,
            action_std[actions.argmax()].item(),
            Node(belief.squeeze(0), state.squeeze(0), self.models),
        )

    def expand(self):
        if self.expanded:
            return

        self.expanded = True
        action_size = 3
        all_actions = F.one_hot(
            torch.arange(0, action_size), num_classes=action_size
        ).reshape(action_size, 1, action_size)

        beliefs, states, _, _ = self.transition_model(
            self.state,
            all_actions,
            self.belief,
        )

        _, action_stds = self.actor_model(self.belief, self.state)

        beliefs, states, stds = (
            beliefs.squeeze(0),
            states.squeeze(0),
            action_stds.squeeze(0),
        )
        for a in range(action_size):
            belief, state, std = beliefs[a], states[a], stds[a]
            self.children.append(Node(belief, state, self.models))
            self.children_uncertainty.append(std.item())

    def value(self):
        return self.value_model(self.belief, self.state).sum().item()

    def violation(self):
        return self.ap_model.forward_violation(self.belief, self.state).sum().item()

    def reward(self):
        return self.reward_model(self.belief, self.state).sum().item()


class RaisinShield(Shield):
    def __init__(
        self,
        models: ModelGroup,
        depth=5,
        violation_threshold=1,
        paths_to_sample=1,
        exploration_weight=3,
        discount=0.5,
        sensitivity=3,
    ):
        self.transition_model = models.transition_model
        self.ap_model = models.ap_model
        self.actor_model = models.actor_model
        self.value_model = models.value_model
        self.observation_model = models.observation_model
        self.models = models

        self.depth = depth
        self.violation_threshold = violation_threshold
        self.paths_to_sample = paths_to_sample

        self.exploration_weight_init = exploration_weight
        self.discount = discount
        self.sensitivity = sensitivity

        self.nodes = dict()

    def step(self, belief, state, action, t):
        "Choose the best successor of node. (Choose a move in the game)"
        k = (belief, state)
        if k in self.nodes:
            node = self.nodes[k]
        else:
            node = Node(belief, state, self.models)
            self.nodes[k] = node

        if node.is_terminal():
            return action, False

        self.exploration_weight = self.exploration_weight_init * math.exp(-0.001 * t)
        for _ in range(self.paths_to_sample):
            self.do_rollout(node)

        action, _, _ = node.find_random_child(sensitivity=self.sensitivity)
        return action.unsqueeze(0), False

    def do_rollout(self, node):
        path = self._select(node)
        leaf, _, _ = path[-1]
        leaf.expand()
        attrs = self._simulate(leaf)
        self._backpropagate(path, *attrs)

    def _select(self, node):
        path = []

        action = None
        while len(path) < 10:
            if not node.expanded or node.is_terminal():
                path.append((node, None, None))
                return path

            action, std, child = node.find_random_child(
                sensitivity=self.sensitivity, exploration_weight=self.exploration_weight
            )
            action = action.argmax().item()
            node.increment_action(action)
            path.append((node, std, action))

            node = child
        return path

    def _simulate(self, node):
        s = 0
        v = 0
        u = 0
        step = 0
        r = 0

        while step < self.depth:
            violation, value, reward = node.violation(), node.value(), node.reward()
            s += value * self.discount**step
            v += violation * self.discount**step
            r += reward * self.discount**step
            if node.is_terminal():
                s += self.discount**step
                break

            _, std, node = node.find_random_child(
                sensitivity=self.sensitivity, exploration_weight=self.exploration_weight
            )

            u += std * self.discount**step
            step += 1

        return s, v, u, r

    def _backpropagate(self, path, value, violation, uncertainty, reward):
        s = value
        v = violation
        u = uncertainty
        r = reward

        for node, std, action in reversed(path):
            if action is None:
                continue

            nv, ns, nr = node.violation(), node.value(), node.reward()

            node.update_action(action, s, v, u, r)

            s = (s + ns) * self.discount
            v = (v + nv) * self.discount
            r = (r + nv) * self.discount
            u = (u + std) * self.discount
