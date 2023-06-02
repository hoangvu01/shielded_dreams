import math
from collections import defaultdict
from agent.models import (
    ActorModel,
    SymbolicObservationModel,
    TransitionModel,
    AtomicModel,
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
        transition_model,
        violation_model,
        actor_model,
        observation_model,
    ) -> None:
        self.belief = belief
        self.state = state

        self.transition_model = transition_model
        self.violation_model = violation_model
        self.actor_model = actor_model
        self.observation_model = observation_model

    def is_terminal(self):
        reach_goal = (
            self.violation_model.forward_goal(self.belief, self.state).squeeze().item()
        )
        return reach_goal > 0.5

    def find_random_child(self):
        actions, _ = self.actor_model.get_action(self.belief, self.state, det=False)
        action = F.one_hot(actions.argmax(), num_classes=actions.size(1))

        belief, state, _, _ = self.transition_model(
            self.state, action.reshape(1, 1, -1), self.belief
        )

        return action, Node(
            belief.squeeze(0),
            state.squeeze(0),
            self.transition_model,
            self.violation_model,
            self.actor_model,
            self.observation_model,
        )

    def find_children(self):
        if self.is_terminal():
            return []

        action_size = 3
        all_actions = F.one_hot(torch.arange(0, action_size), num_classes=action_size)

        beliefs, states, _, _ = zip(
            *[
                self.transition_model(
                    self.state,
                    all_actions[a].reshape(1, 1, -1),
                    self.belief,
                )
                for a in range(action_size)
            ]
        )

        return [
            Node(
                b.squeeze(0),
                s.squeeze(0),
                self.transition_model,
                self.violation_model,
                self.actor_model,
                self.observation_model,
            )
            for b, s in zip(beliefs, states)
        ]

    def reward(self):
        violations = self.violation_model.forward_violation(
            self.belief, self.state
        ).sum()
        reach_goal = self.violation_model.forward_goal(self.belief, self.state).sum()

        _, entropies = self.actor_model.get_action(self.belief, self.state, det=False)
        entropy = entropies.sum()

        std = self.observation_model.std(self.belief, self.state)
        uncertainty = torch.linalg.vector_norm(std)

        return violations.item(), entropy.item(), uncertainty.item(), reach_goal.item()

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Node):
            return self.belief == __value.belief and self.state == __value.state

    def __hash__(self) -> int:
        return hash((self.belief, self.state))


class VanillaShield(Shield):
    def __init__(
        self,
        transition_model: TransitionModel,
        violation_model: AtomicModel,
        actor_model: ActorModel,
        value_model: ValueModel,
        observation_model: SymbolicObservationModel,
        depth=5,
        violation_threshold=1,
        paths_to_sample=1,
        exploration_weight=1,
        discount=0.5,
        sensitivity=0.5,
    ):
        self.transition_model = transition_model
        self.violation_model = violation_model
        self.actor_model = actor_model
        self.value_model = value_model
        self.observation_model = observation_model

        self.depth = depth
        self.violation_threshold = violation_threshold
        self.paths_to_sample = paths_to_sample

        self.N = defaultdict(int)
        self.V = defaultdict(float)
        self.E = defaultdict(float)
        self.U = defaultdict(float)
        self.K = defaultdict(float)
        self.G = defaultdict(int)

        self.children = dict()
        self.exploration_weight = exploration_weight
        self.discount = discount
        self.sensitivity = sensitivity

    def clear(self):
        self.N.clear()
        self.V.clear()
        self.E.clear()
        self.U.clear()
        self.K.clear()
        self.G.clear()

    def step(self, belief, state, action, t):
        "Choose the best successor of node. (Choose a move in the game)"
        node = Node(
            belief,
            state,
            self.transition_model,
            self.violation_model,
            self.actor_model,
            self.observation_model,
        )
        if node.is_terminal():
            return action, False

        for _ in range(self.paths_to_sample):
            self.do_rollout(node)

        if node not in self.children:
            _, node = node.find_random_child()
            return action, False

        scores = [self._score(n) for n in self.children[node]]
        scores = torch.tensor(scores).unsqueeze(0)

        weighted_scores = torch.softmax(scores, dim=1)
        weighted_action = torch.softmax(action, dim=1)

        alpha = max(math.exp(-0.01 * t), 0.25)
        return (
            1 - self.sensitivity
        ) * weighted_action + self.sensitivity * alpha * weighted_scores, True

    def do_rollout(self, node):
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        attrs = self._simulate(leaf)
        self._backpropagate(path, *attrs)

    def _score(self, n):
        novelty = math.exp(-0.1 * self.N[n])
        uncertainty_bonus = 0.1 * self.U[n] / self.N[n]
        safety = -self.V[n] / self.N[n]
        goal_bonus = self.G[n] / self.N[n]
        entropy_bonus = 0.1 * math.exp(self.E[n] / self.N[n])
        # print(novelty, entropy_bonus, uncertainty_bonus, safety, goal_bonus)
        return novelty + entropy_bonus + uncertainty_bonus + safety + goal_bonus

    def _select(self, node):
        path = []
        while len(path) < 10:
            path.append(node)
            if node not in self.children or not self.children[node]:
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._explore_select(node)
        return path

    def _expand(self, node):
        if node in self.children:
            return
        self.children[node] = node.find_children()

    def _simulate(self, root):
        violations = 0
        entropies = 0
        uncert = 0
        goal = 0

        node = root

        num_sims = 1

        for _ in range(num_sims):
            step = 0
            while step < self.depth:
                v, e, u, g = node.reward()
                violations += v * self.discount**step
                uncert += u * self.discount**step
                entropies += e * self.discount**step
                goal += g * self.discount**step

                if node.is_terminal():
                    break

                _, node = node.find_random_child()
                step += 1

        return (
            violations / num_sims,
            entropies / num_sims,
            uncert / num_sims,
            goal / num_sims,
        )

    def _backpropagate(self, path, violation, entropy, uncert, goal):
        e = entropy
        v = violation
        g = goal
        u = uncert

        for node in reversed(path):
            nv, ne, nu, ng = node.reward()

            self.N[node] += 1
            self.V[node] += v
            self.E[node] += e
            self.U[node] += u
            self.K[node] += v * math.exp(-0.1 * ne)
            self.G[node] += g

            e = (ne + e) * self.discount
            v = (nv + v) * self.discount
            g = (ng + g) * self.discount
            u = (nu + u) * self.discount

    def _explore_select(self, node):
        assert all(n in self.children for n in self.children[node])

        def score(n):
            base = self._score(n)
            return base

        return max(self.children[node], key=score)
