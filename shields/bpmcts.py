import math
from collections import defaultdict
from agent.models import ActorModel, TransitionModel, APModel
import torch
import torch.nn.functional as F
from .shield import Shield


class Node:
    def __init__(
        self, belief, state, transition_model, violation_model, actor_model
    ) -> None:
        self.belief = belief
        self.state = state

        self.transition_model = transition_model
        self.violation_model = violation_model
        self.actor_model = actor_model

    def is_terminal(self):
        violations = self.violation_model(self.belief, self.state).squeeze(0)
        reach_goal = violations[-1].item()

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
            )
            for b, s in zip(beliefs, states)
        ]

    def reward(self):
        violations = self.violation_model(self.belief, self.state).squeeze()
        reach_goal = violations[-1] > 0.5
        violations = violations[:-1] > 0.5

        return (reach_goal.int() - violations.sum()).item()

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Node):
            return self.belief == __value.belief and self.state == __value.state

    def __hash__(self) -> int:
        return hash((self.belief, self.state))


class BoundedPrescienceMCTSShield(Shield):
    def __init__(
        self,
        transition_model: TransitionModel,
        violation_model: APModel,
        depth=5,
        violation_threshold=1,
        paths_to_sample=1,
        exploration_weight=2,
        discount=0.5,
    ):
        self.transition_model = transition_model
        self.violation_model = violation_model

        self.depth = depth
        self.violation_threshold = violation_threshold
        self.paths_to_sample = paths_to_sample

        self.Q = defaultdict(int)
        self.N = defaultdict(int)
        self.children = dict()
        self.exploration_weight = exploration_weight
        self.discount = discount

    def step(self, belief, state, action, policy):
        "Choose the best successor of node. (Choose a move in the game)"
        node = Node(belief, state, self.transition_model, self.violation_model, policy)
        if node.is_terminal():
            return action, False

        for _ in range(self.paths_to_sample):
            self.do_rollout(node)

        if node not in self.children:
            _, node = node.find_random_child()
            return action, False

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        scores = [score(n) for n in self.children[node]]
        chosen_action = action.squeeze(0).argmax().item()
        if scores[chosen_action] > self.violation_threshold:
            return action, False

        scores = torch.tensor(scores).unsqueeze(0)
        mask = scores > self.violation_threshold
        if mask.sum().item() == 0:
            return scores, True

        masked_actions = torch.where(mask, action, torch.full_like(action, -10.0))
        return masked_actions, True

    def do_rollout(self, node):
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node):
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)

    def _expand(self, node):
        if node in self.children:
            return
        self.children[node] = node.find_children()

    def _simulate(self, node):
        step = 0
        reward = 0
        while step < self.depth:
            r = node.reward()
            reward += r * self.discount**step

            if node.is_terminal() or r < 0:
                break

            _, node = node.find_random_child()
            step += 1

        return reward

    def _backpropagate(self, path, reward):
        r = reward
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += r
            r = (r + node.reward()) * self.discount

    def _uct_select(self, node):
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)
