from agent.models import TransitionModel, ViolationModel
import torch
import torch.nn.functional as F

from .shield import Shield


class BoundedPrescienceShield(Shield):
    def __init__(
        self,
        transition_model: TransitionModel,
        violation_model: ViolationModel,
        depth=1,
        violation_threshold=1,
        paths_to_sample=1,
    ):
        self.transition_model = transition_model
        self.violation_model = violation_model
        self.depth = depth
        self.violation_threshold = violation_threshold
        self.paths_to_sample = paths_to_sample

    def step(self, belief, state, actions, policy):
        action_size = actions.size(1)
        action = actions.argmax()

        all_actions = F.one_hot(torch.arange(0, action_size), num_classes=action_size)
        beliefs, states, _, _ = zip(
            *[
                self.transition_model(state, all_actions[a].reshape(1, 1, -1), belief)
                for a in range(action_size)
            ]
        )

        violations = torch.tensor(
            [
                self._simulate_n(
                    beliefs[a].squeeze(0),
                    states[a].squeeze(0),
                    policy,
                    self.paths_to_sample,
                )
                for a in range(action_size)
            ]
        )

        # Acceptable number of violations
        shield_interfered = violations[action].item() >= self.violation_threshold

        violations = violations.unsqueeze(0)
        if torch.all(violations > self.violation_threshold):
            return 1 / (violations + 1), shield_interfered

        return violations * (violations < self.violation_threshold), shield_interfered

    def _is_terminal(self, belief, state):
        violations = self.violation_model(belief, state)
        reach_goal = violations[-1]

        return torch.all(reach_goal > 0.5).item()

    def _simulate_n(self, belief, state, policy, n):
        _, vios = zip(*[self._simulate(belief, state, policy) for _ in range(n)])
        violations_per_traj = [sum(vj) for vj in vios]
        return max(violations_per_traj)

    def _simulate(self, belief, state, policy):
        traj = [(belief, state)]
        initial_violation = self.violation_model(belief, state).squeeze()[:-1]
        violations = [(initial_violation > 0.5).sum().item()]

        cur_depth = 1
        num_backtrack = 0
        while cur_depth < self.depth and not self._is_terminal(*traj[-1]):
            cur_depth += 1
            cur_belief, cur_state = traj[-1]

            action, _ = policy.get_action(cur_belief, cur_state, det=False)
            action = F.one_hot(torch.argmax(action, dim=1), num_classes=action.size(1))
            # print(cur_belief, cur_state, action)

            next_belief, next_state, _, _ = self.transition_model(
                cur_state, action.unsqueeze(0), cur_belief
            )
            next_belief = next_belief.squeeze(0)
            next_state = next_state.squeeze(0)

            violation = self.violation_model(next_belief, next_state).squeeze()[:-1]
            if torch.all(violation < 0.5) or num_backtrack > 5:
                traj.append((next_belief, next_state))
                violations.append((violation > 0.5).sum().item())
                num_backtrack = 0
            else:
                num_backtrack += 1

        return traj[:-1], violations
