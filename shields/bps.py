from agent.models import ModelGroup, TransitionModel, APModel
import torch
import torch.nn.functional as F

from .shield import Shield


class BoundedPrescienceShield(Shield):
    def __init__(
        self,
        models: ModelGroup,
        depth=5,
        violation_threshold=3,
        paths_to_sample=20,
        exploration_weight=5,
        discount=0.5,
        sensitivity=2,
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

    def step(self, belief, state, actions, t):
        action_size = actions.size(1)

        all_actions = F.one_hot(
            torch.arange(0, action_size), num_classes=action_size
        ).reshape(action_size, 1, action_size)

        beliefs, states, _, _ = self.transition_model(state, all_actions, belief)

        safes = []
        beliefs, states = (beliefs.squeeze(0), states.squeeze(0))
        for a in range(action_size):
            belief, state = beliefs[a], states[a]
            safes.append(self._is_safe(belief, state, self.paths_to_sample))

        if not any(safes):
            return actions, False

        safes = torch.tensor(safes, dtype=torch.bool).unsqueeze(0)
        mask = torch.full_like(actions, actions.min() - 1)
        
        return torch.where(safes, actions, mask), True

    def _is_terminal(self, belief, state):
        reach_goal = self.ap_model.forward_goal(belief, state)
        return torch.all(reach_goal > 0.5).item()

    def _is_safe(self, belief, state, n):
        traj_violations = [self._simulate(belief, state) for _ in range(n)]
        num_safe = [int(v == 0) for v in traj_violations]
        return sum(num_safe) >= self.violation_threshold

    def _simulate(self, belief, state):
        traj = [(belief, state)]
        initial_violation = self.ap_model.forward_violation(belief, state).squeeze()
        violations = [(initial_violation > 0.5).sum().item()]

        cur_depth = 1
        while cur_depth < self.depth and not self._is_terminal(*traj[-1]):
            cur_depth += 1
            cur_belief, cur_state = traj[-1]

            action, _ = self.actor_model.get_action(cur_belief, cur_state, det=False)
            action = F.one_hot(torch.argmax(action, dim=1), num_classes=action.size(1))

            next_belief, next_state, _, _ = self.transition_model(
                cur_state, action.unsqueeze(0), cur_belief
            )
            next_belief = next_belief.squeeze(0)
            next_state = next_state.squeeze(0)

            violation = self.ap_model.forward_violation(
                next_belief, next_state
            ).squeeze()
            traj.append((next_belief, next_state))
            violations.append((violation > 0.5).sum().item())

        return sum(violations)
