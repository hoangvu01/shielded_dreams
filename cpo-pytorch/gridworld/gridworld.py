import numpy as np

from enum import Enum
from random import Random
from typing import Tuple

from .monitor import SafetyMonitor


class GridWorldAction(Enum):
    NONE = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class Enemy:
    """
    An enemy that patrols in a square with some radius
    """

    def __init__(self, start_position, radius=2):
        waypoints = []
        # Generate waypoints for a square of movement
        start_x, start_y = start_position
        for i in range(radius):
            waypoints.append((start_x + i, start_y))
        for i in range(radius):
            waypoints.append((start_x + radius, start_y - i))
        for i in range(radius):
            waypoints.append((start_x + radius - i, start_y - radius))
        for i in range(radius):
            waypoints.append((start_x, start_y - radius + i))
        self.position = start_position
        # keep track of this so we can penalise the agent and enemy swapping places
        self.previous_position = start_position
        self._waypoints = waypoints
        self._counter = 0

    def step(self):
        self._counter += 1
        if self._counter == len(self._waypoints):
            self._counter = 0
        self.previous_position = self.position
        self.position = self._waypoints[self._counter]


class GridWorld:

    def __init__(self, size=(10, 10), fixed_seed=None, render_size=(256, 256), no_unsafe_positions=2, no_enemies=1):
        self.size = size
        # Rendering
        self._viewer = None
        self._render_size = render_size
        # Generate world
        self._rng = Random() if fixed_seed is None else Random(x=fixed_seed)
        self._no_unsafe_positions = no_unsafe_positions
        self._no_enemies = no_enemies
        self._unsafe_positions = []
        self.__generate_world__()
        # Safety specifications
        ltl_specification = (
            'until',
            (
                'and',
                (
                    'not',
                    'a'
                ),
                (
                    'not',
                    'b'
                )
            ),
            'c'
        )
        self._safety_monitor = SafetyMonitor(ltl_specification)

    def __generate_world__(self):
        xlim, ylim = self.size
        rng = self._rng
        self._agent_position = (rng.randint(0, xlim), rng.randint(0, ylim))
        # keep track of this so we can penalise the agent and enemy swapping places
        self._agent_previous_position = self._agent_position
        self.__move_target__()
        self.__create_unsafe_positions__()
        self.__create_enemies__()

    def __move_target__(self):
        xlim, ylim = self.size
        rng = self._rng
        x, y = rng.randint(0, xlim), rng.randint(0, ylim)
        while (x, y) in self._unsafe_positions:
            x, y = rng.randint(0, xlim), rng.randint(0, ylim)
        self._target_position = (x, y)

    def __create_unsafe_positions__(self):
        xlim, ylim = self.size
        rng = self._rng
        unsafe_positions = [
            (rng.randint(0, xlim), rng.randint(0, ylim)) for i in range(self._no_unsafe_positions)
        ]
        # If agent or target state is unsafe, try again (ugly hack)
        while self._agent_position in unsafe_positions or self._target_position in unsafe_positions:
            unsafe_positions = [
                (rng.randint(0, xlim), rng.randint(0, ylim)) for i in range(self._no_unsafe_positions)
            ]
        self._unsafe_positions = unsafe_positions

    def __create_enemies__(self):
        xlim, ylim = self.size
        rng = self._rng
        enemy_positions = [
            (rng.randint(0, xlim), rng.randint(0, ylim)) for i in range(self._no_enemies)
        ]
        # If agent or target state is in the same places as an enemy, try again (ugly hack)
        while self._agent_position in enemy_positions or self._target_position in enemy_positions:
            enemy_positions = [
                (rng.randint(0, xlim), rng.randint(0, ylim)) for i in range(self._no_unsafe_positions)
            ]
        self._enemies = [Enemy(pos) for pos in enemy_positions]

    def reset(self):
        #self._rng = Random(x=6543) # Uncomment for fixed seed
        self.__generate_world__()

    def step(self, action: GridWorldAction) -> Tuple[float, int]:
        true_props = set()  # True propositions for the safety monitor
        x, y = self._agent_position
        self._agent_previous_position = (x, y)
        # Move agent
        if action == GridWorldAction.NONE:
            pass
        elif action == GridWorldAction.UP:
            y += 1 if y < self.size[1] else 0
        elif action == GridWorldAction.DOWN:
            y -= 1 if y > 0 else 0
        elif action == GridWorldAction.LEFT:
            x -= 1 if x > 0 else 0
        elif action == GridWorldAction.RIGHT:
            x += 1 if x < self.size[0] else 0
        else:
            raise Exception(f'Invalid action: {action}')
        self._agent_position = (x, y)
        # Update enemies
        for enemy in self._enemies:
            enemy.step()
        # Calculate reward
        reward = -1
        # Reward agent for reaching target
        if self._agent_position == self._target_position:
            self.__move_target__()
            reward = 100
        elif self._agent_position == self._agent_previous_position:
            reward = -10 
        # Punish agent for entering unsafe state
        elif self._agent_position in self._unsafe_positions:
            true_props.add('a')
        else:
            # Punish agent for touching enemies
            enemy_positions = [x.position for x in self._enemies]
            if self._agent_position in enemy_positions:
                true_props.add('b')
            enemy_previous_positions = [
                x.previous_position for x in self._enemies]
            if self._agent_position in enemy_previous_positions and self._agent_previous_position in enemy_positions:
                # edge case where the agent and enemy swap places
                true_props.add('b')
        # Step the safety montior
        violation = self._safety_monitor.step(true_props) is not None
        violation = 1 if violation else 0
        return reward, violation

    def render(self) -> np.ndarray:
        if self._viewer is None:
            from . import rendering
            self._viewer = rendering.Viewer(*self._render_size)
            self._viewer.set_bounds(0, self.size[0], 0, self.size[1])
            # Draw unsafe positions
            self._unsafe_trans = []
            for _ in self._unsafe_positions:
                state = rendering.make_polygon(
                    [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)])
                state.set_color(1, 0, 0)
                self._viewer.add_geom(state)
                state_trans = rendering.Transform()
                self._unsafe_trans.append(state_trans)
                state.add_attr(state_trans)
            # Draw enemies
            self._enemy_trans = []
            for _ in self._enemies:
                enemy = rendering.make_circle(0.5)
                enemy.set_color(1, 0, 0)
                self._viewer.add_geom(enemy)
                enemy_trans = rendering.Transform()
                self._enemy_trans.append(enemy_trans)
                enemy.add_attr(enemy_trans)
            # Draw target
            target = rendering.make_circle(0.5)
            target.set_color(0, 0, 0)
            self._viewer.add_geom(target)
            self._target_trans = rendering.Transform()
            target.add_attr(self._target_trans)
            # Draw agent
            agent = rendering.make_circle(0.5)
            agent.set_color(0, 1, 0)
            self._viewer.add_geom(agent)
            self._agent_trans = rendering.Transform()
            agent.add_attr(self._agent_trans)
        # Update target location
        self._target_trans.set_translation(*self._target_position)
        # Update agent location
        self._agent_trans.set_translation(*self._agent_position)
        # Update unsafe position locations
        for i, unsafe_trans in enumerate(self._unsafe_trans):
            unsafe_trans.set_translation(*self._unsafe_positions[i])
        # Update enemy locations
        for i, enemy_trans in enumerate(self._enemy_trans):
            enemy_trans.set_translation(*self._enemies[i].position)

        return self._viewer.render(return_rgb_array=True)
    
    def close(self):
        self._viewer.close()


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    g = GridWorld()
    g.render()
    while True:
        action = input('Pls give an action: ')
        if action == 'exit':
            exit()
        reward = g.step(GridWorldAction(int(action)))
        print(f'Received reward of {reward}')
        g.render()
