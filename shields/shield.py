from abc import ABC, abstractclassmethod

from agent.models import ModelGroup


class Shield(ABC):
    def __init__(self, models: ModelGroup):
        super().__init__()

    @abstractclassmethod
    def step(self, belief, state, action, policy, t):
        pass
