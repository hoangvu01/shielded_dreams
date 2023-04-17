import gymnasium


class ActionWeights(gymnasium.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def action(self, act):
        return act.argmax()
