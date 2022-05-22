"""Implements a base class for project models"""


class ModelWrapper:
    def __init__(self, env, device):
        self.env = env
        self.device = device

    def act(self) -> float:
        """Computes the best action according to the agent policy

        Returns: action in [-1, 1] where -1 means sell all the portfolio and +1 means buy all you can
        """
        raise NotImplementedError
