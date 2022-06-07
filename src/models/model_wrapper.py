"""Implements a base class for project models"""


class ModelWrapper:
    """Base class for model to run on env"""

    def __init__(self, device):
        self.device = device

    def act(self) -> float:
        """Computes the best action according to the agent policy

        Returns: action in [-1, 1] where -1 means sell all the portfolio and +1 means buy all you can
        """
        raise NotImplementedError
