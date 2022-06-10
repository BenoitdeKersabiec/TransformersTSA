"""Implement a hodl model that will only buy"""
from src.models import ModelWrapper


class HodlModel(ModelWrapper):
    """A random model to check that the environment is working"""

    def __init__(self, device):
        super().__init__(device)

    def act(self) -> float:
        return 1
