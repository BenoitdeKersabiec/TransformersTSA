"""Implement a random model to test the environment"""
from random import random

from src.models import ModelWrapper


class RandomModel(ModelWrapper):
    """A random model to check that the environment is working"""
    def __init__(self, device):
        super().__init__(device)

    def act(self) -> float:
        return 2 * random() - 1
