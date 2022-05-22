"""Implement a random model to test the environment"""
from random import random
from src.models import ModelWrapper


class RandomModel(ModelWrapper):
    def __init__(self, env, device):
        super().__init__(env, device)

    def act(self) -> float:
        return 2 * random() - 1
