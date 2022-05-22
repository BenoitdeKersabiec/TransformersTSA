"""Implements a trading environment in gym"""
import gym
from sacred import Ingredient

env_ingredient = Ingredient("env")


class TradingEnv(gym.Env):
    """A Trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}
    visualization = None

    def __init__(self):
        super().__init__()
