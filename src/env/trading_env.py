"""Implements a trading environment in gym"""
import gym
import pandas as pd
from sacred import Ingredient

env_ingredient = Ingredient("env")


@env_ingredient.config
def config():
    """Contains the environment hyper-parameters"""
    initial_balance: int = 1000
    fee: float = 1e-3
    lookback_window: int = 60


class TradingEnv(gym.Env):
    """A Trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}
    visualization = None

    @env_ingredient.capture
    def __init__(
        self, data: pd.DataFrame, initial_balance: int, fee: float, lookback_window: int
    ):
        super().__init__()
        self.data = data
        self.lookback_window = lookback_window

        self.initial_balance = initial_balance
        self.fee = fee

        self.current_date = self.data.index[self.lookback_window]
