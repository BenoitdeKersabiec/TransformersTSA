"""Implements a trading environment in gym"""
import logging
import os.path

import gym
import pandas as pd
from gym import spaces
from numpy import inf
from sacred import Ingredient

env_ingredient = Ingredient("env")


@env_ingredient.config
def config():
    """Contains the environment hyper-parameters"""
    initial_balance: int = 1000
    fee: float = 1e-3
    lookback_window: int = 60
    output_path = "/Users/benoit/Projects/TransformersTSA/results/output.csv"


# pylint: disable=too-many-instance-attributes
class TradingEnv(gym.Env):
    """A Trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human", "csv"]}
    visualization = None

    # pylint: disable=too-many-arguments
    @env_ingredient.capture
    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: int,
        fee: float,
        lookback_window: int,
        output_path: str,
    ):
        super().__init__()
        # Define environment variables
        self.data = data
        self.lookback_window = lookback_window

        self.initial_balance = initial_balance
        self.fee = fee

        # Gym action space
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))

        # Gym observation space
        self.observation_space = spaces.Box(low=0, high=inf, shape=(1,))

        self.render_init = False
        self.output_path = output_path

        # Define experiment variables
        self.current_date = None

        self.balance = None
        self.net_worth = None
        self.portfolio = None
        self.cost_basis = None
        self.trades = None

    @env_ingredient.capture
    def reset(self):
        """Reset the environment to a default state"""
        self.current_date = self.data.index[self.lookback_window]

        self.balance = self.initial_balance
        self.net_worth = self.balance
        self.portfolio = 0
        self.cost_basis = 0
        self.trades = []

    def step(self, action: float):
        """Perform an action on the environment"""
        assert -1 <= action <= 1, "Invalid action"
        if action >= 0:
            self._buy(action)
        elif action < 0:
            self._sell(-action)

        self._update_net_worth()

        self.current_date = self.data.loc[self.current_date :].index[1]

        print(
            f"Balance: {self.balance}, Portfolio: {self.portfolio}, Net-Worth: {self.net_worth}"
        )

    def _buy(self, action: float):
        """Buy a certain amount of assets relative to the action"""

        amount_value = self.balance * action

        # Devil fees
        amount_value *= 1 - self.fee
        current_price = self.data.close.loc[self.current_date]

        self.portfolio += amount_value / current_price
        self.balance -= amount_value

    def _sell(self, action: float):
        """Sell a certain amount of assets relative to the action"""

        sold_assets = self.portfolio * action
        current_price = self.data.close.loc[self.current_date]
        amount_value = sold_assets * current_price

        # Devil fees
        amount_value *= 1 - self.fee

        self.portfolio -= sold_assets
        self.balance += amount_value

    def _update_net_worth(self):
        """Updates the net-worth of the agent"""
        current_price = self.data.close.loc[self.current_date]
        self.net_worth = self.balance + self.portfolio * current_price

    def render(self, mode="csv"):
        if mode == "csv":
            if not self.render_init:
                self._init_csv()
            else:
                with open(self.output_path, "a+", encoding="UTF-8") as output_file:
                    output_file.write(self.current_date.strftime("%d-%m-%Y %H:%M"))
                    output_file.write(f", {self.data.close.loc[self.current_date]}, ")
                    output_file.write(f"{round(self.balance, 2)}, ")
                    output_file.write(f"{round(self.portfolio, 2)}, ")
                    output_file.write(f"{round(self.net_worth, 2)} ")
                    output_file.write("\n")

    def _init_csv(self):
        if os.path.isfile(self.output_path):
            logging.warning("Removing output_file %s", self.output_path)
            os.remove(self.output_path)

        with open(self.output_path, "w", encoding="UTF-8") as output_file:
            output_file.write("Date, Price, Balance, Portfolio, Net-Worth\n")

        self.render_init = True
