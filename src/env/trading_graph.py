"""Contains the code related to live visualization of the agent's performance"""
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from sacred import Ingredient

graph_ingredient = Ingredient("graph")


@graph_ingredient.config
def config():
    """Contains the hyper-parameters related to graph-rendering"""
    volume_chart_height = 0.33
    window_size = 40

    up_color = "#27A59A"
    down_color = "#EF534F"
    up_text_color = "#73D3CC"
    down_text_color = "#DC2C27"

    graph_title = "Agent's performance"


class TradingGraph:
    """A stock trading visualization using matplotlib made to render OpenAI gym environments"""

    @graph_ingredient.capture
    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: int,
        graph_title: str,
        window_size: int,
    ):
        """Instantiate a trading graph

        :param data: A dataframe containing the ohlcv data, timestamped
        :param initial_balance: the agent's initial balance
        :param graph_title: The plot graph title
        :param window_size: The number of timestep to display on screen at a time
        """
        self.data = data.reset_index()
        self.window_size = window_size
        self.net_worth = np.ones(len(self.data["date"])) * initial_balance

        # Create figure and axes inside the figure
        self.fig = mpf.figure(figsize=(12, 8), facecolor=(0.82, 0.83, 0.85))
        self.price_axe = self.fig.add_axes([0.1, 0.15, 0.88, 0.50])
        self.volume_axe = self.fig.add_axes(
            [0.1, 0.05, 0.88, 0.10], sharex=self.price_axe
        )
        self.net_worth_axe = self.fig.add_axes(
            [0.1, 0.65, 0.88, 0.20], sharex=self.price_axe
        )

        # Specifies the labels
        self.price_axe.set_ylabel("price")
        self.volume_axe.set_ylabel("volume")
        self.net_worth_axe.set_ylabel("net-worth")

        # Add title
        self.fig.text(
            0.50, 0.94, graph_title, ha="center", size="16", weight="bold", va="bottom"
        )

        # Show the graph without blocking the rest of the program
        plt.show(block=False)

    def render(self, current_step: int, net_worth: float):
        """Renders a new frame updated to the current timestep

        :param current_step: The current timestep in terms of position in the dataframe
        :param net_worth: The new net-worth for the current step
        """
        # Clear the precedent plot
        self.price_axe.clear()
        self.volume_axe.clear()
        self.net_worth_axe.clear()

        # store the new net-worth
        self.net_worth[current_step] = net_worth

        # compute the plot step-range
        window_start = max(current_step - self.window_size, 0)
        step_range = range(window_start, current_step + 1)

        # Plot data
        net_worth_plot = [
            mpf.make_addplot(self.net_worth[step_range], ax=self.net_worth_axe)
        ]
        mpf.plot(
            self.data.loc[
                step_range, ["date", "open", "high", "low", "close", "volume"]
            ].set_index("date"),
            type="candle",
            ax=self.price_axe,
            volume=self.volume_axe,
            addplot=net_worth_plot,
            datetime_format="%d-%m-%Y %H:%M",
            xrotation=0,
        )
        # Rewrite subplot title because they tend to disappear
        self.price_axe.set_ylabel("price")
        self.volume_axe.set_ylabel("volume")
        self.net_worth_axe.set_ylabel("net-worth")
        mpf.show(block=False)

        # Freeze matplotlib so it renders the frame
        plt.pause(0.0001)

    @staticmethod
    def close():
        """Close the live rendering figure"""
        plt.close()
