import datetime

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplfinance as mpf
import numpy as np
from matplotlib import style
from mplfinance.original_flavor import candlestick_ochl as candlestick
from sacred import Ingredient

# matplotlib.use("TkAgg")



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


def date2num(date):
    return (date.astype(np.int64) / (1e9 * 60)).astype(np.int32)


class TradingGraph:
    """A stock trading visualization using matplotlib made to render OpenAI gym environments"""

    @graph_ingredient.capture
    def __init__(self, data, initial_balance, graph_title, window_size):
        """Instantiate a trading graph"""
        self.data = data.reset_index()
        self.window_size = window_size
        self.net_worth = np.ones(len(self.data["date"])) * initial_balance

        style.use("dark_background")

        # Create top subplot for net worth axis
        self.fig = mpf.figure(figsize=(12, 8), facecolor=(0.82, 0.83, 0.85))
        self.price_axe = self.fig.add_axes([0.1, 0.15, 0.88, 0.50])
        self.volume_axe = self.fig.add_axes(
            [0.1, 0.05, 0.88, 0.10], sharex=self.price_axe
        )
        self.net_worth_axe = self.fig.add_axes(
            [0.1, 0.65, 0.88, 0.20], sharex=self.price_axe
        )

        self.price_axe.set_ylabel("price")
        self.volume_axe.set_ylabel("volume")
        self.net_worth_axe.set_ylabel("net-worth")

        self.title = self.fig.text(
            0.50, 0.94, graph_title, ha="center", size="16", weight="bold", va="bottom"
        )

        # Show the graph without blocking the rest of the program
        plt.show(block=False)

    def render(self, current_step, net_worth, trades):

        self.net_worth[current_step] = net_worth

        window_start = max(current_step - self.window_size, 0)
        step_range = range(window_start, current_step + 1)

        self.price_axe.clear()
        self.volume_axe.clear()
        self.net_worth_axe.clear()
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
        )
        self.price_axe.set_ylabel("price")
        self.volume_axe.set_ylabel("volume")
        self.net_worth_axe.set_ylabel("net-worth")
        mpf.show(block=False)

        plt.pause(0.001)

    @staticmethod
    def close():
        plt.close()
