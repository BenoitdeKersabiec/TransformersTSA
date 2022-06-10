"""Test script for environment"""

from typing import Optional

import pandas as pd
from sacred import Experiment
from torch import cuda
from torch.backends import mps

from src.data import data_ingredient, get_data
from src.env import env_ingredient, TradingEnv
from src.models import HodlModel, RandomModel

ex = Experiment("test", ingredients=[data_ingredient, env_ingredient])
device = "mps" if mps.is_available() else "cuda" if cuda.is_available() else "cpu"


@ex.config
def config():
    """Store the test experiment parameters"""
    model_type: str = "HodlModel"
    model_weights: Optional[str] = None
    total_time_steps: int = 10000


@ex.automain
def test(model_type, model_weights, total_time_steps):
    """Test the environment with the specified model"""
    data: pd.DataFrame = get_data()
    env: TradingEnv = TradingEnv(data=data)

    if model_type == "RandomModel":
        model = RandomModel(device=device)
    elif model_type == "HodlModel":
        model = HodlModel(device=device)
    else:
        raise NameError(f"Unknown model type: '{model_type}'")

    if model_weights is not None:
        raise NotImplementedError

    env.reset()
    for t in range(total_time_steps):
        action = model.act()
        env.step(action)
        env.render()
