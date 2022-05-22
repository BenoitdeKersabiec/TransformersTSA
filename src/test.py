from src.data import get_data, data_ingredient
from src.env import TradingEnv, env_ingredient
from src.models import RandomModel

from sacred import Experiment
import pandas as pd
from torch.backends import mps
from torch import cuda

ex = Experiment("test", ingredients=[data_ingredient, env_ingredient])
device = "mps" if mps.is_available() else "cuda" if cuda.is_available() else "cpu"


@ex.config
def config():
    pass


@ex.automain
def test():
    data: pd.DataFrame = get_data()
    env: TradingEnv = TradingEnv(data=data)
    model = RandomModel(env=env, device=device)
