"""Script to pull/load data"""
import shutil
from os.path import isfile, join

import numpy as np
import pandas as pd
import requests
from sacred import Ingredient

data_ingredient = Ingredient("data")


@data_ingredient.config
def config():
    """Store data ingredient parameters"""
    api_url = "https://www.cryptodatadownload.com/cdd/"
    data_file = "Binance_BTCUSDT_minute.csv"
    data_folder = "/Users/benoit/Projects/TransformersTSA/data/"
    force_pull = False


@data_ingredient.capture
def pull_data(api_url: str, data_file: str, data_folder: str) -> pd.DataFrame:
    """Pull data from the internet

    Args:
        api_url: Url to pull the data from
        data_file: File to pull from the url
        data_folder: Path to data directory

    Returns:
        A dataframe with the ohlc data inside
    """
    # Pulling raw data
    print("Downloading data...")
    raw_data = requests.get(api_url + data_file, stream=True).raw

    with open(join(data_folder, "raw", data_file), "wb") as location:
        shutil.copyfileobj(raw_data, location)
        del raw_data
    print("Processing data...")
    # Processing data
    data = pd.read_csv(join(data_folder, "raw", data_file), header=1, index_col=1)
    data.rename(columns={"Volume USDT": "volume"}, inplace=True)
    data = data[["open", "high", "low", "close", "volume"]].astype(float)
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    # data = data[data.volume != 0]
    data = data.loc[pd.to_datetime("2019-09-08 19:13:00") :]  # Remove weird points
    data = data[data.volume != 0]
    # data.replace(0, np.nan, inplace=True)
    # data.volume = data.volume.ffill()

    data.to_csv(join(data_folder, "processed", data_file))
    print("Data processed and saved to", join(data_folder, "processed", data_file))

    return data


@data_ingredient.capture
def get_data(data_folder: str, data_file: str, force_pull: bool):
    """Load data, either by reading local files or pulling them from the internet

    Args:
        data_folder: Path to data directory
        data_file: File to load
        force_pull: True will force pulling from the internet

    Returns:
        A dataframe with the ohlc data
    """

    print(
        "------------------------------------------- Loading Data -------------------------------------------"
    )
    data_path: str = join(data_folder, "processed", data_file)

    if isfile(data_path) and not force_pull:
        print("Reading data from", data_path)

        data = pd.read_csv(data_path, index_col=0).astype(float)
        data.index = pd.to_datetime(data.index)
    else:
        data = pull_data()

    return data
