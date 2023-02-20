import json
import pickle
from pathlib import Path

import pandas as pd


def read_json(path: str):

    with open(Path(path), "rb") as f:
        out = json.load(f)

    return out


def save_json(data, path: str):

    with open(path, "w") as f:
        json.dump(data, f)


def read_pickle(path: str):

    with open(path, "rb") as f:
        out = pickle.load(f)

    return out


def save_pickle(data, path: str):

    with open(path, "wb") as f:
        pickle.dump(data, f)
