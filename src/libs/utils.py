import json
import pickle
from pathlib import Path

import yaml


def read_json(path: str) -> dict:
    """Reads json file to dictionary.

    Args:
        path (str): Path to json file.

    Returns:
        dict: output dictionary.
    """

    with open(Path(path), "rb") as file:
        out = json.load(file)

    return out


def save_json(data: dict, path: str):
    """Saves dictionary bject to json file.

    Args:
        data (dict): dictionary to be saved.
        path (str): Destination path.
    """

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file)


def read_pickle(path: str) -> dict:
    """Reads pickle file.

    Args:
        path (str): Path to pickle file.

    Returns:
        dict: Output object.
    """

    with open(path, "rb") as file:
        out = pickle.load(file)

    return out


def save_pickle(data: object, path: str):
    """Save object to pickle file.

    Args:
        data (object): Object to be saved.
        path (str): Destination path.
    """

    with open(path, "wb") as file:
        pickle.dump(data, file)


def read_yaml(path: str) -> dict:
    """Reads yaml file.

    Args:
        path (str): Path to yaml file.

    Returns:
        dict: Output dict.
    """

    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    return data
