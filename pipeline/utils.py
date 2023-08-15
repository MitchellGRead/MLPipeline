import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

import numpy as np
import torch
from ray.air import Result
from ray.data import DatasetContext
from ray.train.torch.torch_checkpoint import TorchCheckpoint

from config.config import mlflow

DatasetContext.get_current().execution_options.preserve_order = True


def set_seeds(seed: int = 42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_dict(path: str) -> Dict:
    """Load a dictionary from a JSON's filepath.

    Args:
        path (str): location of file.

    Returns:
        Dict: loaded JSON data.
    """
    with open(path) as fp:
        d = json.load(fp)
    return d


def save_dict(d: Dict, path: str, filename: str, cls: Any = None, sortkeys: bool = False) -> None:
    """Save a dictionary to a specific location.

    Args:
        d (Dict): data to save.
        path (str): location of where to save the data.
        filename (str): name of file to save the data to
        cls (optional): encoder to use on dict data. Defaults to None.
        sortkeys (bool, optional): whether to sort keys alphabetically. Defaults to False.
    """
    create_dir(path)
    save_to = Path(path, filename)
    with open(save_to, "w") as fp:
        json.dump(d, indent=2, fp=fp, cls=cls, sort_keys=sortkeys)
        fp.write("\n")


def create_dir(path: str) -> None:
    """Create a directory for the given path if one does not exist

    Args:
        path (str): Path for directory
    """
    if not os.path.exists(path):  # pragma: no cover, OS operation
        os.makedirs(path)


def pad_array(arr: np.ndarray, dtype=np.int32) -> np.ndarray:
    """Pad an 2D array with zeros until all rows in the
    2D array are of the same length as a the longest
    row in the 2D array.

    Args:
        arr (np.array): input array

    Returns:
        np.array: zero padded array
    """
    max_len = max(len(row) for row in arr)
    padded_arr = np.zeros((arr.shape[0], max_len), dtype=dtype)
    for i, row in enumerate(arr):
        padded_arr[i][: len(row)] = row
    return padded_arr


def get_readable_timestamp() -> str:
    """Get a readable time stamp of the current time

    Returns:
        str: i.e. August 11, 2023 08:18:59 AM
    """
    return datetime.now().strftime("%B %d, %Y %I:%M:%S %p")


def get_filepath_timestamp() -> str:
    """Get a timestamp suitable for filepaths

    Returns:
        str: i.e. 2023_08_11_08_13_23
    """
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def get_run_id(
    experiment_name: str, trial_id: str
) -> str:  # pragma: no cover, mlflow functionality
    """Get the MLflow run ID for a specific Ray trial ID.

    Args:
        experiment_name (str): name of the experiment.
        trial_id (str): id of the trial.

    Returns:
        str: run id of the trial.
    """
    trial_name = f"TorchTrainer_{trial_id}"
    run = mlflow.search_runs(
        experiment_names=[experiment_name], filter_string=f"tags.trial_name = '{trial_name}'"
    ).iloc[0]
    return run.run_id


def get_best_checkpoint(run_id: str) -> TorchCheckpoint:  # pragma: no cover, mlflow logic
    """Get the best checkpoint from a specific run.

    Args:
        run_id (str): ID of the run to get the best checkpoint from.

    Returns:
        TorchCheckpoint: Best checkpoint from the run.
    """
    artifact_dir = urlparse(mlflow.get_run(run_id).info.artifact_uri).path
    results = Result.from_path(artifact_dir)
    return results.best_checkpoints[0][0]


def dict_to_list(data: Dict, keys: List[str]) -> List[Dict[str, Any]]:
    """Convert a dictionary to a list of dictionaries.

    Args:
        data (Dict): input dictionary.
        keys (List[str]): keys to include in the output list of dictionaries.

    Returns:
        List[Dict[str, Any]]: output list of dictionaries.
    """
    list_of_dicts = []
    for i in range(len(data[keys[0]])):
        new_dict = {key: data[key][i] for key in keys}
        list_of_dicts.append(new_dict)
    return list_of_dicts
