import json
import ntpath
import os
from pathlib import Path

from dotenv import load_dotenv


def get_env_value(key: str) -> str:
    """Get an env variable defined in .env file

    Args:
        key (str): key of variable

    Returns:
        str: keys value
    """
    load_dotenv()
    return os.getenv(key)


def load_dict(path: str) -> dict:
    """Load a dictionary from a JSON's filepath.

    Args:
        path (str): location of file.

    Returns:
        Dict: loaded JSON data.
    """
    with open(path) as fp:
        d = json.load(fp)
    return d


def save_dict(d: dict, path: str, filename: str, cls: any = None, sortkeys: bool = False) -> None:
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


def get_file_name_from_path(path: str) -> str:
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)
