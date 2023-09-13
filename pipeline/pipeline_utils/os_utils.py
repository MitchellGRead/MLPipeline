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


def get_name_from_path(file_path: str) -> str:
    """Gets the last name from path in an OS safe manner
    i.e.
    hello/world/test --> test
    hello/world/test.txt --> text.txt

    Args:
        file_path (str): path to file

    Returns:
        str: filename
    """
    head, tail = ntpath.split(file_path)
    return tail or ntpath.basename(head)


def get_files_in_dir(dir_path: str) -> list[str]:
    """Gets the file names as paths within a directory

    Args:
        dir_path (str): path to directory

    Returns:
        list[str]: list of filenames in the directory one layer deep

    Raises:
        ValueError if the provided path is not a directory
    """
    if not os.path.isdir(dir_path):
        raise ValueError(f"{dir_path} is not a path to a directory")
    return [
        Path(dir_path, filename)
        for filename in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, filename))
    ]


def is_file_or_dir(path: str) -> bool:
    """Returns if the path provided is a file or a directory

    Args:
        path (str): OS path

    Returns:
        bool: if path is a file or directory
    """
    return os.path.isfile(path) or os.path.isdir(path)
