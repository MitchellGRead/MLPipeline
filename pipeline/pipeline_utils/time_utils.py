from datetime import datetime


def get_readable_timestamp() -> str:
    """Get a readable time stamp of the current time

    Returns:
        str: i.e. August 11, 2023 08:18:59 AM
    """
    return datetime.now().strftime("%B %d, %Y %I:%M:%S %p")


def get_filepath_timestamp() -> str:
    """Get a timestamp suitable for filepaths

    Returns:
        str: i.e. 2023-08-11_08-13-23
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
