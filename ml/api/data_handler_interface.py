import abc
from abc import ABC

from ray.data import Dataset


class DataHandlerInterface(ABC):
    """Model interface that models must adhere to in order to be registered

    Args:
        ABC (ABC): python Abstract Base Class (ABC) object
    """

    @abc.abstractmethod
    def __init__(self, data_loc: str, num_samples: int = None) -> None:
        """Load data into a Ray Dataset

        Args:
            data_loc (str): data location to pull from
            num_samples (int, optional): number of samples to take from the dataset. Defaults to None (take all samples).
        """
        super().__init__()

    @abc.abstractmethod
    def get_data(self) -> Dataset:
        """Get the create Ray dataset

        Returns:
            Dataset: Ray dataset that this handler is managing
        """
        pass

    @abc.abstractmethod
    def split_data(
        self, test_size: float, shuffle: bool = True, seed: int = 1234
    ) -> tuple[Dataset, Dataset]:
        """Split data into train and eval datasets based on the stratify

        Args:
            test_size (float): size of test data
            shuffle (bool, optional): whether to shuffle the data splits or not. Defaults to True.
            seed (int, optional): seed for shuffling. Defaults to 1234.

        Returns:
            tuple[Dataset, Dataset]: split train and eval data sets
        """
        pass

    @abc.abstractmethod
    def add_to_config(self, train_loop_config: dict) -> dict:
        """Add any data specific configurations to the train loop config. This should primarily be done with the training data split

        Args:
            train_loop_config (dict): training loop config

        Returns:
            dict: updated training loop config
        """
        pass


class IndexableDataset(ABC):
    """An indexable dataset to retrieve some collection of data

    Args:
        ABC (ABC): python Abstract Base Class (ABC) object
    """

    @abc.abstractmethod
    def get(self, idx: int) -> dict[str, any]:
        """Get an index of the dataset

        Args:
            idx (int): index to get

        Returns:
            dict[str, any]: data to return
        """
        pass

    @abc.abstractmethod
    def len(self) -> int:
        """Get the length of the data

        Returns:
            int: data length
        """
        pass
