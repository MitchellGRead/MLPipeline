import abc
from abc import ABC


class ModelInterface(ABC):
    """Model interface that models must adhere to in order to be registered

    Args:
        ABC (ABC): python Abstract Base Class (ABC) object
    """

    @abc.abstractmethod
    def train_loop_per_worker(self, config: dict) -> None:
        """Training loop to used for model training and tuning with Ray workers

        Args:
            config (dict): training parameter config
        """
        pass
