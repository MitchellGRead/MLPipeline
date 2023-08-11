import abc
from abc import ABC


class ModelInterface(ABC):
    """Model interface that models must adhere to in order to be registered

    Args:
        ABC (ABC): python Abstract Base Class (ABC) object
    """

    @abc.abstractmethod
    def train_loop_per_worker(self) -> None:
        """Training loop to used for model training and tuning with Ray workers"""
        pass
