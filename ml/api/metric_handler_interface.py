import abc
from abc import ABC

from ray.data import Dataset
from ray.train.torch.torch_predictor import TorchPredictor


class MetricHandlerInterface(ABC):
    """Model interface that models must adhere to in order to be registered

    Args:
        ABC (ABC): python Abstract Base Class (ABC) object
    """

    @abc.abstractmethod
    def generate_metrics(self, dataset: Dataset, predictor: TorchPredictor) -> dict:
        """Generate dict of performance metrics to track for a specific model

        Args:
            dataset (Dataset): Ray dataset used for eval
            predictor (TorchPredictor): Ray TorchPredictor to get model preprocessor used for training and predict on the dataset

        Returns:
            dict: metrics to track for given model
        """
        pass
