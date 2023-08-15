from collections import OrderedDict

import numpy as np
from ray.data import Dataset
from ray.train.torch.torch_predictor import TorchPredictor
from sklearn.metrics import precision_recall_fscore_support
from snorkel.slicing import PandasSFApplier, slicing_function

from ml.api.metric_handler_interface import MetricHandlerInterface


@slicing_function()
def nlp_llm(x):  # pragma: no cover, eval workload
    """NLP projects that use LLMs."""
    nlp_project = "natural-language-processing" in x.tag
    llm_terms = ["transformer", "llm", "bert"]
    llm_project = any(s.lower() in x.text.lower() for s in llm_terms)
    return nlp_project and llm_project


@slicing_function()
def short_text(x):  # pragma: no cover, eval workload
    """Projects with short titles and descriptions."""
    return len(x.text.split()) < 8  # less than 8 words


class TagifaiMetricHandler(MetricHandlerInterface):
    def _get_overall_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Get overall model performance metrics

        Args:
            y_true (np.ndarray): ground truth labels
            y_pred (np.ndarray): predicted labels

        Returns:
            dict: overall metrics
        """
        metrics = precision_recall_fscore_support(y_true, y_pred, average="weighted")
        overall_metrics = {
            "precision": metrics[0],
            "recall": metrics[1],
            "f1": metrics[2],
            "num_samples": np.float64(len(y_true)),
        }
        return overall_metrics

    def _get_per_class_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, class_to_index: dict
    ) -> dict:
        """Get performance metrics per class

        Args:
            y_true (np.ndarray): ground truth labels
            y_pred (np.ndarray): predicted labels
            class_to_index (dict): dictionary mapping to class index

        Returns:
            dict: per class metrics
        """
        per_class_metrics = {}
        metrics = precision_recall_fscore_support(y_true, y_pred, average=None)
        print(class_to_index)
        for i, _class in enumerate(class_to_index):
            per_class_metrics[_class] = {
                "precision": metrics[0][i],
                "recall": metrics[1][i],
                "f1": metrics[2][i],
                "num_samples": np.float64(metrics[3][i]),
            }
        sorted_metrics = OrderedDict(
            sorted(per_class_metrics.items(), key=lambda tag: tag[1]["f1"], reverse=True)
        )
        return sorted_metrics

    def _get_slice_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, dataset: Dataset) -> dict:
        """Get performance metrics per slice

        Args:
            y_true (np.ndarray): ground truth labels
            y_pred (np.ndarray): predicted labels
            dataset (Dataset): Ray dataset with labels

        Returns:
            dict: performance metrics for slices
        """
        slice_metrics = {}
        df = dataset.to_pandas()
        df["text"] = df["title"] + " " + df["description"]
        slices = PandasSFApplier([nlp_llm, short_text]).apply(df)
        for slice_name in slices.dtype.names:
            mask = slices[slice_name].astype(bool)
            if sum(mask):
                metrics = precision_recall_fscore_support(
                    y_true[mask], y_pred[mask], average="micro"
                )
                slice_metrics[slice_name] = {}
                slice_metrics[slice_name]["precision"] = metrics[0]
                slice_metrics[slice_name]["recall"] = metrics[1]
                slice_metrics[slice_name]["f1"] = metrics[2]
                slice_metrics[slice_name]["num_samples"] = len(y_true[mask])
        return slice_metrics

    def generate_metrics(self, dataset: Dataset, predictor: TorchPredictor) -> dict:
        # y_true
        preprocessor = predictor.get_preprocessor()
        preprocessed_ds = preprocessor.transform(dataset)

        values = preprocessed_ds.select_columns(
            cols=["targets"]
        ).take_all()  # targets is defined in tagifai preprocessing
        y_true = np.stack([item["targets"] for item in values])

        # y_pred
        z = predictor.predict(data=dataset.to_pandas())["predictions"]
        y_pred = np.stack(z).argmax(1)

        return {
            "overall": self._get_overall_metrics(y_true, y_pred),
            "per_class": self._get_per_class_metrics(y_true, y_pred, preprocessor.class_to_index),
            "slices": self._get_slice_metrics(y_true, y_pred, dataset),
        }
