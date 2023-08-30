from urllib.parse import urlparse

import mlflow
from ray.air import Result
from ray.train.torch.torch_checkpoint import TorchCheckpoint


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
