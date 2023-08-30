from config.config import WEIGHTS_AND_BIASES_PROJECT


def get_run_name(
    run_id: str,
) -> str:  # pragma: no cover, weights & biases functionality
    """Name of a Weights and Biases run id under a group

    Args:
        run_id (str): id of the trial

    Returns:
        str: Training run id
    """
    return f"TorchTrainer_{run_id}"


def get_checkpoint_name(
    run_id: str, alias: str = "latest"
) -> str:  # pragma: no cover, weights & biases functionality
    """Name of a Weights and Biases latest model artifact for a TorchTrainer run

    Args:
        run_id (str): id of the trial
        alias (str): alias of the checkpoint artifact. Defaults "latest"

    Returns:
        str: checkpoint model name
    """
    return f"checkpoint_{get_run_name(run_id)}:{alias}"


def get_project_path(project: str) -> str:
    """Gets a W&B project path

    Args:
        project (str): project name

    Returns:
        str: path to the project
    """
    return f"{WEIGHTS_AND_BIASES_PROJECT}/{project}"


def get_run_path(project: str, run_id: str) -> str:
    """Gets a W&B run path

    Args:
        project (str): project name
        run_id (str): run id

    Returns:
        str: path to run
    """
    return f"{get_project_path(project)}/{run_id}"


def get_checkpoint_artifact_path(project: str, run_id: str) -> str:
    """Gets the W&B path name to a checkpoint artifact

    Args:
        project (str): project name to artifact lives under
        run_id (str): id of the trial for a certain run

    Returns:
        str: W&B checkpoint artifact url
    """
    return f"{get_project_path(project)}/{get_checkpoint_name(run_id)}"
