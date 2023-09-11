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


def create_checkpoint_name(run_id: str) -> str:  # pragma: no cover, weights & biases functionality
    return f"checkpoint_{get_run_name(run_id)}"


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


def get_object_path(project: str, id: str) -> str:
    """Gets a W&B run path

    Args:
        project (str): project name
        id (str): path to what we want to retrieve in W&B. i.e, a run id for a specific run or a <name>:<alias> for an artifact

    Returns:
        str: path to object
    """
    return f"{get_project_path(project)}/{id}"


def get_checkpoint_artifact_path(project: str, run_id: str) -> str:
    """Gets the W&B path name to a checkpoint artifact

    Args:
        project (str): project name to artifact lives under
        run_id (str): id of the trial for a certain run

    Returns:
        str: W&B checkpoint artifact url
    """
    return f"{get_project_path(project)}/{get_checkpoint_name(run_id)}"


def verify_artifact_name(name: str) -> bool:
    """Verify an artifact name is in the format of <name>:<alias>

    Args:
        name (str): name of the artifact

    Returns:
        bool: whether it adheres to the <name>:<alias> format
    """
    name_alias = name.split(":")
    if len(name_alias) != 2:
        return False
    return True
