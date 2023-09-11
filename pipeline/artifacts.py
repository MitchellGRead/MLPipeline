import os

import typer
from pipeline_utils import os_utils, wb_utils
from typing_extensions import Annotated

import wandb
from config.config import logger
from pipeline import model_registration

app = typer.Typer()


@app.command()
def delete_run_artifacts(
    model_to_delete_for: Annotated[str, typer.Option(help="model id to delete for")],
    run_id: Annotated[str, typer.Option(help="run id to delete model artifacts from")],
    artifact_type: Annotated[str, typer.Option(help="artifact type to delete")] = "model",
) -> bool:
    """Delete all but the latest model artifacts from a run

    Args:
        model_to_delete_for (str): model id to delete for
        run_id (str): run id to delete model artifacts from
        artifact_type (str, optional): artifact type to delete. Defaults to "model"

    Returns:
        bool: whether an error was encountered
    """
    model_project = model_registration.getModelProject(model_to_delete_for)
    logger.info(f"Only keeping the latest artifact for run {run_id} under {model_to_delete_for}")

    run = wandb.Api().run(wb_utils.get_object_path(model_project.value, run_id))

    artifacts = run.logged_artifacts()
    artifacts_to_delete = sorted(artifacts, key=lambda artifact: artifact.created_at, reverse=True)
    artifacts_to_delete = [
        artifact for artifact in artifacts_to_delete if artifact.type == artifact_type
    ]
    artifacts_deleted = []

    try:
        for artifact in artifacts_to_delete[1:]:
            logger.info(f"Deleting model artifact {artifact.name}")
            artifact.delete(delete_aliases=True)
            artifacts_deleted.append(artifact.name)
    except Exception:
        logger.error(
            f"Error deleting artifacts for {run_id} - \n Managed to delete the following {len(artifacts_deleted)} --> {artifacts_deleted}"
        )
        return False
    else:
        logger.info(f"Successfully deleted {len(artifacts_deleted)} artifacts")
        return True


@app.command()
def clean_up_runs() -> bool:
    """Cleans up all runs in the project that have the tag "delete" associated to them

    Returns:
        bool: whether an error was encountered
    """
    api = wandb.Api()

    for project in model_registration.ModelProject:
        try:
            runs = api.runs(wb_utils.get_project_path(project.value))
            runs = [run for run in runs if "delete" in run.tags]
        except ValueError:
            logger.error(f"No project found for {project.value}")
            continue

        logger.info(f"Found {len(runs)} runs to delete from {project.value}")
        for run in runs:
            logger.info(f"Deleting run {run.id}")
            run.delete()


@app.command()
def process_dataset(
    dataset_loc: Annotated[str, typer.Option(help="location of the locally stored dataset")],
    data_type: Annotated[
        str,
        typer.Option(
            help="type of data ie. raw_data, globally_preprocessed_data, split_data, etc. This will be used for typing the artifact in W&B."
        ),
    ],
    data_for_model_id: Annotated[str, typer.Option(help="model id that this data is for")],
) -> tuple[bool, str]:
    """Process a dataset into W&B for versioning and lineage tracking.
    Note that versioning is determined by the filename so changing it will create a new version in W&B.
    It is better to keep the name the same and apply aliases to the artifacts in W&B for distinctions in training.

    Args:
        dataset_loc str: location of the locally stored dataset
        data_type str: type of data ie. raw_data, globally_preprocessed_data, split_data, etc. This will be used for typing the artifact in W&B.
        data_for_model_id str: model id that this data is for

    Returns:
        tuple[bool, str]: boolean if the dataset processing was successful or not and the saved artifact name. Name is empty on failure.
    """
    assert os.path.isfile(
        dataset_loc
    ), "Dataset processing only supports paths to a file currently."

    filename = os_utils.get_file_name_from_path(dataset_loc)
    run_id = wandb.util.generate_id()
    run_name = f"Dataset_{data_type}_{run_id}"
    model_project = model_registration.getModelProject(data_for_model_id)

    logger.info(
        f"Processing {model_project.value} dataset from {dataset_loc} for file {filename}. W&B run name is {run_name}"
    )

    run = wandb.init(project=model_project.value, job_type=data_type, group="Data", name=run_name)
    data_artifact = wandb.Artifact(name=filename, type=data_type)

    try:
        data_artifact.add_file(dataset_loc)
        saved_artifact = run.log_artifact(data_artifact)
    except:  # noqa: E722
        logger.error(f"Failed to process dataset at {dataset_loc}")
        return False, ""

    run.finish()
    return True, f"{saved_artifact.name}:latest"


if __name__ == "__main__":
    app()
