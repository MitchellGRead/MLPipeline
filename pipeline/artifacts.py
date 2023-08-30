import typer
from pipeline_utils import wb_utils
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

    run = wandb.Api().run(wb_utils.get_run_path(model_project.value, run_id))

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
            f"Error deleting artifacts for {run_id} - \n Managed to delete the follow {len(artifacts_deleted)} --> {artifacts_deleted}"
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


if __name__ == "__main__":
    app()
