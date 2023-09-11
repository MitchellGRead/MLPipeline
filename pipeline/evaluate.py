import json
from pathlib import Path

import typer
from pipeline_utils import os_utils, time_utils, wb_utils
from ray.train.torch.torch_checkpoint import TorchCheckpoint
from ray.train.torch.torch_predictor import TorchPredictor
from typing_extensions import Annotated

import wandb
from config.config import RESULTS_DIR, logger
from ml.api.data_handler_interface import DataHandlerInterface
from ml.api.metric_handler_interface import MetricHandlerInterface
from pipeline import model_registration

app = typer.Typer()


@app.command()
def get_best_run_id(
    model_to_eval: Annotated[str, typer.Option(help="experiment name to get run id for")],
    metric: Annotated[
        str, typer.Option(help="training metric to filter by in wandb-summary.json")
    ] = "val_loss",
    sort_mode: Annotated[
        str, typer.Option(help="mode to sort runs by their metric. ASC/DESC")
    ] = "ASC",
) -> str:  # pragma: no cover, w&b logic
    """Get the best run id for a given experiment by a recorded metric

    Args:
        model_to_eval (str): experiment name to get run id for
        metric (str): training metric to filter by in wandb-summary.json. Defaults to val_loss
        sort_mode (str, option): mode to sort runs by their metric. ASC/DESC. Defaults to ASC

    Returns:
        str: best run id from experiment based on the metric
    """
    model_project = model_registration.getModelProject(model_to_eval)

    if sort_mode == "ASC":
        order = f"+summary_metrics.{metric}"
    else:
        order = f"-summary_metrics.{metric}"

    api = wandb.Api()
    runs = api.runs(
        wb_utils.get_project_path(model_project.value),
        filters={
            "group": model_to_eval,
        },
        order=order,
    )

    assert len(runs) > 0, f"No runs matched with the W&B group {model_to_eval}"

    best_run = runs[0].id

    print(best_run)  # for saving to CLI variable
    return best_run


@app.command()
def evaluate(
    model_to_eval: Annotated[str, typer.Option(help="model id we are evaluting for")],
    dataset_loc: Annotated[str, typer.Option(help="dataset to evaluate on")],
    run_id: Annotated[str, typer.Option(help="run id to evaluate on")] = None,
    metric: Annotated[
        str, typer.Option(help="training metric to filter by in wandb-summary.json")
    ] = "val_loss",
    sort_mode: Annotated[
        str, typer.Option(help="mode to sort runs by their metric. ASC/DESC")
    ] = "ASC",
    results_loc: Annotated[str, typer.Option(help="dataset to evaluate on")] = RESULTS_DIR,
) -> dict:
    """Evaluate the provided run

    Args:
        experiment_name (str): model id we are evaluting for
        trial (str): model run id to load and evaluate for
        dataset_loc (str): location dataset to evaluate on
        run_id (str, optional): run id to evaluate for
        metric (str, optional): training metric to filter by in wandb-summary.json. Defaults to val_loss
        sort_mode (str, optional): mode to sort runs by their metric. ASC/DESC. Defaults to ASC
        results_loc (str, optional): location to save results. Defaults to ./results directory.
            None if you don't want to save the evaluation

    Returns:
        dict: model performance metrics
    """
    data_handler = model_registration.getDataHandler(model_to_eval)(dataset_loc)
    metric_handler = model_registration.getMetricHandler(model_to_eval)()
    model_project = model_registration.getModelProject(model_to_eval)
    assert isinstance(data_handler, DataHandlerInterface)
    assert isinstance(metric_handler, MetricHandlerInterface)
    assert isinstance(model_project, model_registration.ModelProject)

    best_run_id = run_id if run_id else get_best_run_id(model_to_eval, metric, sort_mode)

    return internal_evaluate(
        model_to_eval=model_to_eval,
        model_project=model_project,
        run_id=best_run_id,
        data_handler=data_handler,
        metric_handler=metric_handler,
        results_loc=results_loc,
    )


def internal_evaluate(
    model_to_eval: str,
    model_project: model_registration.ModelProject,
    run_id: str,
    data_handler: DataHandlerInterface,
    metric_handler: MetricHandlerInterface,
    results_loc: str = RESULTS_DIR,
) -> dict:
    """Internal evaluation method for connecting with other pipeline components

    Args:
        model_to_eval (str): unique model pipeline id
        run_id (str): run id to evaluate on
        data_handler (DataHandlerInterface): models data handler
        metric_handler (MetricHandlerInterface): models metric handler
        results_loc (str, optional): location to save results to. Defaults to ./results directory.
            None if you don't want to save the evaluation

    Returns:
        dict: model performance metrics
    """

    if results_loc:
        evalution_results_loc = Path(results_loc, "evaluation", model_to_eval, run_id)
        os_utils.create_dir(evalution_results_loc)
    else:
        evalution_results_loc = None

    trial_checkpoint_name = wb_utils.get_checkpoint_name(run_id)
    logger.info(f"Trial checkpoint name: {trial_checkpoint_name}")

    run_name = f"TorchPredictor_{run_id}"

    logger.info(f"Initializing W&B run under group {model_to_eval} with the run {run_name}")
    run = wandb.init(
        project=model_project.value, job_type="evaluate", group=model_to_eval, name=run_name
    )

    logger.info(
        f"Evaluating model {model_to_eval} for trial checkpoint {trial_checkpoint_name}. Results saved locally to {evalution_results_loc}"
    )

    data = data_handler.get_data()
    checkpoint_model = run.use_artifact(trial_checkpoint_name)
    checkpoint = TorchCheckpoint.from_directory(checkpoint_model.download())
    predictor = TorchPredictor.from_checkpoint(checkpoint)

    metrics = {
        "timestamp": time_utils.get_readable_timestamp(),
        "run_id": run.id,
        "eval_run_id": run_id,
        "metrics": metric_handler.generate_metrics(data, predictor),
    }
    logger.info(json.dumps(metrics, indent=2))

    logger.info("Saving local results and creating W&B artifact")

    artifact_name = f"{model_to_eval}_{run_id}.json"
    evaluation_artifact = wandb.Artifact(name=artifact_name, type="evaluation")

    if evalution_results_loc:
        os_utils.save_dict(
            d=metrics,
            path=evalution_results_loc,
            filename=artifact_name,
        )

    evaluation_artifact.add_file(Path(evalution_results_loc, artifact_name), artifact_name)
    run.log_artifact(evaluation_artifact)
    run.finish()

    return metrics


if __name__ == "__main__":
    app()
