import json
from pathlib import Path

import mlflow
import typer
from ray.train.torch.torch_predictor import TorchPredictor
from typing_extensions import Annotated

from config.config import RESULTS_DIR, logger
from ml.api.data_handler_interface import DataHandlerInterface
from ml.api.metric_handler_interface import MetricHandlerInterface
from pipeline import model_registration, utils

app = typer.Typer()


@app.command()
def get_best_run_id(
    experiment_name: Annotated[str, typer.Option(help="experiment name to get run id for")],
    metric: Annotated[str, typer.Option(help="training metric to filter by")],
    sort_mode: Annotated[
        str, typer.Option(help="mode to sort runs by their metric. ASC/DESC")
    ] = "ASC",
) -> str:  # pragma: no cover, mlflow logic
    """Get the best run id for a given experiment by a recorded metric

    Args:
        experiment_name (str): experiment name to get run id for")
        metric (str): training metric to filter by")
        sort_mode (str, option): mode to sort runs by their metric. ASC/DESC. Defaults to ASC

    Returns:
        str: best run id from experiment based on the metric
    """
    sorted_runs = mlflow.search_runs(
        experiment_names=[experiment_name], order_by=[f"metrics.{metric} {sort_mode}"]
    )
    best_run = sorted_runs.iloc[0].run_id
    print(best_run)  # for saving to CLI variable
    return best_run


@app.command()
def evaluate(
    model_to_eval: Annotated[str, typer.Option(help="model id we are evaluting for")],
    run_id: Annotated[str, typer.Option(help="run id to load and evaluate for")],
    dataset_loc: Annotated[str, typer.Option(help="dataset to evaluate on")],
    results_loc: Annotated[str, typer.Option(help="dataset to evaluate on")] = RESULTS_DIR,
) -> dict:
    """Evaluate the provided run

    Args:
        experiment_name (str): model id we are evaluting for
        run_id (str): model run id to load and evaluate for
        dataset_loc (str): location dataset to evaluate on
        results_loc (str, optional): location to save results. Defaults to ./results directory.
            None if you don't want to save the evaluation

    Returns:
        dict: model performance metrics
    """

    data_handler = model_registration.getDataHandler(model_to_eval)(dataset_loc)
    metric_handler = model_registration.getMetricHandler(model_to_eval)()
    assert isinstance(data_handler, DataHandlerInterface)
    assert isinstance(metric_handler, MetricHandlerInterface)

    return internal_evaluate(model_to_eval, run_id, data_handler, metric_handler, results_loc)


def internal_evaluate(
    model_to_eval: str,
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
        utils.create_dir(evalution_results_loc)
    else:
        evalution_results_loc = None

    logger.info(
        f"Evaluating model {model_to_eval} for run {run_id}. Results saved to {evalution_results_loc}"
    )

    data = data_handler.get_data()
    best_checkpoint = utils.get_best_checkpoint(run_id)
    predictor = TorchPredictor.from_checkpoint(best_checkpoint)

    # metrics
    metrics = {
        "timestamp": utils.get_readable_timestamp(),
        "run_id": run_id,
        "metrics": metric_handler.generate_metrics(data, predictor),
    }
    logger.info(json.dumps(metrics, indent=2))

    if evalution_results_loc:
        utils.save_dict(
            d=metrics,
            path=evalution_results_loc,
            filename=f"performance_results_{utils.get_filepath_timestamp()}.json",
        )
    return metrics


if __name__ == "__main__":
    app()
