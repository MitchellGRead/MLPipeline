import datetime
import json
from datetime import datetime
from pathlib import Path

import ray
import typer
from ray import tune
from ray.air.config import (
    CheckpointConfig,
    DatasetConfig,
    RunConfig,
    ScalingConfig,
)
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.train.torch import TorchTrainer
from ray.tune import Tuner
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch
from typing_extensions import Annotated

from config.config import MLFLOW_TRACKING_URI, RESULTS_DIR, logger
from tagifai import data, train, utils

# Initialize Typer CLI app
app = typer.Typer()


@app.command()
def tune_models(
    experiment_name: Annotated[
        str, typer.Option(help="name of the experiment for this training workload.")
    ] = None,
    dataset_loc: Annotated[str, typer.Option(help="location of the dataset.")] = None,
    initial_params: Annotated[
        str, typer.Option(help="initial config for the tuning workload.")
    ] = None,
    num_workers: Annotated[int, typer.Option(help="number of workers to use for training.")] = 1,
    cpu_per_worker: Annotated[int, typer.Option(help="number of CPUs to use per worker.")] = 1,
    gpu_per_worker: Annotated[int, typer.Option(help="number of GPUs to use per worker.")] = 0,
    num_runs: Annotated[int, typer.Option(help="number of runs in this tuning experiment.")] = 1,
    num_samples: Annotated[int, typer.Option(help="number of samples to use from dataset.")] = None,
    num_epochs: Annotated[int, typer.Option(help="number of epochs to train for.")] = 1,
    batch_size: Annotated[int, typer.Option(help="number of samples per batch.")] = 256,
    results_loc: Annotated[str, typer.Option(help="filepath to save results to.")] = RESULTS_DIR,
) -> ray.tune.result_grid.ResultGrid:
    """Main tuning function to tune hyperparameters.

    Args:
        experiment_name (str): name of the experiment for this training workload.
        dataset_loc (str): location of the dataset.
        initial_params (str): initial config for the tuning workload.
        num_workers (int, optional): number of workers to use for training. Defaults to 1.
        cpu_per_worker (int, optional): number of CPUs to use per worker. Defaults to 1.
        gpu_per_worker (int, optional): number of GPUs to use per worker. Defaults to 0.
        num_runs (int, optional): number of runs in this tuning experiment. Defaults to 1.
        num_samples (int, optional): number of samples to use from dataset.
            If this is passed in, it will override the config. Defaults to None.
        num_epochs (int, optional): number of epochs to train for.
            If this is passed in, it will override the config. Defaults to None.
        batch_size (int, optional): number of samples per batch.
            If this is passed in, it will override the config. Defaults to None.
        results_loc (str, optional): location to save results and ray checkpoints to. Defaults to ./results directory.
            None if you don't want to save training results json, checkpoint results default to ~/ray_results.

    Returns:
        ray.tune.result_grid.ResultGrid: results of the tuning experiment.
    """
    # Set up
    if results_loc:
        tuning_result_loc = Path(
            results_loc, "tuning", experiment_name, datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        )
        utils.create_dir(tuning_result_loc)
    else:
        tuning_result_loc = None
    utils.set_seeds()
    train_loop_config = {}
    train_loop_config["num_samples"] = num_samples
    train_loop_config["num_epochs"] = num_epochs
    train_loop_config["batch_size"] = batch_size
    logger.info(
        f"Setting up tuning for {experiment_name} - Dataset location {dataset_loc} - Results saved to {tuning_result_loc} - Training config\n{json.dumps(train_loop_config, indent=2)}"
    )

    # Scaling config
    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=bool(gpu_per_worker),
        resources_per_worker={"CPU": cpu_per_worker, "GPU": gpu_per_worker},
        _max_cpu_fraction_per_node=0.8,
    )

    # Checkpoint configuration
    checkpoint_config = CheckpointConfig(
        num_to_keep=1,
        checkpoint_score_attribute="val_loss",
        checkpoint_score_order="min",
    )

    # MLflow callback
    mlflow_callback = MLflowLoggerCallback(
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=experiment_name,
        save_artifact=True,
    )

    # Run configuration
    run_config = RunConfig(
        callbacks=[mlflow_callback],
        checkpoint_config=checkpoint_config,
        storage_path=tuning_result_loc,
    )

    # Dataset
    logger.info("Loading training data")
    ds = data.load_data(dataset_loc=dataset_loc, num_samples=train_loop_config["num_samples"])
    train_ds, val_ds = data.stratify_split(ds, stratify="tag", test_size=0.2)
    tags = train_ds.unique(column="tag")
    train_loop_config["num_classes"] = len(tags)

    # Dataset config
    dataset_config = {
        "train": DatasetConfig(fit=False, transform=False, randomize_block_order=False),
        "val": DatasetConfig(fit=False, transform=False, randomize_block_order=False),
    }

    # Preprocess
    logger.info("Preprocessing data")
    preprocessor = data.CustomPreprocessor()
    train_ds = preprocessor.fit_transform(train_ds)
    val_ds = preprocessor.transform(val_ds)
    train_ds = train_ds.materialize()
    val_ds = val_ds.materialize()

    # Trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train.train_loop_per_worker,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config,
        datasets={"train": train_ds, "val": val_ds},
        dataset_config=dataset_config,
        preprocessor=preprocessor,
    )

    # Hyperparameters to start with
    initial_params = json.loads(initial_params)
    logger.info(f"Initial hyper params to start with\n{json.dumps(initial_params, indent=2)}")
    search_alg = HyperOptSearch(points_to_evaluate=initial_params)
    search_alg = ConcurrencyLimiter(
        search_alg, max_concurrent=2
    )  # trade off b/w optimization and search space

    # Parameter space - Range to look for params
    param_space = {
        "train_loop_config": {
            "dropout_p": tune.uniform(0.3, 0.9),
            "lr": tune.loguniform(1e-5, 5e-4),
            "lr_factor": tune.uniform(0.1, 0.9),
            "lr_patience": tune.uniform(1, 10),
        }
    }

    # Scheduler - interrupts and prunes unpromising trials
    # HyperBand is quite aggressive so we set a min grace period of epochs
    scheduler = AsyncHyperBandScheduler(
        max_t=train_loop_config["num_epochs"],  # max epoch (<time_attr>) per trial
        grace_period=2,  # min epoch (<time_attr>) per trial
    )

    # Tune config
    tune_config = tune.TuneConfig(
        metric="val_loss",
        mode="min",
        search_alg=search_alg,
        scheduler=scheduler,
        num_samples=num_runs,
    )

    # Tuner
    tuner = Tuner(
        trainable=trainer,
        run_config=run_config,
        param_space=param_space,
        tune_config=tune_config,
    )

    # Tune
    logger.info("Starting tuning session")
    results = tuner.fit()
    best_trial = results.get_best_result(metric="val_loss", mode="min")
    tuning_data = {
        "timestamp": datetime.now().strftime("%B %d, %Y %I:%M:%S %p"),
        "run_id": utils.get_run_id(
            experiment_name=experiment_name, trial_id=best_trial.metrics["trial_id"]
        ),
        "params": best_trial.config["train_loop_config"],
        "metrics": utils.dict_to_list(
            best_trial.metrics_dataframe.to_dict(), keys=["epoch", "train_loss", "val_loss"]
        ),
    }
    logger.info(json.dumps(tuning_data, indent=2))
    if tuning_result_loc:  # pragma: no cover, saving results
        utils.save_dict(tuning_data, tuning_result_loc, "tuning_results.json")
    return results


if __name__ == "__main__":  # pragma: no cover, application
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    app()
