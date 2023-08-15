import json
from pathlib import Path

import ray
import typer
from ray.air.config import (
    CheckpointConfig,
    DatasetConfig,
    RunConfig,
    ScalingConfig,
)
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.data import Preprocessor
from ray.train.torch import TorchTrainer
from typing_extensions import Annotated

from config.config import MLFLOW_TRACKING_URI, RESULTS_DIR, logger
from ml.api.data_handler_interface import DataHandlerInterface
from ml.api.model_interface import ModelInterface
from pipeline import evaluate, model_registration, utils

# Initialize Typer CLI app
app = typer.Typer()


@app.command()
def train_model(
    model_to_train: Annotated[str, typer.Option(help="name of model that we wish to train for")],
    dataset_loc: Annotated[str, typer.Option(help="location of the dataset.")],
    train_loop_config: Annotated[
        str,
        typer.Option(
            help="arguments to use for training. Can be either a json string or filepath to json file"
        ),
    ],
    num_workers: Annotated[int, typer.Option(help="number of workers to use for training.")] = 1,
    cpu_per_worker: Annotated[int, typer.Option(help="number of CPUs to use per worker.")] = 1,
    gpu_per_worker: Annotated[int, typer.Option(help="number of GPUs to use per worker.")] = 0,
    num_samples: Annotated[int, typer.Option(help="number of samples to use from dataset.")] = None,
    num_epochs: Annotated[int, typer.Option(help="number of epochs to train for.")] = 1,
    batch_size: Annotated[int, typer.Option(help="number of samples per batch.")] = 256,
    eval_model: Annotated[
        bool,
        typer.Option(help="flag on whether to run metric evaluation on the newly trained model"),
    ] = False,
    results_loc: Annotated[str, typer.Option(help="filepath to save results to.")] = RESULTS_DIR,
) -> ray.air.result.Result:
    """Main train function to train our model as a distributed workload.

    Args:
        model_to_train (str): name of model that we wish to train for. See model_registration.py for ids
        dataset_loc (str): location of the dataset.
        train_loop_config (str): arguments to use for training.
        num_workers (int, optional): number of workers to use for training. Defaults to 1.
        cpu_per_worker (int, optional): number of CPUs to use per worker. Defaults to 1.
        gpu_per_worker (int, optional): number of GPUs to use per worker. Defaults to 0.
        num_samples (int, optional): number of samples to use from dataset.
            If this is passed in, it will override the config. Defaults to None.
        num_epochs (int, optional): number of epochs to train for.
            If this is passed in, it will override the config. Defaults to None.
        batch_size (int, optional): number of samples per batch.
            If this is passed in, it will override the config. Defaults to None.
        eval_model (bool, optional): flag on whether to run metric evaluation on the newly trained model. Defaults to False
        results_loc (str, optional): location to save results and ray checkpoints to. Defaults to ./results directory.
            None if you don't want to save training results json, checkpoint results default to ~/ray_results.

    Returns:
        ray.air.result.Result: training results.
    """
    # Set up
    if results_loc:
        training_result_loc = Path(
            results_loc, "training", model_to_train, utils.get_filepath_timestamp()
        )
        utils.create_dir(training_result_loc)
    else:
        training_result_loc = None

    try:
        train_loop_config = json.loads(train_loop_config)
    except ValueError:
        train_loop_config = utils.load_dict(train_loop_config)
    except:  # noqa: E722
        raise TypeError("train_loop_config must be a stringified json or a path to a json file")

    train_loop_config["num_samples"] = num_samples
    train_loop_config["num_epochs"] = num_epochs
    train_loop_config["batch_size"] = batch_size

    training_model = model_registration.getModelFactory(model_to_train)()
    preprocessor = model_registration.getPreprocessorFactory(model_to_train)()
    data_handler = model_registration.getDataHandler(model_to_train)(dataset_loc, num_samples)
    assert isinstance(training_model, ModelInterface)
    assert isinstance(preprocessor, Preprocessor)
    assert isinstance(data_handler, DataHandlerInterface)

    utils.set_seeds()

    logger.info(
        f"Setting up training for {model_to_train} - Dataset location {dataset_loc} - Results saved to {training_result_loc} - Training config\n{json.dumps(train_loop_config, indent=2)}"
    )

    # Scaling config
    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=bool(gpu_per_worker),
        resources_per_worker={"CPU": cpu_per_worker, "GPU": gpu_per_worker},
        _max_cpu_fraction_per_node=0.8,
    )

    # Checkpoint config
    checkpoint_config = CheckpointConfig(
        num_to_keep=1,
        checkpoint_score_attribute="val_loss",
        checkpoint_score_order="min",
    )

    # MLflow callback
    mlflow_callback = MLflowLoggerCallback(
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=model_to_train,
        save_artifact=True,
    )

    # Run config
    run_config = RunConfig(
        callbacks=[mlflow_callback],
        checkpoint_config=checkpoint_config,
        storage_path=training_result_loc,
    )

    # Dataset
    logger.info("Loading training data")
    train_ds, eval_ds = data_handler.split_data(test_size=0.2)
    train_loop_config = data_handler.add_to_config(train_loop_config)

    # Dataset config
    dataset_config = {
        "train": DatasetConfig(fit=False, transform=False, randomize_block_order=False),
        "eval": DatasetConfig(fit=False, transform=False, randomize_block_order=False),
    }

    # Preprocess
    logger.info("Preprocessing data")
    train_ds = preprocessor.fit_transform(train_ds)
    eval_ds = preprocessor.transform(eval_ds)
    train_ds = train_ds.materialize()
    eval_ds = eval_ds.materialize()

    # Trainer
    trainer = TorchTrainer(
        train_loop_per_worker=training_model.train_loop_per_worker,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config,
        run_config=run_config,
        datasets={"train": train_ds, "eval": eval_ds},
        dataset_config=dataset_config,
        preprocessor=preprocessor,
    )

    # Train
    logger.info("Starting training session")
    results = trainer.fit()
    run_id = utils.get_run_id(experiment_name=model_to_train, trial_id=results.metrics["trial_id"])
    results_data = {
        "timestamp": utils.get_readable_timestamp(),
        "run_id": run_id,
        "params": results.config["train_loop_config"],
        "metrics": utils.dict_to_list(
            results.metrics_dataframe.to_dict(), keys=["epoch", "train_loss", "val_loss"]
        ),
    }
    logger.info(json.dumps(results_data, indent=2))
    if training_result_loc:  # pragma: no cover, saving results
        utils.save_dict(results_data, training_result_loc, "training_results.json")

    if eval_model:
        metric_handler = model_registration.getMetricHandler(model_to_train)()
        evaluate.internal_evaluate(
            model_to_train, run_id, data_handler, metric_handler, results_loc
        )

    return results


if __name__ == "__main__":  # pragma: no cover, application
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    app()
