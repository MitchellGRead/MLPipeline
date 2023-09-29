import json
import os
from pathlib import Path

import ray
import torch
import torch_geometric as pyg
import typer
from pipeline_utils import (
    convert_utils,
    os_utils,
    rand_utils,
    time_utils,
    wb_utils,
)
from ray.air.config import (
    CheckpointConfig,
    DatasetConfig,
    RunConfig,
    ScalingConfig,
)
from ray.data import Preprocessor
from ray.train.torch import TorchTrainer
from tqdm import tqdm
from typing_extensions import Annotated

import wandb
from config.config import RESULTS_DIR, logger
from ml.api.data_handler_interface import DataHandlerInterface
from ml.api.model_interface import ModelInterface
from ml.data_handler.complex_physics_gns import OneStepDataset, RolloutDataset
from ml.metric.complex_physics_gns import oneStepMSE, rolloutMSE
from ml.model.complex_physics_gns import LearnedSimulator
from ml.rollout.complex_physics_gns import rollout
from ml.visualize.complex_physics_gns import visualize_pair
from pipeline import artifacts, evaluate, model_registration
from pipeline.wandb_manager import WandbManager

# Initialize Typer CLI app
app = typer.Typer()


def download_dataset(dataset_loc: str, track_dataset: str, model_id: str) -> tuple[str, str]:
    # Convert local into W&B
    if os_utils.is_file_or_dir(dataset_loc) and track_dataset:
        [status, name] = artifacts.process_dataset(
            dataset_loc=dataset_loc, data_type="raw_data", data_for_model_id=model_id
        )

        if not status:
            msg = f"Failed to process the dataset {dataset_loc} into weights and biases"
            logger.error(msg)
            raise ValueError(msg)

        return dataset_loc, name

    # Doing a simple local run
    if os_utils.is_file_or_dir(dataset_loc) and not track_dataset:
        return dataset_loc, None

    # Using dataset already in W&B and we get a dataset alias
    is_artifact_name = wb_utils.verify_artifact_name(dataset_loc)
    if not is_artifact_name:
        msg = "Artifact names should be a <name>:<alias> format. Check if your name is correct or if you instead want to only use a local dataset."
        logger.error(msg)
        raise ValueError(msg)

    # Setup to pull artifact
    model_project = model_registration.getModelProject(model_id)
    wb_path = wb_utils.get_object_path(model_project.value, dataset_loc)
    api = wandb.Api()

    try:
        artifact = api.artifact(wb_path)
    except ValueError:
        msg = f"Could not find an artifact at {wb_path}. Are you sure one exists?"
        logger.error(msg)
        raise ValueError(msg)

    # Download locally
    artifact_dir = artifact.download()
    return artifact_dir, artifact.name


@app.command()
def gns_train_model(
    dataset_loc: Annotated[str, typer.Option(help="Path to dataset in local storage")],
    train_loop_config: Annotated[
        str, typer.Option(help="Path to .json or stringified json object")
    ],
    eval_interval: Annotated[int, typer.Option(help="Interval to eval during training")] = 1,
    vis_interval: Annotated[int, typer.Option(help="Interval to visualize during training")] = 1,
    save_interval: Annotated[
        int, typer.Option(help="Interval to save artifacts during training")
    ] = 1,
    num_workers: Annotated[int, typer.Option(help="Number of workers for training")] = 2,
    batch_size: Annotated[int, typer.Option(help="Batch size to compute data")] = 256,
    seed: Annotated[int, typer.Option(help="Seed for reproducibility")] = 42,
    results_loc: Annotated[str, typer.Option(help="Path to results")] = RESULTS_DIR,
):
    model_to_train = "224w-GNS"
    training_result_loc = get_training_results_loc(results_loc, model_to_train)
    train_loop_config = read_config(train_loop_config)

    rand_utils.set_seeds(seed)
    train_loop_config["seed"] = seed

    logger.info(
        f"Setting up training for {model_to_train} - Dataset location {dataset_loc} - Results saved to {training_result_loc} - Training config\n{json.dumps(train_loop_config, indent=2)}"
    )

    logger.info("Loading datasets")
    train_dataset = OneStepDataset(dataset_loc, "train", noise_std=train_loop_config["noise"])[:10]
    valid_dataset = OneStepDataset(dataset_loc, "valid", noise_std=train_loop_config["noise"])[:1]
    train_loader = pyg.loader.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers
    )
    valid_loader = pyg.loader.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    rollout_dataset = RolloutDataset(dataset_loc, "valid")

    logger.info("Loading training model")
    simulator = LearnedSimulator()
    # simulator = simulator.cuda()

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(simulator.parameters(), lr=train_loop_config["lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1 ** (1 / 5e6))

    logger.info("Init W&B")
    run_id = wandb.util.generate_id()
    run_name = f"PygTrainer_{run_id}"

    run = wandb.init(
        project="GNS",
        job_type="train",
        group=model_to_train,
        name=run_name,
        config=train_loop_config,
    )

    run.define_metric("valid_step")
    run.define_metric("rollout_step")
    run.define_metric("valid_loss", step_metric="valid_step", goal="minimize")
    run.define_metric("onestep_mse", step_metric="valid_step", goal="minimize")
    run.define_metric("rollout_mse", step_metric="rollout_step", goal="minimize")

    logger.info("Starting training loop")
    total_batch = 0
    for epoch in range(train_loop_config["epoch"]):
        simulator.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        total_loss = 0
        batch_count = 0
        for data in progress_bar:
            optimizer.zero_grad()
            # data = data.cuda()
            pred = simulator(data)
            loss = loss_fn(pred, data.y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            batch_count += 1

            # track train metrics
            train_metrics = {
                "loss": loss.item(),
                "avg_loss": total_loss / batch_count,
                "lr": optimizer.param_groups[0]["lr"],
            }
            progress_bar.set_postfix(train_metrics)
            run.log(train_metrics)

            total_batch += 1

            # evaluation
            if eval_interval and total_batch % eval_interval == 0:
                logger.info(f"Running evaluation at {total_batch}")
                eval_loss, onestep_mse = oneStepMSE(
                    simulator, valid_loader, valid_dataset.metadata, train_loop_config["noise"]
                )

                # track eval metrics
                eval = {
                    "valid_loss": eval_loss,
                    "onestep_mse": onestep_mse,
                    "valid_step": total_batch,
                }
                run.log(eval)
                print(f"Eval loss: {total_loss / batch_count}")
                simulator.train()

            # Rollout on valid dataset
            if vis_interval and total_batch % vis_interval == 0:
                logger.info(f"Running rollout at {total_batch}")
                simulator.eval()
                rollout_mse = rolloutMSE(simulator, rollout_dataset, train_loop_config["noise"])

                # track W&B metrics
                rollout_eval = {
                    "rollout_mse": rollout_mse,
                    "rollout_step": total_batch,
                }
                run.log(rollout_eval)

                rollout_data = rollout_dataset[0]
                rollout_out = rollout(
                    simulator, rollout_data, rollout_dataset.metadata, train_loop_config["noise"]
                )
                rollout_out = rollout_out.permute(1, 0, 2)

                # create animation
                anim = visualize_pair(
                    rollout_data["particle_type"],
                    rollout_out,
                    rollout_data["position"],
                    rollout_dataset.metadata,
                )
                anim_path = os.path.join(
                    training_result_loc, f"rollout_{total_batch}_{run_name}.gif"
                )
                fps = 60
                anim.save(
                    anim_path,
                    writer="ffmpeg",
                    fps=fps,
                )
                run.log(
                    {
                        "rollout_video": wandb.Video(
                            anim_path,
                            caption=f"Rollout After {total_batch} Training Steps",
                            fps=fps,
                            format="mp4",
                        )
                    }
                )
                simulator.train()

            if save_interval and total_batch % save_interval == 0:
                checkpoint_name = f"checkpoint_{total_batch}_{run_name}.pt"
                model_checkpoint_path = os.path.join(training_result_loc, checkpoint_name)
                torch.save(
                    {
                        "model": simulator.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "train_loop_config": train_loop_config,
                    },
                    model_checkpoint_path,
                )
                artifact = wandb.Artifact(name=checkpoint_name, type="model")
                artifact.add_file(model_checkpoint_path)
                run.log_artifact(artifact)

    run.finish()


def read_config(config: str) -> dict[str, any]:
    try:
        return json.loads(config)
    except ValueError:
        return os_utils.load_dict(config)
    except:  # noqa: E722
        raise TypeError("config must be a stringified json or a path to a json file")


def get_training_results_loc(results_loc: str, model_to_train: str) -> str | None:
    if results_loc:
        training_result_loc = Path(
            results_loc, "training", model_to_train, time_utils.get_filepath_timestamp()
        )
        os_utils.create_dir(training_result_loc)
        return training_result_loc
    else:
        return None


@app.command()
def ray_train_model(
    model_to_train: Annotated[str, typer.Option(help="name of model that we wish to train for")],
    dataset_loc: Annotated[
        str,
        typer.Option(
            help="location of the dataset. Can be a local file path or a W&B <name>:<alias>"
        ),
    ],
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
    only_keep_latest: Annotated[
        bool,
        typer.Option(
            "--keep-latest/--keep-all," "-k/-K",
            help="flag to only save the latest checkpoint in W&B.",
        ),
    ] = True,
    track_dataset: Annotated[
        bool,
        typer.Option(
            "--track-data/--keep-local," "-t/-T",
            help="flag to track the runs dataset into W&B. If an artifact already exists in W&B with the same name, then a new version is created.",
        ),
    ] = False,
    results_loc: Annotated[str, typer.Option(help="filepath to save results to.")] = RESULTS_DIR,
) -> ray.air.result.Result:
    """Main train function to train our model as a distributed workload.

    Args:
        model_to_train (str): name of model that we wish to train for. See model_registration.py for ids
        dataset_loc (str): location of the dataset. Can be a local file path or a W&B <name>:<alias>
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
        only_keep_latest (bool, optional): only save the latest checkpoint in W&B. Defaults to True
        track_dataset (bool, optional): flag to track the runs dataset into W&B. If an artifact already exists in W&B with the same name, then a new version is created. Default False
        results_loc (str, optional): location to save results and ray checkpoints to. Defaults to ./results directory.

    Returns:
        ray.air.result.Result: training results.
    """
    # Set up
    training_result_loc = get_training_results_loc(results_loc)
    train_loop_config = read_config(train_loop_config)

    train_loop_config["num_samples"] = num_samples
    train_loop_config["num_epochs"] = num_epochs
    train_loop_config["batch_size"] = batch_size

    [local_dataset_loc, wb_dataset_name] = download_dataset(
        dataset_loc, track_dataset, model_to_train
    )

    training_model = model_registration.getModelFactory(model_to_train)()
    preprocessor = model_registration.getPreprocessorFactory(model_to_train)()
    data_handler = model_registration.getDataHandler(model_to_train)(local_dataset_loc, num_samples)
    model_project = model_registration.getModelProject(model_to_train)
    assert isinstance(training_model, ModelInterface)
    assert isinstance(preprocessor, Preprocessor)
    assert isinstance(data_handler, DataHandlerInterface)
    assert isinstance(model_project, model_registration.ModelProject)

    rand_utils.set_seeds()
    weights_and_biases_api_key = os_utils.get_env_value("WEIGHT_AND_BIASES_API_KEY")

    logger.info(
        f"Setting up training for {model_to_train} - Dataset location {dataset_loc} - W&B Name {wb_dataset_name} - Results saved to {training_result_loc} - Training config\n{json.dumps(train_loop_config, indent=2)}"
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
        num_to_keep=1 if only_keep_latest else None,
        checkpoint_score_attribute="val_loss",
        checkpoint_score_order="min",
    )

    # Run config
    run_config = RunConfig(
        storage_path=training_result_loc,
        checkpoint_config=checkpoint_config,
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

    # Training Manager
    train_loop_config["manager"] = lambda: WandbManager(
        config=train_loop_config,
        project=model_project.value,
        group=model_to_train,
        api_key=weights_and_biases_api_key,
        job_type="train",
        use_artifact=wb_dataset_name,
        preprocessor=preprocessor,
    )

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

    del train_loop_config["manager"]
    del results.config["train_loop_config"]["manager"]

    run_id = results.metrics["trial_id"]
    results_data = {
        "timestamp": time_utils.get_readable_timestamp(),
        "run_name": wb_utils.get_run_name(run_id),
        "run_id": run_id,
        "params": results.config["train_loop_config"],
        "metrics": convert_utils.dict_to_list(
            results.metrics_dataframe.to_dict(), keys=["epoch", "train_loss", "val_loss"]
        ),
    }

    logger.info(json.dumps(results_data, indent=2))

    if training_result_loc:  # pragma: no cover, saving results
        os_utils.save_dict(results_data, training_result_loc, "training_results.json")

    if only_keep_latest:
        artifacts.delete_run_artifacts(
            model_to_delete_for=model_to_train, run_id=run_id, artifact_type="model"
        )

    if eval_model:
        metric_handler = model_registration.getMetricHandler(model_to_train)()
        evaluate.internal_evaluate(
            model_to_eval=model_to_train,
            model_project=model_project,
            run_id=run_id,
            data_handler=data_handler,
            metric_handler=metric_handler,
            results_loc=results_loc,
        )

    return results


if __name__ == "__main__":  # pragma: no cover, application
    # if ray.is_initialized():
    #     ray.shutdown()
    # ray.init()
    app()
