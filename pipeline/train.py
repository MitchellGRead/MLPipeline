import json
import os
import random
from pathlib import Path

import torch
import torch.optim as optim
import torch_geometric as pyg
import torch_geometric.data as pyg_data
import typer
from pipeline_utils import os_utils, rand_utils, time_utils
from tqdm import tqdm, trange
from typing_extensions import Annotated

import ml.metric.mesh_graph_net as mesh_metric
import ml.model.mesh_graph_net as mesh_model
import ml.preprocessor.mesh_graph_net as mesh_preprocessor
import ml.visualize.mesh_graph_net as mesh_visualize
import wandb
from config.config import RESULTS_DIR, WEIGHTS_AND_BIASES_PROJECT, logger
from ml.data_handler.complex_physics_gns import OneStepDataset, RolloutDataset
from ml.metric.complex_physics_gns import oneStepMSE, rolloutMSE
from ml.model.complex_physics_gns import LearnedSimulator
from ml.rollout.complex_physics_gns import rollout
from ml.visualize.complex_physics_gns import visualize_pair

# Initialize Typer CLI app
app = typer.Typer()

os.environ["WANDB_API_KEY"] = os_utils.get_env_value("WEIGHT_AND_BIASES_API_KEY")
os.environ["WANDB_USERNAME"] = os_utils.get_env_value("WEIGHT_AND_BIASES_USERNAME")

device = "cuda" if torch.cuda.is_available() else "cpu"


def build_optimizer(
    optimizer: str,
    scheduler: str,
    weight_decay: float,
    scheduler_iters: int,
    schedular_decay_rate: float,
    scheduler_step_size: int,
    lr: float,
    params: dict,
):
    filter_fn = filter(lambda p: p.requires_grad, params)
    if optimizer == "adam":
        optimizer = optim.Adam(filter_fn, lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        optimizer = optim.SGD(filter_fn, lr=lr, momentum=0.95, weight_decay=weight_decay)
    elif optimizer == "rmsprop":
        optimizer = optim.RMSprop(filter_fn, lr=lr, weight_decay=weight_decay)
    elif optimizer == "adagrad":
        optimizer = optim.Adagrad(filter_fn, lr=lr, weight_decay=weight_decay)

    if scheduler == "none":
        return None, optimizer
    elif scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler_step_size, gamma=schedular_decay_rate
        )
    elif scheduler == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_iters)
    return scheduler, optimizer


@app.command()
def mesh_train_model(
    dataset_loc: Annotated[str, typer.Option(help="Path to dataset in local storage")],
    train_loop_config: Annotated[
        str, typer.Option(help="Path to .json or stringified json object")
    ],
    eval_interval: Annotated[int, typer.Option(help="Interval to eval during training")] = 100,
    vis_interval: Annotated[int, typer.Option(help="Interval to visualize during training")] = 2500,
    save_interval: Annotated[
        int, typer.Option(help="Interval to save artifacts during training")
    ] = 1000,
    optimizer: Annotated[
        str,
        typer.Option(
            help="Optimizer to use for training. Options are: adam, sgd, rmsprop, adagrad"
        ),
    ] = "adam",
    scheduler: Annotated[
        str, typer.Option(help="Scheduler to use for training. Options are: none, step, cos")
    ] = "step",
    shuffle: Annotated[
        bool, typer.Option("-s/-S", help="Shuffle dataset before data split and training")
    ] = True,
    num_workers: Annotated[int, typer.Option(help="Number of workers for training")] = 2,
    batch_size: Annotated[int, typer.Option(help="Batch size to compute data")] = 16,
    train_size: Annotated[int, typer.Option(help="Size of train data split")] = 45,
    test_size: Annotated[int, typer.Option(help="Size of test data split")] = 5,
    seed: Annotated[int, typer.Option(help="Seed for reproducibility")] = 42,
    results_loc: Annotated[str, typer.Option(help="Path to results")] = RESULTS_DIR,
):
    model_to_train = "Mesh-Graph-Net"
    training_result_loc = get_training_results_loc(results_loc, model_to_train)
    train_loop_config = read_config(train_loop_config)

    rand_utils.set_seeds(seed)
    train_loop_config["seed"] = seed
    train_loop_config["optimizer"] = optimizer
    train_loop_config["scheduler"] = scheduler

    logger.info(
        f"Setting up training for {model_to_train} - Dataset location {dataset_loc} - Results saved to {training_result_loc} - Training config\n{json.dumps(train_loop_config, indent=2)}"
    )

    logger.info("Loading datasets")
    dataset = torch.load(dataset_loc)[: (train_size + test_size)]
    if shuffle:
        random.shuffle(dataset)

    stats_list = mesh_preprocessor.get_stats(dataset)
    [mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y] = stats_list
    (mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y) = (
        mean_vec_x.to(device),
        std_vec_x.to(device),
        mean_vec_edge.to(device),
        std_vec_edge.to(device),
        mean_vec_y.to(device),
        std_vec_y.to(device),
    )

    loader = pyg_data.DataLoader(
        dataset[:train_size], batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = pyg_data.DataLoader(
        dataset[train_size:], batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    logger.info("Loading training model")
    num_node_features = dataset[0].x.shape[1]
    num_edge_features = dataset[0].edge_attr.shape[1]
    num_classes = 2  # the dynamic variables have the shape of 2 (velocity)

    model = mesh_model.MeshGraphNet(
        num_node_features,
        num_edge_features,
        train_loop_config["hidden_dim"],
        num_classes,
        train_loop_config["num_layers"],
    ).to(device)

    scheduler, optimizer = build_optimizer(
        optimizer=optimizer,
        scheduler=scheduler,
        weight_decay=train_loop_config["weight_decay"],
        scheduler_iters=train_loop_config["scheduler_iters"],
        schedular_decay_rate=0.1 ** (1 / 5e6),
        scheduler_step_size=30,
        lr=train_loop_config["lr"],
        params=model.parameters(),
    )

    logger.info("Init W&B")
    run_id = wandb.util.generate_id()
    run_name = f"PygTrainer_{run_id}"

    run = wandb.init(
        entity=WEIGHTS_AND_BIASES_PROJECT,
        project="GNS",
        job_type="train",
        group=model_to_train,
        name=run_name,
        config=train_loop_config,
    )

    run.define_metric("loss", goal="minimize")
    run.define_metric("avg_loss", goal="minimize")
    run.define_metric("valid_loss", goal="minimize")
    run.define_metric("velocity_loss", goal="minimize")

    logger.info("Starting training loop")
    for epoch in trange(train_loop_config["epoch"], desc="Training", unit="Epochs"):
        total_loss = 0
        model.train()
        num_loops = 0

        for batch in loader:
            # Note that normalization must be done before it's called. The unnormalized
            # data needs to be preserved in order to correctly calculate the loss
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)
            loss = model.loss(pred, batch, mean_vec_y, std_vec_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_loops += 1

            # Train Metrics
            train_metrics = {
                "loss": loss.item(),
                "avg_loss": total_loss / num_loops,
                "lr": optimizer.param_groups[0]["lr"],
            }
            run.log(train_metrics)

        total_loss /= num_loops

        # Evaluation
        if epoch % eval_interval == 0:
            logger.info("Evaluating model")
            model.eval()
            valid_loss, velo_val_rmse = mesh_metric.evaluate(
                test_loader,
                device,
                model,
                mean_vec_x,
                std_vec_x,
                mean_vec_edge,
                std_vec_edge,
                mean_vec_y,
                std_vec_y,
            )

            eval = {"velocity_loss": velo_val_rmse, "valid_loss": valid_loss, "epoch": epoch}
            run.log(eval)
            model.train()

        # Visualize
        if epoch % vis_interval == 0:
            logger.info("Visualizing model")
            model.eval()
            anim = mesh_visualize.visualize(dataset, model, stats_list, delta_t=0.01, skip=1)
            anim_path = os.path.join(
                training_result_loc, f"mesh_graph_net-x_velocity-{run_name}.gif"
            )
            fps = 10
            # writergif = animation.PillowWriter(fps=fps)
            anim.save(anim_path, writer="ffmpeg", fps=fps)
            run.log(
                {
                    "rollout_video": wandb.Video(
                        anim_path,
                        fps=fps,
                        format="mp4",
                    )
                }
            )
            model.train()

        # Save Model
        if epoch % save_interval == 0:
            logger.info("Saving model")
            checkpoint_name = f"checkpoint_{run_name}.pt"
            model_checkpoint_path = os.path.join(training_result_loc, checkpoint_name)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if (scheduler) else {},
                    "train_loop_config": train_loop_config,
                },
                model_checkpoint_path,
            )
            artifact = wandb.Artifact(name=checkpoint_name, type="model")
            artifact.add_file(model_checkpoint_path)
            run.log_artifact(artifact)

    run.finish()


@app.command()
def gns_train_model(
    dataset_loc: Annotated[str, typer.Option(help="Path to dataset in local storage")],
    train_loop_config: Annotated[
        str, typer.Option(help="Path to .json or stringified json object")
    ] = None,
    eval_interval: Annotated[int, typer.Option(help="Interval to eval during training")] = 20000,
    vis_interval: Annotated[
        int, typer.Option(help="Interval to visualize during training")
    ] = 20000,
    save_interval: Annotated[
        int, typer.Option(help="Interval to save artifacts during training")
    ] = 20000,
    num_workers: Annotated[int, typer.Option(help="Number of workers for training")] = 2,
    batch_size: Annotated[int, typer.Option(help="Batch size to compute data")] = 2,
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
    train_dataset = OneStepDataset(dataset_loc, "train", noise_std=train_loop_config["noise"])
    valid_dataset = OneStepDataset(dataset_loc, "valid", noise_std=train_loop_config["noise"])
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
    simulator = simulator.cuda()

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(simulator.parameters(), lr=train_loop_config["lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1 ** (1 / 5e6))

    logger.info("Init W&B")
    run_id = wandb.util.generate_id()
    run_name = f"PygTrainer_{run_id}"

    run = wandb.init(
        entity=WEIGHTS_AND_BIASES_PROJECT,
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
            data = data.cuda()
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
                checkpoint_name = f"checkpoint_{run_name}.pt"
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


if __name__ == "__main__":  # pragma: no cover, application
    # if ray.is_initialized():
    #     ray.shutdown()
    # ray.init()
    os.environ["WANDB_API_KEY"] = os_utils.get_env_value("WEIGHT_AND_BIASES_API_KEY")
    os.environ["WANDB_USERNAME"] = os_utils.get_env_value("WEIGHT_AND_BIASES_USERNAME")
    app()
