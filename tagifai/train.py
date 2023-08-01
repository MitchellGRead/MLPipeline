import json
from argparse import Namespace
from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
import ray
import ray.train as ray_train
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer
from imblearn.over_sampling import RandomOverSampler
from ray.air import session
from ray.air.config import (
    CheckpointConfig,
    DatasetConfig,
    RunConfig,
    ScalingConfig,
)
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.data import Dataset
from ray.train.torch import TorchCheckpoint, TorchTrainer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from transformers import BertModel
from typing_extensions import Annotated

from config.config import ARGS_URI, MLFLOW_TRACKING_URI, logger
from tagifai import data, evaluate, models, predict, utils

# Initialize Typer CLI app
app = typer.Typer()


def train(args, df, trial=None):
    """Train model on data."""

    # Setup
    logger.info(f"Setting up for training. Is trial={trial}")
    utils.set_seeds()
    if args.shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    df = df[: args.subset]  # None = all samples
    df = data.preprocess_data(df, lower=args.lower, stem=args.stem, min_freq=args.min_freq)
    label_encoder = data.LabelEncoder().fit(df.tag)
    X_train, X_val, X_test, y_train, y_val, y_test = data.get_data_splits(
        X=df.text.to_numpy(), y=label_encoder.encode(df.tag)
    )
    test_df = pd.DataFrame({"text": X_test, "tag": label_encoder.decode(y_test)})

    # Tf-idf
    logger.info("Performing transformations")
    vectorizer = TfidfVectorizer(
        analyzer=args.analyzer, ngram_range=(2, args.ngram_max_range)
    )  # char n-grams
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)
    X_test = vectorizer.transform(X_test)

    # Oversample
    oversample = RandomOverSampler(sampling_strategy="all")
    X_over, y_over = oversample.fit_resample(X_train, y_train)

    # Model
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=args.alpha,
        max_iter=1,
        learning_rate="constant",
        eta0=args.learning_rate,
        power_t=args.power_t,
        warm_start=True,
    )

    # Training
    logger.info("Training model")
    for epoch in range(args.num_epochs):
        model.fit(X_over, y_over)
        train_loss = log_loss(y_train, model.predict_proba(X_train))
        val_loss = log_loss(y_val, model.predict_proba(X_val))

        if not epoch % 10:
            logger.info(
                f"Epoch: {epoch:02d} | "
                f"train_loss: {train_loss:.5f}, "
                f"val_loss: {val_loss:.5f}"
            )

        # Log intermediate metrics
        if not trial:
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

    # Threshold
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)
    args.threshold = np.quantile([y_prob[i][j] for i, j in enumerate(y_pred)], q=0.25)  # Q1

    # Evaluation
    other_index = label_encoder.class_to_index["other"]
    y_prob = model.predict_proba(X_test)
    y_pred = predict.custom_predict(y_prob=y_prob, threshold=args.threshold, index=other_index)
    performance = evaluate.get_metrics(
        y_true=y_test, y_pred=y_pred, classes=label_encoder.classes, df=test_df
    )

    return {
        "args": args,
        "label_encoder": label_encoder,
        "vectorizer": vectorizer,
        "model": model,
        "performance": performance,
    }


def objective(args, df, trial):
    """Objective function for optimization trials."""
    # Parameters to tune
    args.analyzer = trial.suggest_categorical("analyzer", ["word", "char", "char_wb"])
    args.ngram_max_range = trial.suggest_int("ngram_max_range", 3, 10)
    args.learning_rate = trial.suggest_loguniform("learning_rate", 1e-2, 1e0)
    args.power_t = trial.suggest_uniform("power_t", 0.1, 0.5)

    # Train & evaluate
    artifacts = train(args=args, df=df, trial=trial)

    # Set additional attributes
    overall_performance = artifacts["performance"]["overall"]
    logger.info(json.dumps(overall_performance, indent=2))
    trial.set_user_attr("precision", overall_performance["precision"])
    trial.set_user_attr("recall", overall_performance["recall"])
    trial.set_user_attr("f1", overall_performance["f1"])

    return overall_performance["f1"]


def train_step(
    dataset: Dataset,
    batch_size: int,
    model: nn.Module,
    num_classes: int,
    loss_fn: torch.nn.modules.loss._WeightedLoss,
    optimizer: torch.optim.Optimizer,
) -> float:  # pragma: no cover, tested via train workload
    """Train step for training loop of worker

    Args:
        dataset (Dataset): dataset to iterate batches from.
        batch_size (int): size of each batch.
        model (nn.Module): model to train.
        num_classes (int): number of classes.
        loss_fn (torch.nn.loss._WeightedLoss): loss function to use between labels and predictions.
        optimizer (torch.optimizer.Optimizer): optimizer to use for updating the model's weights.

    Returns:
        float: cumulative loss for the dataset
    """
    model.train()
    loss = 0.0
    ds_generator = dataset.iter_torch_batches(batch_size=batch_size, collate_fn=utils.collate_fn)
    for i, batch in enumerate(ds_generator):
        optimizer.zero_grad()  # reset gradiants
        z = model(batch)  # forward pass
        targets = F.one_hot(
            batch["targets"], num_classes=num_classes
        ).float()  # one-hot (for loss_fn)
        J = loss_fn(z, targets)  # define loss
        J.backward()  # backward pass
        loss += (J.detach().item() - loss) / (i + 1)
    return loss


def eval_step(
    dataset: Dataset,
    batch_size: int,
    model: nn.Module,
    num_classes: int,
    loss_fn: torch.nn.modules.loss._WeightedLoss,
) -> tuple[float, np.array, np.array]:
    """Eval step for evaluating a looper of worker

    Args:
        dataset (Dataset): _dataset to iterate batches from.
        batch_size (int): size of each batch.
        model (nn.Module): model to train.
        num_classes (int): number of classes.
        loss_fn (torch.nn.loss._WeightedLoss): loss function to use between labels and predictions.

    Returns:
        Tuple[float, np.array, np.array]: cumulative loss, ground truths and predictions.
    """
    model.eval()
    loss = 0.0
    y_trues, y_preds = [], []
    ds_generator = dataset.iter_torch_batches(batch_size=batch_size, collate_fn=utils.collate_fn)
    with torch.inference_mode():
        for i, batch in enumerate(ds_generator):
            z = model(batch)
            targets = F.one_hot(
                batch["targets"], num_classes=num_classes
            ).float()  # one-hot (for loss_fn)
            J = loss_fn(z, targets).item()
            loss += (J - loss) / (i + 1)
            y_trues.extend(batch["targets"].cpu().numpy())
            y_preds.extend(torch.argmax(z, dim=1).cpu().numpy())
    return loss, np.vstack(y_trues), np.vstack(y_preds)


def train_loop_per_worker(config: dict) -> None:  # pragma: no cover, tested via train workload
    """Training loop that each worker will execute

    Args:
        config (dict): arguments to use for training
    """

    # Hyperparameters
    threshold = config["threshold"]
    learning_rate = config["learning_rate"]
    learning_rate_factor = config["learning_rate_factor"]
    learning_rate_patience = config["learning_rate_patience"]
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    num_classes = config["num_classes"]

    # Get datasets
    utils.set_seeds()
    train_data = session.get_dataset_shard("train")
    val_data = session.get_dataset_shard("val")

    # Model
    llm = BertModel.from_pretrained("allenai/scibert_scivocab_uncased", return_dict=False)
    model = models.FinetunedLLM(
        llm=llm, threshold=threshold, embedding_dim=llm.config.hidden_size, num_classes=num_classes
    )
    model = ray_train.torch.prepare_model(model)

    # Training components
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=learning_rate_factor,
        patience=learning_rate_patience,
    )

    # Training
    batch_size_per_worker = batch_size // session.get_world_size()
    for epoch in range(num_epochs):
        # Step
        logger.info(f"Training worker {session.get_node_rank()} epoch {epoch} started")
        train_loss = train_step(
            train_data, batch_size_per_worker, model, num_classes, loss_fn, optimizer
        )
        val_loss, _, _ = eval_step(val_data, batch_size_per_worker, model, num_classes, loss_fn)
        scheduler.step(val_loss)

        # Checkpoint
        metrics = dict(
            epoch=epoch,
            learning_rate=optimizer.param_groups[0]["learning_rate"],
            train_loss=train_loss,
            val_loss=val_loss,
        )
        checkpoint = TorchCheckpoint.from_model(model=model)
        logger.info(
            f"Training for worker {session.get_local_rank()} epoch {epoch} completed -- Checkpoint {checkpoint} -- Metrics: \n {json.dumps(metrics, indent=2)}"
        )
        session.report(metrics, checkpoint=checkpoint)


@app.command()
def train_model(
    experiment_name: Annotated[
        str, typer.Option(help="name of the experiment for this training workload")
    ] = None,
    dataset_loc: Annotated[str, typer.Option(help="location of the dataset")] = None,
    train_config_loc: Annotated[
        str, typer.Option(help="location of arguments to use for training")
    ] = None,
    num_workers: Annotated[int, typer.Option(help="number of workers to use for training")] = 1,
    cpu_per_worker: Annotated[int, typer.Option(help="number of CPUs to use per worker")] = 1,
    gpu_per_worker: Annotated[int, typer.Option(help="number of GPUs to use per worker")] = 0,
    num_samples: Annotated[int, typer.Option(help="number of samples to use from dataset")] = None,
    num_epochs: Annotated[int, typer.Option(help="number of epochs to train for")] = 1,
    batch_size: Annotated[int, typer.Option(help="number of samples per batch")] = 256,
    results_fp: Annotated[str, typer.Option(help="filepath to save results to")] = None,
) -> ray.air.result.Result:
    """Main function for training a model as a distributed workload

    Args:
        experiment_name (str): name of the experiment for this training workload.
        dataset_loc (str): location of the dataset.
        train_config_loc (str): location of arguments to use for training.
        num_workers (int, optional): number of workers to use for training. Defaults to 1.
        cpu_per_worker (int, optional): number of CPUs to use per worker. Defaults to 1.
        gpu_per_worker (int, optional): number of GPUs to use per worker. Defaults to 0.
        num_samples (int, optional): number of samples to use from dataset.
            If this is passed in, it will override the config. Defaults to None.
        num_epochs (int, optional): number of epochs to train for.
            If this is passed in, it will override the config. Defaults to None.
        batch_size (int, optional): number of samples per batch.
            If this is passed in, it will override the config. Defaults to None.
        results_fp (str, optional): filepath to save results to. Defaults to None.

    Returns:
        ray.air.result.Result: training results
    """

    # Set up args config
    config_loc = train_config_loc if train_config_loc else ARGS_URI
    args_config = Namespace(**utils.load_dict(filepath=config_loc))
    args_config["num_samples"] = num_samples
    args_config["num_epochs"] = num_epochs
    args_config["batch_size"] = batch_size
    logger.info(
        f"Training for {experiment_name} with data from {dataset_loc}. Results written to {results_fp}."
    )
    logger.info(f"Starting training with args --> \n {json.dumps(args_config, indent=2)}")

    # Set up scaling config
    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=bool(gpu_per_worker),
        resources_per_worker={"CPU": cpu_per_worker, "GPU": gpu_per_worker},
    )

    # Checkpoint config
    checkpoint_config = CheckpointConfig(
        num_to_keep=1, checkpoint_score_attribute="val_loss", checkpoint_score_order="min"
    )

    # MLflow callback
    mlflow_callback = MLflowLoggerCallback(
        tracking_uri=MLFLOW_TRACKING_URI, experiment_name=experiment_name, save_artifact=True
    )

    # Run config
    run_config = RunConfig(callbacks=[mlflow_callback], checkpoint_config=checkpoint_config)

    # Dataset
    dataset = data.load_data(dataset_loc=dataset_loc, num_samples=args_config["num_samples"])
    train_data, val_data = data.stratify_split(data=dataset, stratify="tag", test_size=0.2)
    tags = train_data.unique(column="tag")
    args_config["num_classes"] = len(tags)

    # Dataset config
    dataset_config = {
        "train": DatasetConfig(fit=False, transform=False, randomize_block_order=False),
        "val": DatasetConfig(fit=False, transform=False, randomize_block_order=False),
    }

    # Preprocess
    preprocessor = data.CustomPreprocessor()
    train_data = preprocessor.fit_transform(train_data)
    val_data = preprocessor.fit_transform(val_data)
    train_data = train_data.materialize()
    val_data = val_data.materialize()

    # Trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=args_config,
        scaling_config=scaling_config,
        run_config=run_config,
        datasets={"train": train_data, "val": val_data},
        dataset_config=dataset_config,
        preprocessor=preprocessor,
    )

    # Train
    results = trainer.fit()
    results_data = {
        "timestamp": datetime.now().strftime("%B %d, %Y %I:%M:%S %p"),
        "run_id": utils.get_run_id(
            experiment_name=experiment_name, trial_id=results.metrics["trial_id"]
        ),
        "params": results.config["train_loop_config"],
        "metrics": utils.dict_to_list(
            results.metrics_dataframe.to_dict(), keys=["epoch", "train_loss", "val_loss"]
        ),
    }
    logger.info(f"Results data --> \n {json.dumps(results_data, indent=2)}")
    if results_fp:  # pragma: no cover, saving results
        utils.save_dict(results_data, results_fp)
    return results


if __name__ == "__main__":  # pragma: no cover, application
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    app()
