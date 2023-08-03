import json
import tempfile
import warnings
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict

import joblib
import mlflow
import optuna
import pandas as pd
from numpyencoder import NumpyEncoder
from optuna.integration.mlflow import MLflowCallback

from config import config
from config.config import logger
from tagifai import data_old, predict, train_old, utils_old

warnings.filterwarnings("ignore")


def elt_data() -> None:
    """Extract, load and transform our data assets."""

    # Extract + Load
    logger.info("Start extracting and loading data")
    projects = pd.read_csv(config.PROJECTS_URL)
    tags = pd.read_csv(config.TAGS_URL)
    projects.to_csv(Path(config.DATA_DIR, "projects.csv"), index=False)
    tags.to_csv(Path(config.DATA_DIR, "tags.csv"), index=False)

    # Transform
    logger.info("Start transforming data")
    df = pd.merge(projects, tags, on="id")
    df = df[df.tag.notnull()]  # drop rows w/ no tag
    df.to_csv(Path(config.DATA_DIR, "labeled_projects.csv"), index=False)

    logger.info(f"✅ Saved data to {config.DATA_DIR}")


def train_model(experiment_name: str, run_name: str, args_fp: str = "config/args.json") -> None:
    """Train a model given arguments

    Args:
        experiment_name (str): Name of trained experiment
        run_name (str): Name of experiment run
        args_fp (str, optional): Filepath to args json configuration. Defaults to "config/args.json".
    """
    # Load labeled data
    logger.info("Reading in training data to train model")
    df = pd.read_csv(Path(config.DATA_DIR, "labeled_projects.csv"))

    # Train
    args = Namespace(**utils_old.load_dict(filepath=args_fp))
    logger.info(f"Training model with args={args}")

    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        logger.info(f"Run ID: {run_id}")
        artifacts = train_old.train(df=df, args=args)
        performance = artifacts["performance"]
        logger.info(json.dumps(performance, indent=2))

        # Log metrics and parameters
        performance = artifacts["performance"]
        mlflow.log_metrics({"precision": performance["overall"]["precision"]})
        mlflow.log_metrics({"recall": performance["overall"]["recall"]})
        mlflow.log_metrics({"f1": performance["overall"]["f1"]})
        mlflow.log_params(vars(artifacts["args"]))

        # Log artifacts
        with tempfile.TemporaryDirectory() as dp:
            utils_old.save_dict(vars(artifacts["args"]), Path(dp, "args.json"), cls=NumpyEncoder)
            artifacts["label_encoder"].save(Path(dp, "label_encoder.json"))
            joblib.dump(artifacts["vectorizer"], Path(dp, "vectorizer.pkl"))
            joblib.dump(artifacts["model"], Path(dp, "model.pkl"))
            utils_old.save_dict(performance, Path(dp, "performance.json"))
            mlflow.log_artifacts(dp)

    # Save to config
    open(Path(config.CONFIG_DIR, "run_id.txt"), "w").write(run_id)
    utils_old.save_dict(performance, Path(config.CONFIG_DIR, "performance.json"))


def optimize(
    args_fp: str = "config/args.json", study_name: str = "optimization", num_trials: int = 20
) -> None:
    """Optimize hyperparameters arguments

    Args:
        args_fp (str, optional): Filepath to args json config. Defaults to "config/args.json".
        study_name (str, optional): Name of optimization study. Defaults to "optimization".
        num_trials (int, optional): Number of optimization runs to perform. Defaults to 20.
    """
    # Load labeled data
    logger.info("Reading in training data to optimize model")
    df = pd.read_csv(Path(config.DATA_DIR, "labeled_projects.csv"))

    # Optimize
    args = Namespace(**utils_old.load_dict(filepath=args_fp))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(study_name=study_name, direction="maximize", pruner=pruner)
    mlflow_callback = MLflowCallback(tracking_uri=mlflow.get_tracking_uri(), metric_name="f1")
    study.optimize(
        lambda trial: train_old.objective(args, df, trial),
        n_trials=num_trials,
        callbacks=[mlflow_callback],
    )

    # Best trial
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values(["user_attrs_f1"], ascending=False)
    utils_old.save_dict({**args.__dict__, **study.best_trial.params}, args_fp, cls=NumpyEncoder)
    logger.info(f"Best value (f1): {study.best_trial.value}")
    logger.info(f"Best hyperparameters: {json.dumps(study.best_trial.params, indent=2)}")


def predict_tag(text: str, run_id=None) -> Dict[str, str]:
    """Predict tag for text

    Args:
        text (str): Text to predict a tag for
        run_id (str, optional): Model run id to use for prediction. Defaults to None. Grabs run id from config run_id.txt if None.

    Returns:
        Dict[str, str]: Json object of input text and the predicted tag for it
    """
    if not run_id:
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()

    logger.info(f"Predicting tag with run_id={run_id}")
    artifacts = load_artifacts(run_id=run_id)
    prediction = predict.predict(texts=[text], artifacts=artifacts)
    logger.info(json.dumps(prediction, indent=2))
    return prediction


def load_artifacts(run_id: str) -> Dict[str, Any]:
    """Load artifacts for a given run_id

    Args:
        run_id (str): Model run id to loading artifacts for

    Returns:
        Dict[str, Any]: Json object containing primary model artifacts
    """
    # Locate specifics artifacts directory
    experiment_id = mlflow.get_run(run_id=run_id).info.experiment_id
    artifacts_dir = Path(config.MODEL_REGISTRY, experiment_id, run_id, "artifacts")

    # Load objects from run
    args = Namespace(**utils_old.load_dict(filepath=Path(artifacts_dir, "args.json")))
    vectorizer = joblib.load(Path(artifacts_dir, "vectorizer.pkl"))
    label_encoder = data_old.LabelEncoder.load(fp=Path(artifacts_dir, "label_encoder.json"))
    model = joblib.load(Path(artifacts_dir, "model.pkl"))
    performance = utils_old.load_dict(filepath=Path(artifacts_dir, "performance.json"))

    return {
        "args": args,
        "label_encoder": label_encoder,
        "vectorizer": vectorizer,
        "model": model,
        "performance": performance,
    }
