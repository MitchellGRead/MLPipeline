from pipeline_utils import wb_utils
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.air.integrations.wandb import setup_wandb
from ray.data import Preprocessor

import wandb


class WandbManager:
    def __init__(
        self,
        config: dict[str, any],
        project: str,
        group: str,
        api_key: str,
        job_type: str,
        use_artifact: str | None,
        preprocessor: Preprocessor,
    ) -> None:
        self._run = setup_wandb(
            config=config, api_key=api_key, group=group, job_type=job_type, project=project
        )
        self._preprocessor_used = preprocessor
        if use_artifact:
            self._run.use_artifact(use_artifact)

    def report(self, metrics: dict[str, any], checkpoint: Checkpoint) -> None:
        session.report(metrics, checkpoint=checkpoint)
        checkpoint.set_preprocessor(self._preprocessor_used)

        self.log(metrics)
        self.save_checkpoint(checkpoint)

    def log(self, metrics: dict[str, any]) -> None:
        self._run.log(metrics)

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        artifact = wandb.Artifact(
            name=wb_utils.create_checkpoint_name(session.get_trial_id()), type="model"
        )
        artifact.add_dir(checkpoint.to_directory())
        self._run.log_artifact(artifact)

    def cleanup(self) -> None:
        self._run.finish()
