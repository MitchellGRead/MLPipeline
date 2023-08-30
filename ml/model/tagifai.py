import numpy as np
import ray.train as train
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.air import session
from ray.data import Dataset
from ray.train.torch import TorchCheckpoint, get_device
from transformers import BertModel

from config.config import logger
from ml.api.model_interface import ModelInterface
from ml.ml_utils import numpy_utils


class TagifaiModel(ModelInterface):
    """Tagifai LLM that identifies a set of tags within text

    Args:
        ModelInterface (ModelInterface): Model Interface to adhere to for commonality of ML Pipeline
    """

    def __init__(self) -> None:
        super().__init__()
        logger.info("Tagifai model created")

    def __collate_fn(
        self, batch: dict[str, np.ndarray]
    ) -> dict[str, torch.Tensor]:  # pragma: no cover, air internal
        """Convert a batch of numpy arrays to tensors (with appropriate padding).

        Args:
            batch (Dict[str, np.ndarray]): input batch as a dictionary of numpy arrays.

        Returns:
            Dict[str, torch.Tensor]: output batch as a dictionary of tensors.
        """
        batch["ids"] = numpy_utils.pad_array(batch["ids"])
        batch["masks"] = numpy_utils.pad_array(batch["masks"])
        dtypes = {"ids": torch.int32, "masks": torch.int32, "targets": torch.int64}
        tensor_batch = {}
        for key, array in batch.items():
            tensor_batch[key] = torch.as_tensor(array, dtype=dtypes[key], device=get_device())
        return tensor_batch

    def __train_step(
        self,
        dataset: Dataset,
        batch_size: int,
        model: nn.Module,
        num_classes: int,
        loss_fn: torch.nn.modules.loss._WeightedLoss,
        optimizer: torch.optim.Optimizer,
    ) -> float:  # pragma: no cover, tested via train workload
        """Train step.

        Args:
            ds (Dataset): dataset to iterate batches from.
            batch_size (int): size of each batch.
            model (nn.Module): model to train.
            num_classes (int): number of classes.
            loss_fn (torch.nn.loss._WeightedLoss): loss function to use between labels and predictions.
            optimizer (torch.optimizer.Optimizer): optimizer to use for updating the model's weights.

        Returns:
            float: cumulative loss for the dataset.
        """
        model.train()
        loss = 0.0
        ds_generator = dataset.iter_torch_batches(
            batch_size=batch_size, collate_fn=self.__collate_fn
        )
        for i, batch in enumerate(ds_generator):
            optimizer.zero_grad()  # reset gradients
            z = model(batch)  # forward pass
            targets = F.one_hot(
                batch["targets"], num_classes=num_classes
            ).float()  # one-hot (for loss_fn)
            J = loss_fn(z, targets)  # define loss
            J.backward()  # backward pass
            optimizer.step()  # update weights
            loss += (J.detach().item() - loss) / (i + 1)  # cumulative loss
        return loss

    def __eval_step(
        self,
        dataset: Dataset,
        batch_size: int,
        model: nn.Module,
        num_classes: int,
        loss_fn: torch.nn.modules.loss._WeightedLoss,
    ) -> tuple[float, np.array, np.array]:  # pragma: no cover, tested via train workload
        """Eval step.

        Args:
            ds (Dataset): dataset to iterate batches from.
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
        ds_generator = dataset.iter_torch_batches(
            batch_size=batch_size, collate_fn=self.__collate_fn
        )
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

    def train_loop_per_worker(
        self, config: dict
    ) -> None:  # pragma: no cover, tested via train workload
        """Training loop that each worker will execute.

        Args:
            config (dict): arguments to use for training.
        """
        # Hyperparameters
        dropout_p = config["dropout_p"]
        lr = config["lr"]
        lr_factor = config["lr_factor"]
        lr_patience = config["lr_patience"]
        batch_size = config["batch_size"]
        num_epochs = config["num_epochs"]
        num_classes = config["num_classes"]

        # Get datasets
        train_ds = session.get_dataset_shard("train")
        eval_ds = session.get_dataset_shard("eval")

        # Model
        llm = BertModel.from_pretrained("allenai/scibert_scivocab_uncased", return_dict=False)
        model = FinetunedLLM(
            llm=llm,
            dropout_p=dropout_p,
            embedding_dim=llm.config.hidden_size,
            num_classes=num_classes,
        )
        model = train.torch.prepare_model(model)

        # Training components
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=lr_factor, patience=lr_patience
        )

        # Training
        batch_size_per_worker = batch_size // session.get_world_size()
        for epoch in range(num_epochs):
            # Step
            train_loss = self.__train_step(
                train_ds, batch_size_per_worker, model, num_classes, loss_fn, optimizer
            )
            val_loss, _, _ = self.__eval_step(
                eval_ds, batch_size_per_worker, model, num_classes, loss_fn
            )
            scheduler.step(val_loss)

            # Checkpoint
            metrics = dict(
                epoch=epoch,
                lr=optimizer.param_groups[0]["lr"],
                train_loss=train_loss,
                val_loss=val_loss,
            )
            checkpoint = TorchCheckpoint.from_model(model=model)
            session.report(metrics, checkpoint=checkpoint)


class FinetunedLLM(nn.Module):  # pragma: no cover, torch model
    """Model architecture for a Large Language Model (LLM) that we will fine-tune."""

    def __init__(self, llm, dropout_p, embedding_dim, num_classes):
        super(FinetunedLLM, self).__init__()
        self.llm = llm
        self.dropout = torch.nn.Dropout(dropout_p)
        self.fc1 = torch.nn.Linear(embedding_dim, num_classes)

    def forward(self, batch):
        ids, masks = batch["ids"], batch["masks"]
        _, pool = self.llm(input_ids=ids, attention_mask=masks)
        z = self.dropout(pool)
        z = self.fc1(z)
        return z
