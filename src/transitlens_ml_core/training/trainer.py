"""Deterministic model training, validation, and checkpoint lifecycle."""

import json
import os
import pickle
import random
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from transitlens_ml_core.config import TrainingConfig
from transitlens_ml_core.training.losses import create_binary_loss
from transitlens_ml_core.training.optimizer import create_optimizer
from transitlens_ml_core.training.scheduler import EarlyStopping, create_scheduler

_CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class EpochRecord:
    """Loss and learning-rate history for one completed epoch."""

    epoch: int
    training_loss: float
    validation_loss: float
    learning_rate: float


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Stable summary returned after fitting completes."""

    history: tuple[EpochRecord, ...]
    best_epoch: int
    best_validation_loss: float
    stopped_early: bool
    best_checkpoint: Path
    latest_checkpoint: Path
    history_path: Path


def set_deterministic_seed(seed: int) -> None:
    """Seed supported random generators and request deterministic PyTorch ops.

    Args:
        seed: Non-negative reproducibility seed.

    Raises:
        ValueError: If ``seed`` is negative.

    """
    if seed < 0:
        raise ValueError("deterministic seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


class Trainer:
    """Compose deterministic binary model training dependencies."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
        seed: int,
    ) -> None:
        """Initialize model, optimization, scheduling, and artifact state.

        Args:
            model: Probability-producing binary classifier.
            config: Validated training configuration.
            device: Device used for training and validation.
            seed: Reproducibility seed for stochastic operations.

        """
        set_deterministic_seed(seed)
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.loss_function = create_binary_loss()
        self.optimizer = create_optimizer(self.model.parameters(), config)
        self.scheduler = create_scheduler(self.optimizer, config)
        self.early_stopping = EarlyStopping(
            config.early_stopping_patience,
            config.early_stopping_min_delta,
        )
        self.best_checkpoint = (
            config.checkpoint_directory / config.best_checkpoint_filename
        )
        self.latest_checkpoint = (
            config.checkpoint_directory / config.latest_checkpoint_filename
        )
        self.history_path = config.experiment_directory / config.history_filename

    def fit(
        self,
        training_loader: DataLoader[Any],
        validation_loader: DataLoader[Any],
        resume_from: str | Path | None = None,
    ) -> TrainingResult:
        """Train the model and persist recoverable state after every epoch.

        Args:
            training_loader: Batches used for gradient updates.
            validation_loader: Batches used only for validation loss.
            resume_from: Optional latest checkpoint to resume.

        Returns:
            Completed training history and artifact locations.

        Raises:
            ValueError: If a loader is empty or a checkpoint cannot be resumed.

        """
        if len(training_loader) == 0 or len(validation_loader) == 0:
            raise ValueError("training and validation loaders must not be empty")

        history: list[EpochRecord] = []
        start_epoch = 1
        if resume_from is not None:
            start_epoch, history = self._resume(Path(resume_from))
        if start_epoch > self.config.epochs:
            raise ValueError("checkpoint already reaches configured training epochs")

        stopped_early = False
        for epoch in range(start_epoch, self.config.epochs + 1):
            learning_rate = self.optimizer.param_groups[0]["lr"]
            training_loss = self._run_training_epoch(training_loader)
            validation_loss = self._run_validation_epoch(validation_loader)
            self.scheduler.step(validation_loss)
            improved, should_stop = self.early_stopping.observe(validation_loss)
            record = EpochRecord(
                epoch=epoch,
                training_loss=training_loss,
                validation_loss=validation_loss,
                learning_rate=learning_rate,
            )
            history.append(record)
            if improved:
                self._save_checkpoint(self.best_checkpoint, epoch, history)
            self._save_checkpoint(self.latest_checkpoint, epoch, history)
            self._write_history(history)
            if should_stop:
                stopped_early = True
                break

        best_record = next(
            item
            for item in history
            if item.validation_loss == self.early_stopping.best_loss
        )
        return TrainingResult(
            history=tuple(history),
            best_epoch=best_record.epoch,
            best_validation_loss=self.early_stopping.best_loss,
            stopped_early=stopped_early,
            best_checkpoint=self.best_checkpoint,
            latest_checkpoint=self.latest_checkpoint,
            history_path=self.history_path,
        )

    def _run_training_epoch(self, loader: DataLoader[Any]) -> float:
        """Run one gradient-update epoch and return sample-weighted loss."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        for features, targets in loader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            probabilities = self.model(features)
            targets = targets.reshape_as(probabilities)
            loss = self.loss_function(probabilities, targets)
            loss.backward()
            self.optimizer.step()
            batch_size = features.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size
        if total_samples == 0:
            raise ValueError("training loader produced no samples")
        return total_loss / total_samples

    def _run_validation_epoch(self, loader: DataLoader[Any]) -> float:
        """Run one inference-only validation epoch and return weighted loss."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.inference_mode():
            for features, targets in loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                probabilities = self.model(features)
                targets = targets.reshape_as(probabilities)
                loss = self.loss_function(probabilities, targets)
                batch_size = features.shape[0]
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        if total_samples == 0:
            raise ValueError("validation loader produced no samples")
        return total_loss / total_samples

    def _checkpoint_payload(
        self, epoch: int, history: list[EpochRecord]
    ) -> dict[str, Any]:
        """Build complete state required to resume training."""
        cuda_rng_state = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        )
        return {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_validation_loss": self.early_stopping.best_loss,
            "bad_epochs": self.early_stopping.bad_epochs,
            "history": [asdict(record) for record in history],
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": cuda_rng_state,
        }

    def _save_checkpoint(
        self, path: Path, epoch: int, history: list[EpochRecord]
    ) -> None:
        """Atomically save a complete training checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}-",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        try:
            torch.save(self._checkpoint_payload(epoch, history), temporary_path)
            os.replace(temporary_path, path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    def _write_history(self, history: list[EpochRecord]) -> None:
        """Atomically persist human-readable training history."""
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(
            [asdict(record) for record in history], indent=2, sort_keys=True
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.history_path.parent,
            prefix=f".{self.history_path.name}-",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(content)
            temporary_file.write("\n")
        try:
            os.replace(temporary_path, self.history_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    def _resume(self, path: Path) -> tuple[int, list[EpochRecord]]:
        """Restore model and training state from a trusted local checkpoint."""
        if not path.is_file():
            raise ValueError(f"resume checkpoint does not exist: {path}")
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        except (EOFError, OSError, pickle.UnpicklingError, RuntimeError) as error:
            raise ValueError("unable to read resume checkpoint") from error
        if not isinstance(checkpoint, Mapping):
            raise ValueError("resume checkpoint root must be a mapping")
        required = {
            "schema_version",
            "epoch",
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "best_validation_loss",
            "bad_epochs",
            "history",
            "torch_rng_state",
            "cuda_rng_state",
        }
        if missing := sorted(required.difference(checkpoint)):
            raise ValueError(f"resume checkpoint is missing fields: {missing}")
        if checkpoint["schema_version"] != _CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported training checkpoint schema version")

        try:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.early_stopping.best_loss = float(checkpoint["best_validation_loss"])
            self.early_stopping.bad_epochs = int(checkpoint["bad_epochs"])
            history = [EpochRecord(**record) for record in checkpoint["history"]]
            epoch = int(checkpoint["epoch"])
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
            if torch.cuda.is_available() and checkpoint["cuda_rng_state"]:
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            raise ValueError("resume checkpoint contains invalid state") from error
        if epoch < 1 or not history or history[-1].epoch != epoch:
            raise ValueError("resume checkpoint epoch and history are inconsistent")
        return epoch + 1, history
