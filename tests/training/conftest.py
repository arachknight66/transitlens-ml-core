"""Shared training test fixtures."""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from transitlens_ml_core.config import TrainingConfig
from transitlens_ml_core.training import set_deterministic_seed


def make_training_config(
    root: Path,
    *,
    epochs: int = 4,
    learning_rate: float = 0.05,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 0.0,
    scheduler_patience: int = 2,
) -> TrainingConfig:
    """Create an isolated training configuration."""
    return TrainingConfig(
        batch_size=4,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=0.0,
        optimizer_betas=(0.9, 0.999),
        optimizer_epsilon=1e-8,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        scheduler_patience=scheduler_patience,
        scheduler_factor=0.5,
        scheduler_min_lr=1e-6,
        checkpoint_directory=root / "weights",
        best_checkpoint_filename="best.pt",
        latest_checkpoint_filename="latest.pt",
        experiment_directory=root / "experiments",
        history_filename="history.json",
    )


def make_model(seed: int = 11) -> nn.Module:
    """Create a deterministic lightweight sigmoid classifier."""
    set_deterministic_seed(seed)
    return nn.Sequential(nn.Flatten(), nn.Linear(4, 1), nn.Sigmoid())


def make_loaders() -> tuple[DataLoader, DataLoader]:
    """Create fixed separable training and validation batches."""
    negative = -torch.ones(8, 1, 4)
    positive = torch.ones(8, 1, 4)
    features = torch.cat((negative, positive))
    targets = torch.cat((torch.zeros(8), torch.ones(8)))
    dataset = TensorDataset(features, targets)
    return (
        DataLoader(dataset, batch_size=4, shuffle=False),
        DataLoader(dataset, batch_size=8, shuffle=False),
    )
