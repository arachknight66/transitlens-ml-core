"""Optimizer construction for model training."""

from collections.abc import Iterable

from torch import Tensor
from torch.optim import Adam

from transitlens_ml_core.config import TrainingConfig


def create_optimizer(parameters: Iterable[Tensor], config: TrainingConfig) -> Adam:
    """Create the configured Adam optimizer.

    Args:
        parameters: Trainable model parameters.
        config: Validated training configuration.

    Returns:
        Configured Adam optimizer.

    """
    return Adam(
        parameters,
        lr=config.learning_rate,
        betas=config.optimizer_betas,
        eps=config.optimizer_epsilon,
        weight_decay=config.weight_decay,
    )
