"""Learning-rate scheduling and early stopping."""

from dataclasses import dataclass

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from transitlens_ml_core.config import TrainingConfig


@dataclass(slots=True)
class EarlyStopping:
    """Track validation improvements and signal when training should stop."""

    patience: int
    min_delta: float
    best_loss: float = float("inf")
    bad_epochs: int = 0

    def __post_init__(self) -> None:
        """Validate early-stopping parameters."""
        if self.patience <= 0:
            raise ValueError("early-stopping patience must be positive")
        if self.min_delta < 0.0:
            raise ValueError("early-stopping min_delta must be non-negative")

    def observe(self, validation_loss: float) -> tuple[bool, bool]:
        """Record one validation loss.

        Args:
            validation_loss: Mean loss for the completed validation epoch.

        Returns:
            A pair containing whether the loss improved and whether to stop.

        Raises:
            ValueError: If ``validation_loss`` is not finite.

        """
        if validation_loss != validation_loss or validation_loss in (
            float("inf"),
            float("-inf"),
        ):
            raise ValueError("validation loss must be finite")
        improved = validation_loss < self.best_loss - self.min_delta
        if improved:
            self.best_loss = validation_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience


def create_scheduler(optimizer: Optimizer, config: TrainingConfig) -> ReduceLROnPlateau:
    """Create the configured validation-loss plateau scheduler.

    Args:
        optimizer: Optimizer whose learning rate will be adjusted.
        config: Validated training configuration.

    Returns:
        Configured reduce-on-plateau scheduler.

    """
    return ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        threshold=config.early_stopping_min_delta,
        threshold_mode="abs",
        min_lr=config.scheduler_min_lr,
    )
