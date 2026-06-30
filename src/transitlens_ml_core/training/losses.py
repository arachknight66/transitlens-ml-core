"""Loss construction for model training."""

from torch import nn


def create_binary_loss() -> nn.BCELoss:
    """Create the loss matching the baseline model's sigmoid probabilities.

    Returns:
        Binary cross-entropy loss with sample-mean reduction.

    """
    return nn.BCELoss(reduction="mean")
