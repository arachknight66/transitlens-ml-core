"""Binary classification heads for transit detection."""

import torch
from torch import nn


class BinaryTransitClassifier(nn.Module):
    """Global-average-pooling classifier returning transit probabilities."""

    def __init__(self, input_channels: int) -> None:
        """Initialize the binary classifier.

        Args:
            input_channels: Number of incoming feature channels.

        Raises:
            ValueError: If ``input_channels`` is not positive.

        """
        super().__init__()
        if input_channels <= 0:
            raise ValueError("classifier input_channels must be positive")
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(start_dim=1)
        self.output = nn.Linear(input_channels, 1)
        self.probability = nn.Sigmoid()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Convert feature sequences into transit probabilities.

        Args:
            features: Tensor shaped ``[batch, channels, samples]``.

        Returns:
            Probability tensor shaped ``[batch, 1]``.

        """
        pooled = self.flatten(self.global_pool(features))
        return self.probability(self.output(pooled))
