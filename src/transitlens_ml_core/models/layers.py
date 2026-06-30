"""Reusable neural-network layers for light-curve models."""

import torch
from torch import nn


class ConvBlock(nn.Module):
    """One-dimensional convolution, normalization, activation, and pooling."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        pool_size: int | None = None,
    ) -> None:
        """Initialize a length-preserving convolution block.

        Args:
            input_channels: Number of incoming feature channels.
            output_channels: Number of learned convolution channels.
            kernel_size: Odd convolution kernel width.
            pool_size: Optional max-pooling width and stride.

        Raises:
            ValueError: If channel counts, kernel size, or pool size are invalid.

        """
        super().__init__()
        if input_channels <= 0 or output_channels <= 0:
            raise ValueError("convolution channel counts must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("convolution kernel_size must be a positive odd integer")
        if pool_size is not None and pool_size <= 0:
            raise ValueError("pool_size must be positive when provided")

        self.convolution = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.normalization = nn.BatchNorm1d(output_channels)
        self.activation = nn.ReLU()
        self.pooling = (
            nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)
            if pool_size is not None
            else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Transform a channel-first sequence tensor.

        Args:
            inputs: Tensor shaped ``[batch, channels, samples]``.

        Returns:
            The transformed feature sequence.

        """
        return self.pooling(
            self.activation(self.normalization(self.convolution(inputs)))
        )
