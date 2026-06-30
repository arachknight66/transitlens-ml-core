"""Modular baseline convolutional network for transit detection."""

import torch
from torch import nn

from transitlens_ml_core.config import ModelConfig
from transitlens_ml_core.models.classifier import BinaryTransitClassifier
from transitlens_ml_core.models.layers import ConvBlock


class CNNFeatureExtractor(nn.Module):
    """Two-block one-dimensional CNN feature extractor."""

    def __init__(
        self,
        input_channels: int,
        convolution_channels: tuple[int, int],
        kernel_size: int,
        pool_size: int,
    ) -> None:
        """Initialize convolutional feature extraction.

        Args:
            input_channels: Number of input signal channels.
            convolution_channels: Output width of each convolution block.
            kernel_size: Odd convolution kernel width.
            pool_size: First-block max-pooling width and stride.

        """
        super().__init__()
        first_channels, second_channels = convolution_channels
        self.first_block = ConvBlock(
            input_channels,
            first_channels,
            kernel_size,
            pool_size,
        )
        self.second_block = ConvBlock(
            first_channels,
            second_channels,
            kernel_size,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Extract a sequence of learned features.

        Args:
            inputs: Tensor shaped ``[batch, channels, samples]``.

        Returns:
            Learned feature tensor for a downstream classifier or attention block.

        """
        return self.second_block(self.first_block(inputs))


class BaselineCNN(nn.Module):
    """Lightweight baseline model producing binary transit probabilities."""

    def __init__(
        self,
        input_channels: int,
        convolution_channels: tuple[int, int],
        kernel_size: int,
        pool_size: int,
    ) -> None:
        """Initialize the composed baseline model.

        Args:
            input_channels: Number of input signal channels.
            convolution_channels: Output width of each convolution block.
            kernel_size: Odd convolution kernel width.
            pool_size: First-block max-pooling width and stride.

        Raises:
            ValueError: If ``pool_size`` is not positive.

        """
        super().__init__()
        if pool_size <= 0:
            raise ValueError("pool_size must be positive")
        self.input_channels = input_channels
        self.minimum_samples = pool_size
        self.feature_extractor = CNNFeatureExtractor(
            input_channels,
            convolution_channels,
            kernel_size,
            pool_size,
        )
        self.classifier = BinaryTransitClassifier(convolution_channels[-1])

    @classmethod
    def from_config(cls, config: ModelConfig) -> "BaselineCNN":
        """Construct the baseline model from validated configuration.

        Args:
            config: Baseline model configuration.

        Returns:
            A configured baseline CNN.

        """
        return cls(
            input_channels=config.input_channels,
            convolution_channels=config.convolution_channels,
            kernel_size=config.kernel_size,
            pool_size=config.pool_size,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict transit probabilities for a batch of light curves.

        Args:
            inputs: Float tensor shaped ``[batch, channels, samples]``.

        Returns:
            Transit probabilities shaped ``[batch, 1]``.

        Raises:
            ValueError: If the input shape, channel count, or length is invalid.

        """
        if inputs.ndim != 3:
            raise ValueError("model input must have shape [batch, channels, samples]")
        if inputs.shape[1] != self.input_channels:
            raise ValueError(
                f"model expected {self.input_channels} input channels, "
                f"received {inputs.shape[1]}"
            )
        if inputs.shape[2] < self.minimum_samples:
            raise ValueError(
                f"model input must contain at least {self.minimum_samples} samples"
            )
        features = self.feature_extractor(inputs)
        return self.classifier(features)
