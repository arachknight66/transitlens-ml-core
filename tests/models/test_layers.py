"""Tests for reusable convolutional model layers."""

import pytest
import torch
from torch import nn

from transitlens_ml_core.models import ConvBlock


def test_conv_block_applies_expected_shape_and_pooling() -> None:
    block = ConvBlock(1, 4, kernel_size=5, pool_size=2)

    output = block(torch.ones(3, 1, 16))

    assert output.shape == (3, 4, 8)
    assert isinstance(block.convolution, nn.Conv1d)
    assert isinstance(block.normalization, nn.BatchNorm1d)
    assert isinstance(block.activation, nn.ReLU)
    assert isinstance(block.pooling, nn.MaxPool1d)
    assert block.convolution.bias is None


def test_conv_block_without_pooling_preserves_length() -> None:
    block = ConvBlock(4, 8, kernel_size=3)

    output = block(torch.ones(2, 4, 11))

    assert output.shape == (2, 8, 11)
    assert isinstance(block.pooling, nn.Identity)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((0, 4, 3, None), "channel counts"),
        ((1, 0, 3, None), "channel counts"),
        ((1, 4, 2, None), "positive odd"),
        ((1, 4, 3, 0), "pool_size"),
    ],
)
def test_conv_block_rejects_invalid_configuration(
    arguments: tuple[int, int, int, int | None], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ConvBlock(*arguments)
