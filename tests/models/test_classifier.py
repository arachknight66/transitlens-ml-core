"""Tests for the binary transit classifier."""

import pytest
import torch
from torch import nn

from transitlens_ml_core.models import BinaryTransitClassifier


def test_classifier_pools_variable_length_features_into_probabilities() -> None:
    classifier = BinaryTransitClassifier(input_channels=8)

    short = classifier(torch.ones(2, 8, 5))
    long = classifier(torch.ones(2, 8, 17))

    assert short.shape == long.shape == (2, 1)
    assert torch.all((short >= 0.0) & (short <= 1.0))
    assert isinstance(classifier.global_pool, nn.AdaptiveAvgPool1d)
    assert isinstance(classifier.output, nn.Linear)
    assert isinstance(classifier.probability, nn.Sigmoid)


def test_classifier_rejects_invalid_channel_count() -> None:
    with pytest.raises(ValueError, match="positive"):
        BinaryTransitClassifier(input_channels=0)
