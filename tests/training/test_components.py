"""Tests for loss, optimizer, scheduling, and deterministic setup."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from transitlens_ml_core.training import (
    EarlyStopping,
    create_binary_loss,
    create_optimizer,
    create_scheduler,
    set_deterministic_seed,
)

from .conftest import make_training_config


def test_binary_loss_matches_probability_model() -> None:
    loss = create_binary_loss()

    value = loss(torch.tensor([[0.25], [0.75]]), torch.tensor([[0.0], [1.0]]))

    assert isinstance(loss, nn.BCELoss)
    assert value.item() == pytest.approx(-np.log(0.75))


def test_optimizer_and_scheduler_use_configuration(tmp_path: Path) -> None:
    config = make_training_config(tmp_path, scheduler_patience=1)
    model = nn.Linear(2, 1)
    optimizer = create_optimizer(model.parameters(), config)
    scheduler = create_scheduler(optimizer, config)

    assert optimizer.param_groups[0]["lr"] == config.learning_rate
    assert optimizer.param_groups[0]["betas"] == config.optimizer_betas
    scheduler.step(1.0)
    scheduler.step(1.0)
    scheduler.step(1.0)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(
        config.learning_rate * config.scheduler_factor
    )


def test_early_stopping_tracks_improvement_and_patience() -> None:
    stopping = EarlyStopping(patience=2, min_delta=0.1)

    assert stopping.observe(1.0) == (True, False)
    assert stopping.observe(0.95) == (False, False)
    assert stopping.observe(0.96) == (False, True)
    assert stopping.observe(0.8) == (True, False)
    assert stopping.best_loss == 0.8
    assert stopping.bad_epochs == 0


@pytest.mark.parametrize(
    ("patience", "min_delta", "message"),
    [(0, 0.0, "patience"), (1, -0.1, "min_delta")],
)
def test_early_stopping_rejects_invalid_configuration(
    patience: int, min_delta: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        EarlyStopping(patience, min_delta)


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), float("-inf")])
def test_early_stopping_rejects_non_finite_loss(loss: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        EarlyStopping(1, 0.0).observe(loss)


def test_deterministic_seed_repeats_supported_generators() -> None:
    set_deterministic_seed(123)
    first_numpy = np.random.random(3)
    first_torch = torch.rand(3)

    set_deterministic_seed(123)

    np.testing.assert_array_equal(first_numpy, np.random.random(3))
    torch.testing.assert_close(first_torch, torch.rand(3), rtol=0.0, atol=0.0)
    assert torch.are_deterministic_algorithms_enabled()


def test_deterministic_seed_rejects_negative_value() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        set_deterministic_seed(-1)
