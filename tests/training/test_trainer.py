"""Tests for the deterministic training lifecycle."""

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from transitlens_ml_core.config import ModelConfig
from transitlens_ml_core.models import BaselineCNN
from transitlens_ml_core.training import Trainer

from .conftest import make_loaders, make_model, make_training_config


def test_trainer_learns_and_persists_complete_artifacts(tmp_path: Path) -> None:
    model = make_model()
    initial_weight = next(model.parameters()).detach().clone()
    config = make_training_config(tmp_path, epochs=4)
    training_loader, validation_loader = make_loaders()
    trainer = Trainer(model, config, torch.device("cpu"), seed=19)

    result = trainer.fit(training_loader, validation_loader)

    assert len(result.history) == 4
    assert result.history[-1].training_loss < result.history[0].training_loss
    assert result.best_epoch == 4
    assert not result.stopped_early
    assert not torch.equal(initial_weight, next(model.parameters()))
    assert result.best_checkpoint.is_file()
    assert result.latest_checkpoint.is_file()
    assert result.history_path.is_file()
    history_document = json.loads(result.history_path.read_text(encoding="utf-8"))
    assert history_document[-1]["epoch"] == 4
    checkpoint = torch.load(result.latest_checkpoint, weights_only=True)
    assert checkpoint["schema_version"] == 1
    assert checkpoint["epoch"] == 4
    assert len(checkpoint["history"]) == 4
    assert "optimizer_state_dict" in checkpoint
    assert "scheduler_state_dict" in checkpoint


def test_trainer_runs_baseline_cnn_end_to_end(tmp_path: Path) -> None:
    negative = torch.ones(4, 1, 16)
    positive = torch.ones(4, 1, 16)
    positive[:, :, 6:10] = 0.9
    dataset = TensorDataset(
        torch.cat((negative, positive)),
        torch.cat((torch.zeros(4), torch.ones(4))),
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model = BaselineCNN.from_config(
        ModelConfig(
            input_channels=1,
            convolution_channels=(2, 4),
            kernel_size=3,
            pool_size=2,
        )
    )
    trainer = Trainer(
        model,
        make_training_config(tmp_path, epochs=1),
        torch.device("cpu"),
        seed=7,
    )

    result = trainer.fit(loader, loader)

    assert len(result.history) == 1
    assert torch.isfinite(torch.tensor(result.history[0].validation_loss))


def test_training_is_reproducible(tmp_path: Path) -> None:
    first_loaders = make_loaders()
    first = Trainer(
        make_model(31),
        make_training_config(tmp_path / "first", epochs=3),
        torch.device("cpu"),
        seed=31,
    ).fit(*first_loaders)
    second_loaders = make_loaders()
    second = Trainer(
        make_model(31),
        make_training_config(tmp_path / "second", epochs=3),
        torch.device("cpu"),
        seed=31,
    ).fit(*second_loaders)

    assert first.history == second.history
    assert first.best_validation_loss == second.best_validation_loss


def test_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    full_model = make_model(41)
    full = Trainer(
        full_model,
        make_training_config(tmp_path / "full", epochs=4),
        torch.device("cpu"),
        seed=41,
    ).fit(*make_loaders())

    partial_config = make_training_config(tmp_path / "resumed", epochs=2)
    partial = Trainer(make_model(41), partial_config, torch.device("cpu"), seed=41).fit(
        *make_loaders()
    )
    resumed_model = make_model(999)
    resumed = Trainer(
        resumed_model,
        make_training_config(tmp_path / "resumed", epochs=4),
        torch.device("cpu"),
        seed=999,
    ).fit(*make_loaders(), resume_from=partial.latest_checkpoint)

    assert resumed.history == full.history
    for full_parameter, resumed_parameter in zip(
        full_model.parameters(), resumed_model.parameters(), strict=True
    ):
        torch.testing.assert_close(
            full_parameter, resumed_parameter, rtol=0.0, atol=0.0
        )


def test_early_stopping_ends_training(tmp_path: Path) -> None:
    config = make_training_config(
        tmp_path,
        epochs=8,
        learning_rate=1e-6,
        early_stopping_patience=1,
        early_stopping_min_delta=1.0,
    )
    trainer = Trainer(make_model(), config, torch.device("cpu"), seed=1)

    result = trainer.fit(*make_loaders())

    assert result.stopped_early
    assert len(result.history) == 2
    assert result.best_epoch == 1


def test_fit_rejects_empty_loaders(tmp_path: Path) -> None:
    empty_dataset = TensorDataset(torch.empty(0, 1, 4), torch.empty(0))
    empty_loader = DataLoader(empty_dataset, batch_size=2)
    trainer = Trainer(
        make_model(),
        make_training_config(tmp_path),
        torch.device("cpu"),
        seed=1,
    )

    with pytest.raises(ValueError, match="must not be empty"):
        trainer.fit(empty_loader, empty_loader)


def test_resume_rejects_missing_and_completed_checkpoint(tmp_path: Path) -> None:
    config = make_training_config(tmp_path, epochs=1)
    trainer = Trainer(make_model(), config, torch.device("cpu"), seed=1)

    with pytest.raises(ValueError, match="does not exist"):
        trainer.fit(*make_loaders(), resume_from=tmp_path / "missing.pt")

    completed = trainer.fit(*make_loaders())
    new_trainer = Trainer(make_model(), config, torch.device("cpu"), seed=1)
    with pytest.raises(ValueError, match="already reaches"):
        new_trainer.fit(*make_loaders(), resume_from=completed.latest_checkpoint)


def test_resume_rejects_unreadable_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "corrupt.pt"
    checkpoint.write_bytes(b"not a torch checkpoint")
    trainer = Trainer(
        make_model(),
        make_training_config(tmp_path),
        torch.device("cpu"),
        seed=1,
    )

    with pytest.raises(ValueError, match="unable to read"):
        trainer.fit(*make_loaders(), resume_from=checkpoint)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "root must be a mapping"),
        ({"schema_version": 1}, "missing fields"),
        (
            {
                "schema_version": 99,
                "epoch": 1,
                "model_state_dict": {},
                "optimizer_state_dict": {},
                "scheduler_state_dict": {},
                "best_validation_loss": 1.0,
                "bad_epochs": 0,
                "history": [],
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": [],
            },
            "unsupported",
        ),
    ],
)
def test_resume_rejects_invalid_checkpoint(
    tmp_path: Path, payload: object, message: str
) -> None:
    path = tmp_path / "invalid.pt"
    torch.save(payload, path)
    trainer = Trainer(
        make_model(),
        make_training_config(tmp_path),
        torch.device("cpu"),
        seed=1,
    )

    with pytest.raises(ValueError, match=message):
        trainer.fit(*make_loaders(), resume_from=path)
