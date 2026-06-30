"""Tests for processed light-curve loading and PyTorch datasets."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from transitlens_ml_core.datasets import (
    LightCurveDataset,
    ProcessedLightCurve,
    load_processed_light_curve,
)


def write_artifact(
    path: Path,
    *,
    time: object | None = None,
    normalized_flux: object | None = None,
    wavelet_flux: object | None = None,
    metadata: object | None = None,
) -> Path:
    """Write a representative pipeline-compatible NumPy artifact."""
    values = {
        "time": np.asarray(
            [1.0, 2.0, 3.0] if time is None else time,
        ),
        "normalized_flux": np.asarray(
            [1.0, 0.99, 1.01] if normalized_flux is None else normalized_flux,
        ),
        "wavelet_flux": np.asarray(
            [1.0, 0.995, 1.005] if wavelet_flux is None else wavelet_flux,
        ),
        "features_json": np.frombuffer(
            json.dumps(
                {"metadata": {"schema_version": "1.0"}}
                if metadata is None
                else metadata
            ).encode("utf-8"),
            dtype=np.uint8,
        ),
    }
    np.savez(path, **values)
    return path


def test_load_processed_light_curve_validates_and_freezes_arrays(
    tmp_path: Path,
) -> None:
    artifact = write_artifact(tmp_path / "curve.npz")

    curve = load_processed_light_curve(artifact)

    assert isinstance(curve, ProcessedLightCurve)
    assert curve.time.dtype == np.float64
    assert curve.metadata["metadata"]["schema_version"] == "1.0"
    assert not curve.time.flags.writeable
    assert not curve.normalized_flux.flags.writeable
    assert not curve.wavelet_flux.flags.writeable


def test_dataset_loads_files_and_returns_channel_first_tensors(tmp_path: Path) -> None:
    first = write_artifact(tmp_path / "first.npz")
    second = write_artifact(tmp_path / "second.npz", wavelet_flux=[1.0, 0.98, 1.0])

    dataset = LightCurveDataset.from_files(
        [first, second], labels=[0, 1], input_field="wavelet_flux"
    )
    features, target = dataset[1]

    assert len(dataset) == 2
    assert features.shape == (1, 3)
    assert features.dtype == torch.float32
    assert target.dtype == torch.float32
    assert target.item() == 1.0
    assert dataset.light_curve(1).metadata["metadata"]["schema_version"] == "1.0"


def test_dataset_can_select_normalized_flux() -> None:
    curve = ProcessedLightCurve(
        time=np.array([1.0]),
        normalized_flux=np.array([0.9]),
        wavelet_flux=np.array([0.8]),
        metadata={},
    )

    features, _ = LightCurveDataset([curve], [0], "normalized_flux")[0]

    assert features.item() == pytest.approx(0.9)


def test_processed_record_validates_direct_construction() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        ProcessedLightCurve(
            time=np.array([2.0, 1.0]),
            normalized_flux=np.ones(2),
            wavelet_flux=np.ones(2),
            metadata={},
        )
    with pytest.raises(ValueError, match="metadata must be a mapping"):
        ProcessedLightCurve(
            time=np.array([1.0]),
            normalized_flux=np.ones(1),
            wavelet_flux=np.ones(1),
            metadata=[],  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("curves", "labels", "input_field", "message"),
    [
        ([], [], "wavelet_flux", "at least one"),
        ([object()], [], "wavelet_flux", "equal lengths"),
        ([object()], [2], "wavelet_flux", "binary integers"),
        ([object()], [True], "wavelet_flux", "binary integers"),
        ([object()], [1.0], "wavelet_flux", "binary integers"),
        ([object()], [0], "flux", "input_field"),
    ],
)
def test_dataset_rejects_invalid_construction(
    curves: list[object], labels: list[object], input_field: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        LightCurveDataset(curves, labels, input_field)  # type: ignore[arg-type]


def test_from_files_checks_alignment_before_loading(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        LightCurveDataset.from_files(
            [tmp_path / "missing.npz"], [], input_field="wavelet_flux"
        )


def test_loader_rejects_wrong_extension_and_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"\.npz"):
        load_processed_light_curve(tmp_path / "curve.parquet")
    with pytest.raises(FileNotFoundError):
        load_processed_light_curve(tmp_path / "missing.npz")


def test_loader_wraps_unreadable_artifact(tmp_path: Path) -> None:
    unreadable_path = tmp_path / "directory.npz"
    unreadable_path.mkdir()

    with pytest.raises(ValueError, match="unable to read"):
        load_processed_light_curve(unreadable_path)


def test_loader_rejects_missing_required_field(tmp_path: Path) -> None:
    path = tmp_path / "curve.npz"
    np.savez(path, time=np.array([1.0]))

    with pytest.raises(ValueError, match="missing fields"):
        load_processed_light_curve(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("time", [[1.0, 2.0]], "one-dimensional"),
        ("normalized_flux", [], "must not be empty"),
        ("wavelet_flux", ["bad"], "numeric"),
        ("normalized_flux", [1.0, np.nan, 1.0], "finite"),
    ],
)
def test_loader_rejects_invalid_arrays(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    arguments = {field: value}
    path = write_artifact(tmp_path / "curve.npz", **arguments)

    with pytest.raises(ValueError, match=message):
        load_processed_light_curve(path)


def test_loader_rejects_misaligned_or_unsorted_cadences(tmp_path: Path) -> None:
    misaligned = write_artifact(tmp_path / "misaligned.npz", normalized_flux=[1.0, 1.0])
    unsorted = write_artifact(tmp_path / "unsorted.npz", time=[1.0, 3.0, 2.0])

    with pytest.raises(ValueError, match="equal lengths"):
        load_processed_light_curve(misaligned)
    with pytest.raises(ValueError, match="strictly increasing"):
        load_processed_light_curve(unsorted)


def test_loader_rejects_invalid_metadata_encoding(tmp_path: Path) -> None:
    invalid_utf8 = write_artifact(tmp_path / "invalid.npz")
    with np.load(invalid_utf8) as artifact:
        values = {name: artifact[name] for name in artifact.files}
    values["features_json"] = np.array([255], dtype=np.uint8)
    np.savez(invalid_utf8, **values)

    with pytest.raises(ValueError, match="valid UTF-8 JSON"):
        load_processed_light_curve(invalid_utf8)


def test_loader_rejects_invalid_metadata_shape_and_root(tmp_path: Path) -> None:
    invalid_shape = write_artifact(tmp_path / "shape.npz")
    with np.load(invalid_shape) as artifact:
        values = {name: artifact[name] for name in artifact.files}
    values["features_json"] = np.array([[1]], dtype=np.uint8)
    np.savez(invalid_shape, **values)
    invalid_root = write_artifact(tmp_path / "root.npz", metadata=[])

    with pytest.raises(ValueError, match="one-dimensional uint8"):
        load_processed_light_curve(invalid_shape)
    with pytest.raises(ValueError, match="root must be an object"):
        load_processed_light_curve(invalid_root)
