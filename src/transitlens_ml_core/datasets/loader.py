"""Loading and validation for processed TransitLens light curves."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

ModelInputField = Literal["normalized_flux", "wavelet_flux"]

_REQUIRED_ARRAYS = ("time", "normalized_flux", "wavelet_flux", "features_json")


@dataclass(frozen=True, slots=True)
class ProcessedLightCurve:
    """Validated processed light curve produced by the data pipeline.

    Attributes:
        time: Strictly increasing observation times.
        normalized_flux: Normalized flux values aligned with ``time``.
        wavelet_flux: Denoised flux values aligned with ``time``.
        metadata: Parsed pipeline feature and provenance metadata.

    """

    time: NDArray[np.float64]
    normalized_flux: NDArray[np.float64]
    wavelet_flux: NDArray[np.float64]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate and freeze cadence arrays regardless of construction path."""
        time = _validated_array(self.time, "time")
        normalized_flux = _validated_array(self.normalized_flux, "normalized_flux")
        wavelet_flux = _validated_array(self.wavelet_flux, "wavelet_flux")
        _validate_alignment(time, normalized_flux, wavelet_flux)
        if np.any(np.diff(time) <= 0.0):
            raise ValueError("time values must be strictly increasing")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "normalized_flux", normalized_flux)
        object.__setattr__(self, "wavelet_flux", wavelet_flux)


def load_processed_light_curve(path: str | Path) -> ProcessedLightCurve:
    """Load one pipeline-produced NumPy light-curve artifact.

    Args:
        path: Path to a deterministic ``.npz`` artifact from the data pipeline.

    Returns:
        A validated processed light curve with immutable array copies.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the extension, schema, metadata, or cadence data is invalid.

    """
    artifact_path = Path(path)
    if artifact_path.suffix.lower() != ".npz":
        raise ValueError("processed light-curve artifact must use the .npz extension")

    try:
        with np.load(artifact_path, allow_pickle=False) as artifact:
            missing = sorted(set(_REQUIRED_ARRAYS).difference(artifact.files))
            if missing:
                raise ValueError(
                    f"processed light-curve artifact is missing fields: {missing}"
                )
            time = artifact["time"]
            normalized_flux = artifact["normalized_flux"]
            wavelet_flux = artifact["wavelet_flux"]
            metadata = _decode_metadata(artifact["features_json"])
    except FileNotFoundError:
        raise
    except (OSError, EOFError) as error:
        raise ValueError(
            f"unable to read processed light-curve artifact: {path}"
        ) from error

    return ProcessedLightCurve(
        time=time,
        normalized_flux=normalized_flux,
        wavelet_flux=wavelet_flux,
        metadata=metadata,
    )


class LightCurveDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch dataset pairing processed light curves with binary labels."""

    def __init__(
        self,
        light_curves: Sequence[ProcessedLightCurve],
        labels: Sequence[int],
        input_field: ModelInputField,
    ) -> None:
        """Initialize a validated supervised dataset.

        Args:
            light_curves: Processed light curves to expose as model inputs.
            labels: Binary transit labels aligned with ``light_curves``.
            input_field: Flux representation used as the model input.

        Raises:
            ValueError: If inputs are empty, misaligned, or contain invalid labels.

        """
        if not light_curves:
            raise ValueError("dataset must contain at least one light curve")
        if len(light_curves) != len(labels):
            raise ValueError("light curves and labels must have equal lengths")
        if any(
            isinstance(label, bool)
            or not isinstance(label, (int, np.integer))
            or label not in (0, 1)
            for label in labels
        ):
            raise ValueError("labels must be binary integers")
        if input_field not in ("normalized_flux", "wavelet_flux"):
            raise ValueError("input_field must be normalized_flux or wavelet_flux")

        self._light_curves = tuple(light_curves)
        self._labels = tuple(labels)
        self._input_field = input_field

    @classmethod
    def from_files(
        cls,
        paths: Sequence[str | Path],
        labels: Sequence[int],
        input_field: ModelInputField,
    ) -> "LightCurveDataset":
        """Load processed artifacts and construct a supervised dataset.

        Args:
            paths: Ordered processed ``.npz`` artifact paths.
            labels: Binary transit labels aligned with ``paths``.
            input_field: Flux representation used as the model input.

        Returns:
            A validated PyTorch dataset.

        """
        if len(paths) != len(labels):
            raise ValueError("artifact paths and labels must have equal lengths")
        return cls(
            [load_processed_light_curve(path) for path in paths],
            labels,
            input_field,
        )

    def __len__(self) -> int:
        """Return the number of light curves in the dataset."""
        return len(self._light_curves)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one channel-first flux tensor and its binary target.

        Args:
            index: Dataset position to retrieve.

        Returns:
            A ``[1, samples]`` float tensor and scalar float target tensor.

        """
        curve = self._light_curves[index]
        flux = getattr(curve, self._input_field)
        features = torch.from_numpy(flux.copy()).to(dtype=torch.float32).unsqueeze(0)
        target = torch.tensor(self._labels[index], dtype=torch.float32)
        return features, target

    def light_curve(self, index: int) -> ProcessedLightCurve:
        """Return the validated source record for a dataset position.

        Args:
            index: Dataset position to retrieve.

        Returns:
            The immutable processed light-curve record.

        """
        return self._light_curves[index]


def _validated_array(array: NDArray[np.generic], field: str) -> NDArray[np.float64]:
    """Copy and validate one numeric cadence array."""
    if array.ndim != 1:
        raise ValueError(f"{field} must be a one-dimensional array")
    if array.size == 0:
        raise ValueError(f"{field} must not be empty")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{field} must contain numeric values")
    result = np.asarray(array, dtype=np.float64).copy()
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{field} must contain only finite values")
    result.setflags(write=False)
    return result


def _decode_metadata(encoded: NDArray[np.generic]) -> Mapping[str, Any]:
    """Decode the pipeline's canonical UTF-8 JSON metadata bytes."""
    if encoded.dtype != np.uint8 or encoded.ndim != 1:
        raise ValueError("features_json must be a one-dimensional uint8 array")
    try:
        metadata = json.loads(encoded.tobytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("features_json must contain valid UTF-8 JSON") from error
    if not isinstance(metadata, dict):
        raise ValueError("features_json root must be an object")
    return metadata


def _validate_alignment(*arrays: NDArray[np.float64]) -> None:
    """Require all cadence arrays to have the same sample count."""
    if len({len(array) for array in arrays}) != 1:
        raise ValueError("processed light-curve arrays must have equal lengths")
