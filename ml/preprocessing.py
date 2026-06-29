from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from .contracts import ContractError

@dataclass
class FrozenPreprocessor:
    feature_order: list[str]
    imputer: SimpleImputer | None = None
    lower_bounds: np.ndarray | None = None
    upper_bounds: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> "FrozenPreprocessor":
        self._validate(frame)
        values = frame.loc[:, self.feature_order].astype(float)
        self.imputer = SimpleImputer(strategy="median", add_indicator=True).fit(values)
        self.lower_bounds = values.quantile(0.001).to_numpy()
        self.upper_bounds = values.quantile(0.999).to_numpy()
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        self._validate(frame)
        if self.imputer is None:
            raise ContractError("preprocessor has not been fitted")
        return self.imputer.transform(frame.loc[:, self.feature_order].astype(float))

    def out_of_range(self, frame: pd.DataFrame) -> np.ndarray:
        self._validate(frame)
        values = frame.loc[:, self.feature_order].astype(float).to_numpy()
        return np.any((values < self.lower_bounds) | (values > self.upper_bounds), axis=1)

    def _validate(self, frame: pd.DataFrame) -> None:
        missing = [name for name in self.feature_order if name not in frame]
        if missing:
            raise ContractError(f"missing features: {missing}")
        model_like = [c for c in frame.columns if c not in self.feature_order and c not in {
            "tic_id", "target_id", "observation_id", "sector", "split", "label",
            "canonical_label", "label_strength", "evidence_level", "source_checksum",
            "phase2_diagnostics_version", "feature_schema_version", "ephemeris_mode",
            "candidate_detected", "source_type",
        }]
        if model_like:
            raise ContractError(f"unknown model features: {model_like}")
