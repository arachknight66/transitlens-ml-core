"""Validated application configuration loading."""

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator


class StrictConfigModel(BaseModel):
    """Base model that rejects unknown configuration keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectConfig(StrictConfigModel):
    """Project identity and reproducibility configuration."""

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    seed: int = Field(ge=0)


class DataConfig(StrictConfigModel):
    """Processed dataset schema and split configuration."""

    input_directory: Path
    time_field: str = Field(min_length=1)
    normalized_flux_field: str = Field(min_length=1)
    wavelet_flux_field: str = Field(min_length=1)
    metadata_field: str = Field(min_length=1)
    model_input_field: Literal["normalized_flux", "wavelet_flux"]
    train_fraction: float = Field(gt=0.0, lt=1.0)
    validation_fraction: float = Field(gt=0.0, lt=1.0)
    test_fraction: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_split_fractions(self) -> Self:
        """Ensure the dataset fractions form a complete split.

        Returns:
            The validated data configuration.

        Raises:
            ValueError: If the split fractions do not sum to one.

        """
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError("dataset split fractions must sum to 1.0")
        return self


class ModelConfig(StrictConfigModel):
    """Baseline CNN hyperparameter configuration."""

    input_channels: PositiveInt
    convolution_channels: tuple[PositiveInt, PositiveInt]
    kernel_size: PositiveInt
    pool_size: PositiveInt

    @model_validator(mode="after")
    def validate_odd_kernel(self) -> Self:
        """Require odd kernels so length-preserving padding is symmetric.

        Returns:
            The validated model configuration.

        Raises:
            ValueError: If the convolution kernel size is even.

        """
        if self.kernel_size % 2 == 0:
            raise ValueError("model kernel_size must be odd")
        return self


class TrainingConfig(StrictConfigModel):
    """Training lifecycle configuration."""

    batch_size: PositiveInt
    epochs: PositiveInt
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    optimizer_betas: tuple[float, float]
    optimizer_epsilon: float = Field(gt=0.0)
    early_stopping_patience: PositiveInt
    early_stopping_min_delta: float = Field(ge=0.0)
    scheduler_patience: PositiveInt
    scheduler_factor: float = Field(gt=0.0, lt=1.0)
    scheduler_min_lr: float = Field(ge=0.0)
    checkpoint_directory: Path
    best_checkpoint_filename: str = Field(min_length=1)
    latest_checkpoint_filename: str = Field(min_length=1)
    experiment_directory: Path
    history_filename: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_training_artifacts(self) -> Self:
        """Validate optimizer values and artifact filenames.

        Returns:
            The validated training configuration.

        Raises:
            ValueError: If optimizer betas or artifact filenames are invalid.

        """
        if any(beta < 0.0 or beta >= 1.0 for beta in self.optimizer_betas):
            raise ValueError("optimizer betas must be in the range [0, 1)")
        filenames = (
            (self.best_checkpoint_filename, ".pt"),
            (self.latest_checkpoint_filename, ".pt"),
            (self.history_filename, ".json"),
        )
        for filename, suffix in filenames:
            if Path(filename).name != filename or not filename.endswith(suffix):
                raise ValueError(
                    f"artifact filename must be a basename ending in {suffix}"
                )
        if self.best_checkpoint_filename == self.latest_checkpoint_filename:
            raise ValueError("best and latest checkpoint filenames must differ")
        return self


class ExportConfig(StrictConfigModel):
    """Model export configuration."""

    onnx_opset_version: int = Field(ge=17)
    sample_length: PositiveInt
    output_directory: Path
    pytorch_filename: str = Field(min_length=1)
    onnx_filename: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_export_filenames(self) -> Self:
        """Require safe artifact basenames with format-specific suffixes.

        Returns:
            The validated export configuration.

        Raises:
            ValueError: If an artifact filename contains a path or wrong suffix.

        """
        filenames = (
            (self.pytorch_filename, ".pt"),
            (self.onnx_filename, ".onnx"),
        )
        for filename, suffix in filenames:
            if Path(filename).name != filename or not filename.endswith(suffix):
                raise ValueError(
                    f"export filename must be a basename ending in {suffix}"
                )
        return self


class EvaluationConfig(StrictConfigModel):
    """Binary metric and report configuration."""

    classification_threshold: float = Field(gt=0.0, lt=1.0)
    report_directory: Path
    report_filename: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_report_filename(self) -> Self:
        """Require a safe JSON report basename.

        Returns:
            The validated evaluation configuration.

        Raises:
            ValueError: If the report filename contains a path or wrong suffix.

        """
        if Path(
            self.report_filename
        ).name != self.report_filename or not self.report_filename.endswith(".json"):
            raise ValueError("evaluation report filename must be a .json basename")
        return self


class AppConfig(StrictConfigModel):
    """Top-level TransitLens ML configuration."""

    project: ProjectConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    export: ExportConfig


def load_config(path: str | Path) -> AppConfig:
    """Load and validate a YAML configuration file.

    Args:
        path: Path to a YAML configuration file.

    Returns:
        A validated, immutable application configuration.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the YAML document is empty or not a mapping.
        yaml.YAMLError: If the file contains malformed YAML.
        pydantic.ValidationError: If configuration values are invalid.

    """
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)

    if not isinstance(raw_config, dict):
        raise ValueError("configuration root must be a mapping")

    return AppConfig.model_validate(raw_config)
