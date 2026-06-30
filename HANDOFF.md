# HANDOFF.md

# TransitLens ML Core Handoff

## Repository Status

Current Phase

Phase 7 - Model Export (Completed)

Next Phase

None - All Prototype Phases Completed

---

## Public Interface

Input

ProcessedLightCurve

Fields

- time

- normalized_flux

- wavelet_flux

- metadata

Output

PredictionResult

Fields

- probability

- confidence

- predicted_class

- model_version

- inference_time

---

## Contracts

Input format is owned by

transitlens-data-pipeline

Output format is consumed by

transitlens-platform

Public interfaces must remain stable.

---

## Current Deliverables

Completed

- Python package structure under `src/transitlens_ml_core`

- Python 3.11+ project and dependency configuration in `pyproject.toml`

- PyTorch runtime dependency

- Ruff, Black, pytest, and 90% coverage gate configuration

- Immutable Pydantic configuration models and YAML loader

- Checked-in prototype configuration at `configs/prototype.yaml`

- Configuration validation tests (8 passing, 100% coverage)

- Immutable processed light-curve records loaded from pipeline `.npz` artifacts

- Validation for the fixed `time`, `normalized_flux`, `wavelet_flux`, and
  metadata input contract

- PyTorch dataset with separately supplied binary supervised labels

- Configuration-selected `normalized_flux` or `wavelet_flux` model inputs

- Seeded, non-overlapping train, validation, and test splits

- Dataset and configuration tests (36 passing, 100% coverage)

- Modular two-block one-dimensional CNN feature extractor

- Convolution, batch normalization, ReLU, and max-pooling layer composition

- Global average pooling and dense sigmoid binary classifier

- Configuration-driven `BaselineCNN.from_config` construction

- Shape and channel validation at the model boundary

- Model, dataset, and configuration tests (56 passing, 100% coverage)

- Deterministic sample-weighted training and validation loops

- Configuration-driven binary cross-entropy, Adam optimizer, and
  reduce-on-plateau learning-rate scheduler

- Validation-loss early stopping and best-model selection

- Atomic best and latest training checkpoints with safe resume support

- Model, optimizer, scheduler, early-stopping, history, and RNG checkpoint state

- Atomic human-readable JSON training history after every epoch

- Exact tested continuity between resumed and uninterrupted CPU training

- Training, model, dataset, and configuration tests (77 passing, above 95%
  coverage)

- Validated accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix
  calculation

- Configuration-driven inclusive binary classification threshold

- Inference-only model evaluation over labeled data loaders

- Stable 2×2 confusion matrix, including single-class test datasets

- Atomic, versioned JSON evaluation reports generated on successful evaluation

- Evaluation preserves the model's prior training or evaluation mode

- Evaluation, training, model, dataset, and configuration tests (99 passing,
  above 96% coverage)

- Safe restricted loading of baseline weights from Phase 4 checkpoints

- Configuration-driven single-light-curve prediction from either validated flux
  representation

- Stable immutable `PredictionResult` output with probability, confidence,
  predicted class, model version, and inference time

- Normalized decision-confidence estimation relative to the classification
  threshold

- Deterministic inference under `torch.inference_mode()` with inclusive class
  thresholding

- Model-only inference timing in milliseconds with CUDA synchronization support

- Inference, evaluation, training, model, dataset, and configuration tests (139
  passing, above 96% coverage)

- Atomic self-describing PyTorch inference checkpoint export

- Restricted loading of exported PyTorch artifacts directly into `Predictor`

- Dynamic-batch and dynamic-cadence ONNX export with stable tensor names

- Embedded ONNX metadata for schema version, model version, architecture, input
  field, classification threshold, and tensor names

- ONNX checker validation before artifact publication

- Verified numerical equivalence between PyTorch and ONNX across multiple batch
  sizes and cadence lengths

- Composed export service producing both platform formats

- Export, inference, evaluation, training, model, dataset, and configuration
  tests (158 passing, above 95% coverage)

Pending

None for the seven-phase prototype plan.

---

## Known Risks

Dataset imbalance

False positives

Overfitting

Model calibration

Small training dataset

---

## Future Expansion

Denoising Autoencoder

Attention

Explainability

Feature Fusion

Model Ensemble

---

## Notes For Next Repository

Platform should never directly access datasets.

Platform communicates only with exported inference APIs.

---

## Phase 1 Decisions

- The existing `ProcessedLightCurve` and `PredictionResult` contracts were not
  changed.

- Dataset, model, training, evaluation, inference, and export packages contain
  package boundaries only; their implementations remain scoped to later phases.

- The default model input is configured as `wavelet_flux`, while all four fixed
  processed-input field names remain explicit and validated in configuration.

- Runtime paths and hyperparameters live in YAML rather than source code.

- Verified with PyTorch 2.12.1 on the available Python 3.14 host. The package
  declares support for Python 3.11 through 3.14.

---

## Phase 2 Decisions

- The canonical NumPy artifact produced by `transitlens-data-pipeline` stores
  the public `metadata` field as UTF-8 JSON bytes under `features_json`. The
  loader decodes this producer-specific serialization back into the stable
  `ProcessedLightCurve.metadata` interface.

- Supervised labels are supplied separately when constructing
  `LightCurveDataset`; no label was added to the fixed processed-input schema.

- The loader accepts only safe, pickle-free `.npz` artifacts in this phase.
  Parquet support was not added because it would introduce a second loader
  interface and dependencies not required by the prototype contract.

- All cadence arrays must be non-empty, one-dimensional, finite, equally sized,
  and ordered by strictly increasing time.

- Splitting is deterministic for a configured seed and guarantees non-empty
  train, validation, and test subsets. At least three labeled curves are
  therefore required for splitting.

---

## Phase 3 Decisions

- Feature extraction and classification are separate modules so a denoising
  autoencoder can be placed before the CNN and attention can consume the CNN
  feature sequence later without changing the classifier contract.

- Convolutions use symmetric length-preserving padding and therefore require
  positive odd kernel sizes. The checked-in configuration remains at kernel
  size 5.

- Max pooling is applied only after the first convolution block, matching the
  prototype architecture. Global average pooling makes the classifier support
  variable input sequence lengths.

- The unused Phase 1 `dropout` configuration key was removed because dropout
  is not part of the source-of-truth prototype architecture. This is an
  intentional configuration-interface correction made before training and is
  validated by strict configuration tests.

- The prototype has 2,769 trainable parameters. A warmed single-sample forward
  pass over 1,024 cadences averaged 0.809 ms on the available CPU, well below
  the 100 ms target (model loading excluded).

---

## Phase 4 Decisions

- Training uses binary cross-entropy because the source-of-truth model contract
  requires sigmoid probability output. Targets are reshaped to the model's
  stable `[batch, 1]` output inside the trainer.

- Epoch losses are weighted by batch size, so history remains correct when the
  final batch is smaller than the configured batch size.

- Both best and latest checkpoints are written atomically. Latest checkpoints
  contain model, optimizer, scheduler, early-stopping, epoch history, and
  PyTorch RNG states required to continue training.

- Resume loading uses PyTorch's restricted `weights_only=True` mode. The
  checkpoint schema is versioned and validated before state restoration.

- Training history records the learning rate actually used for each epoch and
  is atomically persisted as sorted, indented JSON after every epoch.

- `set_deterministic_seed` must be called before model and shuffled data-loader
  construction to make initialization and data order reproducible. `Trainer`
  calls it again at startup for training-time stochastic operations and enables
  deterministic PyTorch algorithms.

- Training artifacts and optimizer/scheduler behavior are configuration-driven;
  no paths, filenames, learning rates, tolerances, or optimizer coefficients are
  embedded in trainer logic.

---

## Phase 5 Decisions

- Metric calculation is a pure validated function, while loader inference and
  report persistence are composed separately in `evaluation.validation`.

- Probabilities equal to the configured classification threshold are classified
  as positive. The prototype threshold is 0.5 and is stored in configuration.

- Confusion matrices always use label order `[0, 1]`, producing the stable
  layout `[[true_negative, false_positive], [false_negative, true_positive]]`.

- ROC-AUC is reported as `null` when the evaluated targets contain only one
  class because the metric is mathematically undefined in that case. All other
  metrics and the fixed 2×2 confusion matrix are still generated.

- Precision, recall, and F1 use zero for undefined divisions, avoiding warnings
  and unstable report values when no positive predictions or labels exist.

- Evaluation runs under `torch.inference_mode()` and restores the model's prior
  train/eval mode even when the model returns invalid output.

- Reports are written atomically with schema version 1, sample count, threshold,
  and the complete metric collection. Report paths and filenames are fully
  configuration-driven.

---

## Phase 6 Decisions

- `PredictionResult` implements exactly the pre-existing public fields:
  `probability`, `confidence`, `predicted_class`, `model_version`, and
  `inference_time`. Its serializer preserves that field order and adds nothing.

- `inference_time` is measured in milliseconds and covers model execution only;
  checkpoint loading and processed-record tensor preparation are excluded.

- Checkpoints are loaded with restricted `weights_only=True` deserialization.
  The root mapping, required `model_state_dict`, and strict architecture state
  compatibility are validated before the predictor is returned.

- Predictor construction composes existing project, data, model, and evaluation
  configuration rather than adding a duplicate inference configuration section.

- Confidence is normalized distance from the configured decision threshold: zero
  at the class boundary and one at probability 0 or 1. It is explicitly decision
  confidence, not statistical calibration; probability calibration remains a
  known model-quality risk requiring representative validation data.

- Direct `PredictionResult` construction validates every public value, preventing
  non-finite probabilities/timings, invalid classes, or empty versions from
  crossing the platform boundary.

- A warmed 100-run benchmark over 1,024 cadences averaged 1.961 ms on the
  available CPU, with a maximum of 28.659 ms, below the 100 ms per-sample target
  (model loading excluded).

---

## Phase 7 Decisions

- The PyTorch export is distinct from the resumable training checkpoint. It
  contains only inference weights and the metadata required to reconstruct the
  model and platform-facing predictor without external configuration.

- PyTorch export schema version 1 contains the format identifier, model version,
  complete baseline architecture configuration, processed input field,
  classification threshold, stable tensor names, and CPU model state.

- ONNX input is named `light_curve` with shape
  `[dynamic_batch, input_channels, dynamic_samples]`; output is named
  `transit_probability` with shape `[dynamic_batch, 1]`.

- ONNX artifacts embed the same version, architecture, input-field, threshold,
  and tensor-name metadata used by the PyTorch artifact, allowing the platform
  to consume either format without accessing training code.

- The TorchScript-based ONNX exporter is used deliberately with the declared
  `torch<3` compatibility range. On the available PyTorch 2.12 runtime, the
  modern dynamo exporter specialized the cadence dimension despite dynamic
  declarations; the selected exporter preserves and tests dynamic cadence
  lengths correctly. Known exporter-only deprecation warnings are locally
  suppressed.

- Every artifact is written to a temporary file, validated, and atomically moved
  into its configured destination. An incomplete ONNX artifact is never
  published after conversion or checker failure.

- ONNX reference execution matches PyTorch within `rtol=1e-5` and `atol=1e-6`
  for batches of 1, 2, and 3 and cadence lengths of 16, 32, and 65.
