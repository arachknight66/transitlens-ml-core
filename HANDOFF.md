# HANDOFF.md

# TransitLens ML Core Handoff

## Repository Status

Current Phase

Not Started

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

None

Pending

- Dataset loader
- CNN
- Training
- Evaluation
- Inference
- Export

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