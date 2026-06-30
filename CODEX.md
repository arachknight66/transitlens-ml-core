# CODEX.md

# TransitLens ML Core Codex Guide

Read this document before implementing code.

---

# Repository Mission

Build a modular machine learning engine capable of detecting exoplanet transit signals from preprocessed astronomical light curves.

The implementation should prioritize maintainability, reproducibility, and future extensibility.

---

# Scope

Implement

- Dataset loading
- Dataset validation
- Baseline CNN
- Training
- Evaluation
- Inference
- Checkpointing
- ONNX export

---

# Do Not Implement

FITS parsing

Astroquery

Wavelet preprocessing

Frontend

FastAPI

Database

Authentication

Background jobs

Scheduler

---

# Technology Stack

Python 3.11+

PyTorch

NumPy

Scikit-learn

TensorBoard

ONNX

Pydantic

PyYAML

---

# Baseline Architecture

Input

↓

1D Convolution

↓

Batch Normalization

↓

ReLU

↓

Max Pool

↓

1D Convolution

↓

Batch Normalization

↓

ReLU

↓

Global Average Pooling

↓

Fully Connected

↓

Sigmoid

Output

Transit Probability

The architecture should be modular so that a Denoising Autoencoder can later be inserted before the CNN and an Attention block after the CNN without major refactoring.

---

# Directory Structure

Do not change the planned directory layout without justification.

datasets

models

training

evaluation

inference

export

configs

weights

tests

experiments

---

# Coding Rules

Every public function must include

- Type hints
- Google-style docstrings

Use dataclasses or Pydantic models for structured inputs and outputs.

Avoid global state.

Use dependency injection where practical.

Never hardcode paths or hyperparameters.

All configurable values must come from configuration files.

---

# Model Rules

Prototype model

- Lightweight
- Fast to train
- Small memory footprint

Target

Inference under 100 ms per sample on CPU (excluding model loading).

---

# Training Rules

Implement

- Early stopping
- Checkpoint saving
- Best-model selection
- Learning-rate scheduling
- Deterministic random seeds

Support resuming from checkpoints.

---

# Evaluation Rules

Automatically compute

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

Generate evaluation reports after training.

---

# Export Rules

Support

- PyTorch checkpoint (.pt)
- ONNX

The exported model should be directly consumable by transitlens-platform.

---

# Testing

Every module requires tests.

Target coverage

90%

Test

- Dataset loader
- Model forward pass
- Training loop
- Inference
- Export

---

# Files You May Modify

src

tests

configs

weights

experiments

---

# Files You Must Not Modify

README.md

ARCHITECTURE.md

TASKS.md

HANDOFF.md

CODEX.md

unless explicitly instructed.

---

# Git Guidelines

One logical feature per commit.

Examples

feat(models): implement baseline cnn

feat(training): add early stopping

feat(inference): add predictor

---

# If Requirements Are Ambiguous

Do not invent architecture.

Prefer the simplest production-quality solution.

Document assumptions in code comments or pull request notes.

---

# Definition of Done

The repository is complete when it can

- Load processed datasets
- Train a baseline CNN
- Evaluate performance
- Export the best model
- Run deterministic inference
- Produce confidence scores
- Pass all tests
- Expose stable interfaces for transitlens-platform

Future features should be addable without breaking existing public APIs.