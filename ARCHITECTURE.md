# ARCHITECTURE.md

# TransitLens ML Core Architecture

## High-Level Design

Input

↓

Dataset

↓

Training

↓

Model

↓

Evaluation

↓

Checkpoint

↓

Inference

---

# Directory Structure

src/

    datasets/

        loader.py

        transforms.py

        split.py

    models/

        cnn.py

        layers.py

        classifier.py

    training/

        trainer.py

        losses.py

        optimizer.py

        scheduler.py

    evaluation/

        metrics.py

        validation.py

    inference/

        predictor.py

        confidence.py

    explainability/

        integrated_gradients.py

    export/

        onnx.py

        checkpoint.py

configs/

weights/

tests/

experiments/

---

# Module Responsibilities

datasets

Responsible for

- dataset loading
- train validation split
- batching
- preprocessing checks

---

models

Contains neural network architectures.

Prototype

- 1D CNN
- Global Average Pooling
- Dense Classifier

Future

- Autoencoder
- Attention
- Feature Fusion

---

training

Responsible for

- epochs
- optimization
- checkpoint saving
- learning rate scheduling

---

evaluation

Responsible for

- accuracy
- precision
- recall
- F1
- ROC AUC
- confusion matrix

---

inference

Loads exported models.

Produces deterministic predictions.

No training logic.

---

explainability

Future expansion.

Prototype interface should exist even if implementation is minimal.

---

export

Responsible for

- ONNX export
- Torch checkpoint export

---

# Data Flow

Processed Dataset

↓

Dataset Loader

↓

CNN

↓

Classifier

↓

Probability

↓

Confidence

↓

Prediction

---

# Dependencies

Depends on

transitlens-data-pipeline

Independent from

transitlens-platform

Platform consumes inference outputs only.