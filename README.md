# transitlens-ml-core

> AI-enabled detection and classification of exoplanet transit signals from noisy astronomical light curves.

**Bharatiya Antariksh Hackathon 2026 — Problem Statement PS7**

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/your-org/transitlens-ml-core.git
cd transitlens-ml-core

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the analysis pipeline (Python)
python -c "
from pipeline import analyze_light_curve
import numpy as np
rng = np.random.default_rng(42)
time = np.linspace(0, 27, 18000)
flux = 1.0 + rng.normal(0, 0.001, 18000)
result = analyze_light_curve(time, flux)
print(f'Class: {result[\"predicted_class\"]}  Confidence: {result[\"confidence\"]:.2f}')
"

# 4. Start the API server
uvicorn api.app:app --host 0.0.0.0 --port 8000

# 5. Test the API
curl http://localhost:8000/health
curl http://localhost:8000/demo/a
```

## What This Repo Does

`transitlens-ml-core` is the **brain** of the TransitLens system. It receives a raw light curve and returns a complete analysis result including:

- **Transit detection** via Box Least Squares (BLS) period search
- **16 physically-interpretable features** extracted from the detected signal
- **Classification** into `exoplanet_transit`, `eclipsing_binary`, `blend_contamination`, or `stellar_variability_or_other`
- **Calibrated confidence score** with per-component breakdown
- **4 diagnostic plots** as base64-encoded PNG strings
- **Human-readable explanation** of the classification decision

### Position in the tri-repo system

```
transitlens-data-pipeline  →  transitlens-ml-core  →  transitlens-platform
       load_light_curve()       analyze_light_curve()     POST /analyze
          (feeds)                    (analyses)               (displays)
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service health check |
| `POST` | `/analyze` | Analyse a light curve (full pipeline) |
| `GET` | `/demo/{a\|b\|c}` | Run a synthetic demo case |
| `GET` | `/docs` | Swagger UI (auto-generated) |

Start the server:
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

## Project Structure

```
transitlens-ml-core/
├── pipeline.py                    ← analyze_light_curve() — public entry point
├── config.yaml                    ← global pipeline configuration
├── core/
│   ├── preprocess.py              ← normalise, sigma-clip, detrend
│   ├── bls_detector.py            ← BLS period search
│   ├── feature_extractor.py       ← 11 interpretable features
│   ├── classifier.py              ← rule-based + optional ML classifier
│   ├── confidence.py              ← calibrated confidence scoring
│   ├── plotter.py                 ← 4 diagnostic plots (base64 PNG)
│   ├── utils.py                   ← shared math utilities
│   └── exceptions.py              ← custom exception hierarchy
├── api/
│   ├── app.py                     ← FastAPI application
│   ├── routes.py                  ← endpoint handlers
│   ├── schema.py                  ← Pydantic request/response models
│   └── middleware.py              ← timing, logging, error handling
├── models/
│   ├── rule_config.yaml           ← all classification thresholds
│   ├── model_card.md              ← model documentation
│   └── *.pkl                      ← optional trained models
├── eval/
│   ├── evaluate.py                ← classification report generator
│   ├── metrics.py                 ← precision, recall, F1, period recovery
│   └── benchmark.py               ← speed benchmarks
└── tests/                         ← full pytest test suite
```

## Running Tests

```bash
pytest tests/ -v
```

## Running Evaluation

```bash
python -m eval.evaluate     # classification report + confusion matrix
python -m eval.benchmark    # speed benchmarks
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.24 | Array operations |
| scipy | ≥1.11 | Statistics, signal processing |
| astropy | ≥5.3 | BLS implementation (primary) |
| scikit-learn | ≥1.3 | Optional ML classifier |
| matplotlib | ≥3.7 | Diagnostic plots |
| fastapi | ≥0.110 | HTTP API |
| uvicorn | ≥0.27 | ASGI server |
| pydantic | ≥2.5 | Request/response validation |
| pyyaml | ≥6.0 | Config file parsing |

Install all:
```bash
pip install -r requirements.txt
```

## Configuration

All pipeline parameters are configurable via `config.yaml`. Override any parameter at runtime:

```python
result = analyze_light_curve(
    time, flux,
    config={"bls": {"period_max_days": 5.0}}
)
```

Classification thresholds live in `models/rule_config.yaml` — change a value there and the classification behaviour changes without modifying Python code.

## License

See [LICENSE](LICENSE) for details.
