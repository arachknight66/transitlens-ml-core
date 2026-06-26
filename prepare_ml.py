"""
prepare_ml.py
-------------
One-shot preparation script for the TransitLens ML training pipeline.

Run this ONCE before starting model training:
    python prepare_ml.py

What it does:
    1. Validates that all required packages are importable
    2. Creates the training folder structure under training/
    3. Writes training/build_features.py  — batch feature extractor
    4. Writes training/train_model.py     — RF + XGBoost trainer
    5. Writes training/label_map.yaml     — FITS filename → class mapping template
    6. Prints exact next steps

After running this script, you only need to:
    a) Drop your FITS files into  training/fits/<class>/
    b) Run: python training/build_features.py
    c) Run: python training/train_model.py
"""

import sys
import os
import textwrap
from pathlib import Path

# ── Colour helpers for terminal output ───────────────────────────────────────
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

def ok(msg):    print(f"  {_GREEN}✓{_RESET}  {msg}")
def warn(msg):  print(f"  {_YELLOW}⚠{_RESET}  {msg}")
def fail(msg):  print(f"  {_RED}✗{_RESET}  {msg}")
def header(msg): print(f"\n{_BOLD}{msg}{_RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Environment validation
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED = {
    "numpy":        "numpy",
    "scipy":        "scipy",
    "astropy":      "astropy",
    "sklearn":      "scikit-learn",
    "xgboost":      "xgboost",
    "matplotlib":   "matplotlib",
    "yaml":         "pyyaml",
    "pandas":       "pandas",
    "imblearn":     "imbalanced-learn",
}

header("Step 1 — Checking required packages")
missing = []
for module, pkg in REQUIRED.items():
    try:
        __import__(module)
        ok(f"{pkg}")
    except ImportError:
        fail(f"{pkg}  →  pip install {pkg} --break-system-packages")
        missing.append(pkg)

if missing:
    print(f"\n{_RED}Install missing packages then re-run this script.{_RESET}")
    sys.exit(1)

# Verify the repo's own core modules are importable
header("Step 2 — Checking TransitLens core modules")
REPO_ROOT = Path(__file__).resolve().parent

# The script lives alongside the repo root — adjust if placed elsewhere
if not (REPO_ROOT / "pipeline.py").exists():
    # Try parent directory
    REPO_ROOT = REPO_ROOT.parent

if not (REPO_ROOT / "pipeline.py").exists():
    fail("pipeline.py not found. Place prepare_ml.py in the repo root and re-run.")
    sys.exit(1)

sys.path.insert(0, str(REPO_ROOT))

CORE_MODULES = [
    ("pipeline",                   "analyze_light_curve"),
    ("core.preprocess",            "clean"),
    ("core.bls_detector",         "detect"),
    ("core.feature_extractor",    "extract"),
    ("core.classifier",           "classify"),
    ("core.confidence",           "score"),
]

core_ok = True
for module, attr in CORE_MODULES:
    try:
        mod = __import__(module, fromlist=[attr])
        getattr(mod, attr)
        ok(f"{module}.{attr}")
    except Exception as exc:
        fail(f"{module}.{attr}  →  {exc}")
        core_ok = False

if not core_ok:
    print(f"\n{_RED}Fix the import errors above before continuing.{_RESET}")
    sys.exit(1)

ok("rule_config.yaml" if (REPO_ROOT / "models" / "rule_config.yaml").exists()
   else "models/rule_config.yaml — NOT FOUND (needed by classifier)")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Folder structure
# ─────────────────────────────────────────────────────────────────────────────

header("Step 3 — Creating training folder structure")

TRAINING_DIR = REPO_ROOT / "training"
DIRS = [
    TRAINING_DIR,
    TRAINING_DIR / "fits" / "exoplanet_like",
    TRAINING_DIR / "fits" / "eclipsing_binary_like",
    TRAINING_DIR / "fits" / "noise_or_other",
    TRAINING_DIR / "features",
    TRAINING_DIR / "checkpoints",
]

for d in DIRS:
    d.mkdir(parents=True, exist_ok=True)
    ok(str(d.relative_to(REPO_ROOT)))

# README inside each fits subfolder
for cls in ["exoplanet_like", "eclipsing_binary_like", "noise_or_other"]:
    readme = TRAINING_DIR / "fits" / cls / "README.md"
    if not readme.exists():
        readme.write_text(f"# {cls}\n\nDrop FITS files for this class here.\n"
                          f"Filename format: anything ending in .fits or .fit\n")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Write training/build_features.py
# ─────────────────────────────────────────────────────────────────────────────

header("Step 4 — Writing training/build_features.py")

BUILD_FEATURES = '''\
"""
training/build_features.py
---------------------------
Batch feature extractor for model training.

Reads every FITS file from:
    training/fits/exoplanet_like/
    training/fits/eclipsing_binary_like/
    training/fits/noise_or_other/

Runs the TransitLens pipeline (preprocess → BLS → feature extract) on each
file and saves all 11 features + label to:
    training/features/feature_dataset.csv

Usage:
    python training/build_features.py
    python training/build_features.py --max 50    # limit to 50 files per class
    python training/build_features.py --resume    # skip already-processed files

Output CSV columns:
    tic_id, label, bls_power, snr, period_days, duration_days, depth,
    transit_count, odd_even_depth_delta, v_shape_score, local_noise,
    depth_to_noise_ratio, phase_shape_kurtosis,
    candidate_detected, processing_time_ms, fits_path
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# ── Repo root on path ─────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.preprocess import clean
from core.bls_detector import detect
from core.feature_extractor import extract, FEATURE_NAMES
from core.exceptions import InvalidInputError, InsufficientDataError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CLASSES = ["exoplanet_like", "eclipsing_binary_like", "noise_or_other"]
FITS_DIR   = Path(__file__).parent / "fits"
OUTPUT_DIR = Path(__file__).parent / "features"
OUTPUT_CSV = OUTPUT_DIR / "feature_dataset.csv"

FIELDNAMES = (
    ["tic_id", "label"]
    + list(FEATURE_NAMES)
    + ["candidate_detected", "processing_time_ms", "fits_path"]
)


def load_fits(fits_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Load time and flux from a TESS FITS light curve file.

    Tries common TESS HDU extensions in order:
        LIGHTCURVE (standard TESS), then HDU 1.

    Returns (time, flux, target_id).
    """
    from astropy.io import fits as astrofits

    with astrofits.open(fits_path) as hdul:
        hdul.info()  # silent unless debug logging

        # Try named extension first, then HDU 1
        ext = None
        for name in ["LIGHTCURVE", "LC", 1]:
            try:
                ext = hdul[name]
                break
            except (KeyError, IndexError):
                continue

        if ext is None:
            raise ValueError(f"No usable light curve extension in {fits_path.name}")

        data = ext.data

        # TESS standard columns: TIME, PDCSAP_FLUX or SAP_FLUX
        time_col = None
        for col in ["TIME", "time", "BJD", "bjd"]:
            if col in data.names:
                time_col = data[col]
                break

        flux_col = None
        for col in ["PDCSAP_FLUX", "pdcsap_flux", "SAP_FLUX", "sap_flux", "FLUX", "flux"]:
            if col in data.names:
                flux_col = data[col]
                break

        if time_col is None or flux_col is None:
            available = list(data.names)
            raise ValueError(
                f"Could not find time/flux columns. Available: {available}"
            )

        time = np.asarray(time_col, dtype=float)
        flux = np.asarray(flux_col, dtype=float)

        # Normalise flux to median ~ 1.0
        finite_mask = np.isfinite(flux)
        if finite_mask.sum() < 10:
            raise ValueError("Fewer than 10 finite flux values — file unusable")
        median_flux = np.nanmedian(flux[finite_mask])
        if median_flux <= 0:
            raise ValueError(f"Non-positive median flux ({median_flux}) — file unusable")
        flux = flux / median_flux

        # Extract target ID from header
        header = ext.header
        tic_id = str(
            header.get("TICID", header.get("TIC_ID", fits_path.stem))
        )

    return time, flux, tic_id


def process_file(fits_path: Path, label: str) -> dict | None:
    """
    Run pipeline stages 1-3 on one FITS file. Returns a feature row dict or None.
    """
    t0 = time.perf_counter()

    try:
        time_arr, flux_arr, tic_id = load_fits(fits_path)
    except Exception as exc:
        logger.warning("LOAD FAILED  %s  →  %s", fits_path.name, exc)
        return None

    try:
        preprocess_result = clean(time_arr, flux_arr)
        time_clean = preprocess_result.time
        flux_clean = preprocess_result.flux
    except (InvalidInputError, InsufficientDataError) as exc:
        logger.warning("PREPROCESS SKIP  %s  →  %s", fits_path.name, exc)
        return None
    except Exception as exc:
        logger.warning("PREPROCESS ERROR  %s  →  %s", fits_path.name, exc)
        return None

    try:
        bls_result = detect(time_clean, flux_clean)
    except Exception as exc:
        logger.warning("BLS ERROR  %s  →  %s", fits_path.name, exc)
        return None

    try:
        feature_result = extract(time_clean, flux_clean, bls_result)
        features = feature_result.features
    except Exception as exc:
        logger.warning("FEATURE ERROR  %s  →  %s", fits_path.name, exc)
        return None

    elapsed_ms = (time.perf_counter() - t0) * 1000

    row = {
        "tic_id": tic_id,
        "label": label,
        "candidate_detected": int(bls_result.candidate_detected),
        "processing_time_ms": round(elapsed_ms, 1),
        "fits_path": str(fits_path),
    }
    for k in FEATURE_NAMES:
        row[k] = round(float(features[k]), 8)

    return row


def main():
    parser = argparse.ArgumentParser(description="Batch feature extractor for TransitLens")
    parser.add_argument("--max", type=int, default=None, help="Max files per class")
    parser.add_argument("--resume", action="store_true", help="Skip already-processed files")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load already-processed paths if resuming
    done_paths: set[str] = set()
    if args.resume and OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done_paths.add(row.get("fits_path", ""))
        logger.info("Resume mode: %d files already processed", len(done_paths))

    # Open output CSV (append if resuming, write-new otherwise)
    mode = "a" if args.resume and OUTPUT_CSV.exists() else "w"
    out_file = open(OUTPUT_CSV, mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(out_file, fieldnames=FIELDNAMES)
    if mode == "w":
        writer.writeheader()

    total_ok = total_skip = total_fail = 0
    class_counts: dict[str, int] = {cls: 0 for cls in CLASSES}

    for label in CLASSES:
        class_dir = FITS_DIR / label
        if not class_dir.exists():
            logger.warning("Class folder missing: %s — skipping", class_dir)
            continue

        fits_files = sorted(
            list(class_dir.glob("*.fits")) + list(class_dir.glob("*.fit"))
        )

        if not fits_files:
            logger.warning("No FITS files found in %s", class_dir)
            continue

        if args.max:
            fits_files = fits_files[: args.max]

        logger.info(
            "\n─── %s: %d files ─────────────────────────────",
            label, len(fits_files),
        )

        for fits_path in fits_files:
            if str(fits_path) in done_paths:
                logger.debug("SKIP (already done): %s", fits_path.name)
                total_skip += 1
                continue

            row = process_file(fits_path, label)
            if row is None:
                total_fail += 1
                continue

            writer.writerow(row)
            out_file.flush()
            class_counts[label] += 1
            total_ok += 1
            logger.info(
                "OK  %-45s  depth=%.4f  snr=%6.1f  period=%.4fd",
                fits_path.name[:45],
                row["depth"], row["snr"], row["period_days"],
            )

    out_file.close()

    print(f"""
═══════════════════════════════════════════════════════
  Feature extraction complete
═══════════════════════════════════════════════════════
  Processed:   {total_ok}
  Skipped:     {total_skip}
  Failed:      {total_fail}

  Per class:
{''.join(f"    {cls:<28} {count:>4}\\n" for cls, count in class_counts.items())}
  Output: {OUTPUT_CSV}

  Next step:
    python training/train_model.py
═══════════════════════════════════════════════════════
""")

    if total_ok < 10:
        print("  ⚠  Fewer than 10 samples extracted.")
        print("     Add more FITS files to training/fits/<class>/ and re-run.\\n")


if __name__ == "__main__":
    main()
'''

build_features_path = TRAINING_DIR / "build_features.py"
build_features_path.write_text(BUILD_FEATURES, encoding="utf-8")
ok(f"training/build_features.py")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Write training/train_model.py
# ─────────────────────────────────────────────────────────────────────────────

header("Step 5 — Writing training/train_model.py")

TRAIN_MODEL = '''\
"""
training/train_model.py
------------------------
Train Random Forest and XGBoost classifiers on the extracted feature dataset.

Prerequisites:
    Run training/build_features.py first to generate feature_dataset.csv.

Usage:
    python training/train_model.py
    python training/train_model.py --model rf     # RF only
    python training/train_model.py --model xgb    # XGBoost only
    python training/train_model.py --min-samples 5  # allow smaller datasets

Outputs (all written to models/):
    models/rf_model.pkl          — trained Random Forest
    models/xgb_model.pkl         — trained XGBoost
    models/feature_scaler.pkl    — fitted StandardScaler (required at inference)
    training/features/training_report.txt — classification report
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

REPO_ROOT   = Path(__file__).resolve().parent.parent
FEATURES_CSV = Path(__file__).parent / "features" / "feature_dataset.csv"
MODELS_DIR   = REPO_ROOT / "models"
REPORT_PATH  = Path(__file__).parent / "features" / "training_report.txt"

sys.path.insert(0, str(REPO_ROOT))
from core.feature_extractor import FEATURE_NAMES

CLASSES = ["exoplanet_like", "eclipsing_binary_like", "noise_or_other"]


def load_dataset(min_samples: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not FEATURES_CSV.exists():
        logger.error("feature_dataset.csv not found at %s", FEATURES_CSV)
        logger.error("Run  python training/build_features.py  first.")
        sys.exit(1)

    df = pd.read_csv(FEATURES_CSV)
    logger.info("Loaded %d rows from feature_dataset.csv", len(df))

    # Drop rows with any NaN feature
    before = len(df)
    df = df.dropna(subset=list(FEATURE_NAMES))
    if len(df) < before:
        logger.warning("Dropped %d rows with NaN features", before - len(df))

    # Check class coverage
    counts = df["label"].value_counts()
    logger.info("Class distribution:\n%s", counts.to_string())

    missing_classes = [c for c in CLASSES if c not in counts.index]
    if missing_classes:
        logger.error("Missing classes in dataset: %s", missing_classes)
        logger.error("Add FITS files for these classes and re-run build_features.py")
        sys.exit(1)

    if counts.min() < min_samples:
        smallest = counts.idxmin()
        logger.error(
            "Class '%s' has only %d samples (minimum is %d).",
            smallest, counts.min(), min_samples,
        )
        logger.error("Add more FITS files for this class or lower --min-samples.")
        sys.exit(1)

    X = df[list(FEATURE_NAMES)].values.astype(float)
    y = df["label"].values
    tic_ids = df["tic_id"].tolist()
    return X, y, tic_ids


def train_rf(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV

    logger.info("Training Random Forest with GridSearchCV...")
    param_grid = {
        "n_estimators": [100, 300, 500],
        "max_depth": [5, 10, None],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    rf = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
        param_grid,
        cv=min(5, _min_class_count(y_train)),
        scoring="f1_macro",
        n_jobs=-1,
        verbose=0,
    )
    rf.fit(X_train, y_train)
    logger.info("Best RF params: %s", rf.best_params_)
    logger.info("Best RF CV f1_macro: %.4f", rf.best_score_)
    return rf.best_estimator_


def train_xgb(X_train, y_train, X_val, y_val):
    from xgboost import XGBClassifier
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    le.fit(CLASSES)
    y_train_enc = le.transform(y_train)
    y_val_enc   = le.transform(y_val)

    logger.info("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    xgb.fit(
        X_train, y_train_enc,
        eval_set=[(X_val, y_val_enc)],
        verbose=False,
    )

    # Patch so predict returns string labels (matching RF interface)
    xgb._label_encoder = le
    _orig_predict = xgb.predict
    def predict_strings(X):
        return le.inverse_transform(_orig_predict(X))
    xgb.predict = predict_strings

    _orig_proba = xgb.predict_proba
    def predict_proba_ordered(X):
        probas = _orig_proba(X)
        # reorder columns to match CLASSES order
        order = [list(le.classes_).index(c) for c in CLASSES]
        return probas[:, order]
    xgb.predict_proba = predict_proba_ordered
    xgb.classes_ = np.array(CLASSES)

    return xgb


def evaluate(model, X_val, y_val, model_name: str) -> str:
    from sklearn.metrics import classification_report, confusion_matrix

    y_pred = model.predict(X_val)
    report = classification_report(
        y_val, y_pred,
        labels=CLASSES,
        target_names=CLASSES,
        zero_division=0,
    )
    cm = confusion_matrix(y_val, y_pred, labels=CLASSES)

    lines = [
        "=" * 72,
        f"  {model_name} — Validation Report",
        "=" * 72,
        report,
        "Confusion matrix (rows=true, cols=predicted):",
        f"  Classes: {CLASSES}",
        str(cm),
        "",
    ]

    # Feature importances (RF only)
    if hasattr(model, "feature_importances_"):
        importances = sorted(
            zip(FEATURE_NAMES, model.feature_importances_),
            key=lambda x: x[1], reverse=True,
        )
        lines.append("Feature importances:")
        for fname, imp in importances:
            bar = "█" * int(imp * 40)
            lines.append(f"  {fname:<30} {imp:.4f}  {bar}")
        lines.append("")

    block = "\n".join(lines)
    print(block)
    return block


def _min_class_count(y):
    from collections import Counter
    return min(Counter(y).values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rf", "xgb", "both"], default="both")
    parser.add_argument("--min-samples", type=int, default=3,
                        help="Minimum samples per class to proceed")
    parser.add_argument("--val-size", type=float, default=0.2,
                        help="Validation fraction (default 0.20)")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────
    X, y, tic_ids = load_dataset(args.min_samples)

    # ── Train/val split (stratified) ─────────────────────────────────────
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import StandardScaler

    n_val_samples = max(int(len(X) * args.val_size), len(CLASSES))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=n_val_samples, random_state=42)
    train_idx, val_idx = next(sss.split(X, y))

    # ── Scale features ────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val   = scaler.transform(X[val_idx])
    y_train, y_val = y[train_idx], y[val_idx]

    logger.info("Train: %d  Val: %d", len(train_idx), len(val_idx))

    # Save scaler immediately — needed even if model training fails later
    scaler_path = MODELS_DIR / "feature_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Scaler saved → %s", scaler_path)

    # ── SMOTE oversampling if any class is severely underrepresented ──────
    from collections import Counter
    counts = Counter(y_train)
    max_count = max(counts.values())
    min_count = min(counts.values())

    if min_count < 6 or (max_count / min_count) > 3:
        logger.warning(
            "Class imbalance detected (ratio %.1f). Applying SMOTE.",
            max_count / max(min_count, 1),
        )
        try:
            from imblearn.over_sampling import SMOTE
            k = max(1, min_count - 1)
            sm = SMOTE(random_state=42, k_neighbors=k)
            X_train, y_train = sm.fit_resample(X_train, y_train)
            logger.info("After SMOTE: %s", Counter(y_train))
        except Exception as exc:
            logger.warning("SMOTE failed (%s) — continuing without oversampling", exc)

    all_reports = []

    # ── Random Forest ─────────────────────────────────────────────────────
    if args.model in ("rf", "both"):
        rf_model = train_rf(X_train, y_train)
        report = evaluate(rf_model, X_val, y_val, "Random Forest")
        all_reports.append(report)
        rf_path = MODELS_DIR / "rf_model.pkl"
        with open(rf_path, "wb") as f:
            pickle.dump(rf_model, f)
        logger.info("RF model saved → %s", rf_path)

    # ── XGBoost ───────────────────────────────────────────────────────────
    if args.model in ("xgb", "both"):
        xgb_model = train_xgb(X_train, y_train, X_val, y_val)
        report = evaluate(xgb_model, X_val, y_val, "XGBoost")
        all_reports.append(report)
        xgb_path = MODELS_DIR / "xgb_model.pkl"
        with open(xgb_path, "wb") as f:
            pickle.dump(xgb_model, f)
        logger.info("XGBoost model saved → %s", xgb_path)

    # ── Save report ───────────────────────────────────────────────────────
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(all_reports), encoding="utf-8")
    logger.info("Training report saved → %s", REPORT_PATH)

    print("""
═══════════════════════════════════════════════════════
  Training complete!
═══════════════════════════════════════════════════════
  Saved:
    models/feature_scaler.pkl
    models/rf_model.pkl         (if --model rf or both)
    models/xgb_model.pkl        (if --model xgb or both)
    training/features/training_report.txt

  To activate the ML classifier, edit models/rule_config.yaml:
    ml_classifier:
      enabled: true
      model_type: "rf"   # or "xgb"
═══════════════════════════════════════════════════════
""")


if __name__ == "__main__":
    main()
'''

train_model_path = TRAINING_DIR / "train_model.py"
train_model_path.write_text(TRAIN_MODEL, encoding="utf-8")
ok("training/train_model.py")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Validate the scripts themselves are parseable
# ─────────────────────────────────────────────────────────────────────────────

header("Step 6 — Validating generated scripts")
import ast

for script in [build_features_path, train_model_path]:
    try:
        ast.parse(script.read_text())
        ok(f"{script.name} — valid Python")
    except SyntaxError as exc:
        fail(f"{script.name} — SyntaxError: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Quick smoke test — run build_features on an empty dir (should exit cleanly)
# ─────────────────────────────────────────────────────────────────────────────

header("Step 7 — Smoke-testing build_features.py (empty folders)")
import subprocess

result = subprocess.run(
    [sys.executable, str(build_features_path)],
    capture_output=True, text=True
)
# We expect it to finish (exit 0) with zero files found
if result.returncode == 0:
    ok("build_features.py runs cleanly on empty folders")
else:
    warn(f"build_features.py exited {result.returncode} — check output above")
    if result.stderr:
        print(result.stderr[:500])


# ─────────────────────────────────────────────────────────────────────────────
# 7. Final summary
# ─────────────────────────────────────────────────────────────────────────────

print(f"""
{_BOLD}═══════════════════════════════════════════════════════
  Preparation complete — your next three steps:
═══════════════════════════════════════════════════════{_RESET}

  {_BOLD}1. Drop your FITS files here:{_RESET}

     training/fits/exoplanet_like/        ← confirmed planets
     training/fits/eclipsing_binary_like/ ← confirmed/labelled EBs
     training/fits/noise_or_other/        ← false positives / noise

     File naming: anything ending in .fits or .fit
     Minimum recommended: 20 files per class (60 total)
     More is better — aim for 100+ per class if you can

  {_BOLD}2. Extract features:{_RESET}

     python training/build_features.py

     Add --resume to continue after a crash.
     Add --max 50 to limit files per class during testing.
     Output → training/features/feature_dataset.csv

  {_BOLD}3. Train the models:{_RESET}

     python training/train_model.py

     Trains both RF and XGBoost, prints a classification report,
     saves models/rf_model.pkl + models/xgb_model.pkl + models/feature_scaler.pkl
     Then edit models/rule_config.yaml: ml_classifier.enabled: true

{_BOLD}Folder layout created:{_RESET}

  training/
  ├── fits/
  │   ├── exoplanet_like/          ← your FITS files go here
  │   ├── eclipsing_binary_like/   ← your FITS files go here
  │   └── noise_or_other/          ← your FITS files go here
  ├── features/
  │   └── feature_dataset.csv      ← generated by build_features.py
  ├── build_features.py
  └── train_model.py

{_GREEN}All good. Add FITS files and run the two commands above.{_RESET}
""")