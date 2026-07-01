"""Periodic labeled TESSCut ingestion and baseline model training."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io.fits import HDUList
from astroquery.mast import Tesscut
from scipy.ndimage import median_filter
from torch.utils.data import DataLoader

from transitlens_ml_core.config import load_config
from transitlens_ml_core.datasets import LightCurveDataset, split_dataset
from transitlens_ml_core.evaluation.validation import evaluate_model
from transitlens_ml_core.export import export_pytorch_checkpoint
from transitlens_ml_core.models import BaselineCNN
from transitlens_ml_core.training import Trainer

LOGGER = logging.getLogger("transitlens.tesscut")
Tesscut.TIMEOUT = 60
POSITIVE_DISPOSITIONS = frozenset({"CP", "KP"})
NEGATIVE_DISPOSITIONS = frozenset({"FP"})


@dataclass(frozen=True, slots=True)
class Target:
    """One archive target with a defensible binary label."""

    tic_id: str
    ra: float
    dec: float
    label: int


def read_targets(path: Path) -> list[Target]:
    """Read confirmed planets and false positives from a TOI archive CSV."""
    targets: dict[str, Target] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(line for line in stream if not line.startswith("#"))
        for row in rows:
            disposition = (row.get("tfopwg_disp") or "").strip().upper()
            if disposition in POSITIVE_DISPOSITIONS:
                label = 1
            elif disposition in NEGATIVE_DISPOSITIONS:
                label = 0
            else:
                continue
            try:
                target = Target(
                    tic_id=str(int(float(row["tid"]))),
                    ra=float(row["ra"]),
                    dec=float(row["dec"]),
                    label=label,
                )
            except (KeyError, TypeError, ValueError):
                continue
            targets[target.tic_id] = target
    return sorted(targets.values(), key=lambda item: item.tic_id)


def extract_light_curve(
    cutout: HDUList, sample_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a TESS target-pixel cube into a fixed-length aperture light curve."""
    table = cutout[1].data
    time_values = np.asarray(table["TIME"], dtype=np.float64)
    cube = np.asarray(table["FLUX"], dtype=np.float64)
    quality = np.asarray(table["QUALITY"], dtype=np.int64)
    median_image = np.nanmedian(cube, axis=0)
    background = np.nanmedian(median_image)
    scatter = np.nanmedian(np.abs(median_image - background))
    aperture = median_image > background + 3.0 * max(scatter, np.finfo(float).eps)
    if not np.any(aperture):
        aperture.flat[int(np.nanargmax(median_image))] = True
    flux = np.nansum(cube[:, aperture], axis=1)
    valid = np.isfinite(time_values) & np.isfinite(flux) & (quality == 0)
    time_values, flux = time_values[valid], flux[valid]
    if len(time_values) < 32:
        raise ValueError("TESSCut contains fewer than 32 valid cadences")
    order = np.argsort(time_values)
    time_values, flux = time_values[order], flux[order]
    unique = np.concatenate(([True], np.diff(time_values) > 0))
    time_values, flux = time_values[unique], flux[unique]
    baseline = np.nanmedian(flux)
    if not np.isfinite(baseline) or baseline == 0:
        raise ValueError("TESSCut aperture flux has no finite baseline")
    normalized = flux / baseline
    trend = median_filter(
        normalized, size=min(101, len(normalized) // 2 * 2 + 1), mode="nearest"
    )
    detrended = normalized / np.where(np.abs(trend) > 1e-12, trend, 1.0)
    grid = np.linspace(time_values[0], time_values[-1], sample_length)
    return grid, np.interp(grid, time_values, detrended)


def save_sample(
    path: Path, target: Target, sector: int, time_axis: np.ndarray, flux: np.ndarray
) -> None:
    """Write an artifact compatible with ``LightCurveDataset``."""
    metadata = {
        "source": "TESSCut",
        "tic_id": target.tic_id,
        "sector": sector,
        "label": target.label,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        time=time_axis,
        normalized_flux=flux,
        wavelet_flux=flux,
        features_json=np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8),
    )


def collect(
    targets: list[Target], data_dir: Path, limit: int, sample_length: int
) -> int:
    """Fetch at most ``limit`` unseen target-sector samples sequentially."""
    data_dir.mkdir(parents=True, exist_ok=True)
    cursor_path = data_dir / "collector-cursor.txt"
    try:
        cursor = int(cursor_path.read_text(encoding="ascii")) % len(targets)
    except (FileNotFoundError, ValueError):
        cursor = 0
    added = 0
    attempts = min(len(targets), max(10, limit * 5))
    for offset in range(attempts):
        if added >= limit:
            break
        index = (cursor + offset) % len(targets)
        target = targets[index]
        cursor_path.write_text(str((index + 1) % len(targets)), encoding="ascii")
        LOGGER.info("querying TIC %s", target.tic_id)
        try:
            coordinates = SkyCoord(target.ra, target.dec, unit=(u.deg, u.deg))
            sectors = Tesscut.get_sectors(coordinates=coordinates)
            for sector_value in sectors["sector"]:
                sector = int(sector_value)
                destination = (
                    data_dir / f"tic-{target.tic_id}-s{sector:04d}-y{target.label}.npz"
                )
                if destination.exists():
                    continue
                cutouts = Tesscut.get_cutouts(
                    coordinates=coordinates, sector=sector, size=7
                )
                if not cutouts:
                    continue
                time_axis, flux = extract_light_curve(cutouts[0], sample_length)
                save_sample(destination, target, sector, time_axis, flux)
                added += 1
                LOGGER.info(
                    "stored TIC %s sector %s label %s",
                    target.tic_id,
                    sector,
                    target.label,
                )
                break
        except Exception:  # One unavailable archive target must not stop the worker.
            LOGGER.exception("failed to collect TIC %s", target.tic_id)
    return added


def train(data_dir: Path, config_path: Path, minimum_per_class: int) -> Path | None:
    """Train and publish a checkpoint when both labels have enough samples."""
    paths = sorted(data_dir.glob("*.npz"))
    labels = [int(path.stem.rsplit("-y", 1)[1]) for path in paths]
    if min(labels.count(0), labels.count(1)) < minimum_per_class:
        LOGGER.info(
            "waiting for %s samples in each class; counts=%s/%s",
            minimum_per_class,
            labels.count(0),
            labels.count(1),
        )
        return None
    config = load_config(config_path)
    dataset = LightCurveDataset.from_files(paths, labels, config.data.model_input_field)
    splits = split_dataset(
        dataset,
        config.data.train_fraction,
        config.data.validation_fraction,
        config.data.test_fraction,
        config.project.seed,
    )
    train_loader = DataLoader(
        splits.train, batch_size=config.training.batch_size, shuffle=True
    )
    validation_loader = DataLoader(
        splits.validation, batch_size=config.training.batch_size
    )
    test_loader = DataLoader(splits.test, batch_size=config.training.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineCNN.from_config(config.model)
    Trainer(model, config.training, device, config.project.seed).fit(
        train_loader, validation_loader
    )
    evaluate_model(model, test_loader, device, config.evaluation)
    checkpoint = export_pytorch_checkpoint(model.eval(), config)
    LOGGER.info("published model checkpoint %s", checkpoint)
    return checkpoint


def run(args: argparse.Namespace) -> None:
    """Run collection and training cycles until interrupted."""
    targets = read_targets(args.catalog)
    if not targets:
        raise ValueError("catalog contains no confirmed planets or false positives")
    while True:
        added = collect(targets, args.data_dir, args.max_per_cycle, args.sample_length)
        if added:
            train(args.data_dir, args.config, args.minimum_per_class)
        if args.once:
            return
        time.sleep(args.interval_seconds)


def main() -> None:
    """Parse command-line arguments and start the worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/prototype.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/tesscut"))
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--max-per-cycle", type=int, default=2)
    parser.add_argument("--minimum-per-class", type=int, default=20)
    parser.add_argument("--sample-length", type=int, default=1024)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if (
        args.interval_seconds < 60
        or args.max_per_cycle < 1
        or args.minimum_per_class < 3
    ):
        parser.error(
            "interval must be >=60, max-per-cycle >=1, and minimum-per-class >=3"
        )
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run(args)


if __name__ == "__main__":
    main()
