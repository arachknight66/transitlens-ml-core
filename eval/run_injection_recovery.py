"""
eval/run_injection_recovery.py
------------------------------
Phase 4 Injection-Recovery Suite CLI entry point for TransitLens.

Usage examples:
    # From transitlens-ml-core directory:
    python -m eval.run_injection_recovery --mode quick
    python -m eval.run_injection_recovery --mode standard
    python -m eval.run_injection_recovery --mode full

    # With custom options:
    python -m eval.run_injection_recovery --mode standard --seed 123 --output-dir /tmp/ir_results
    python -m eval.run_injection_recovery --mode quick --no-plots --max-trials 50
    python -m eval.run_injection_recovery --mode full --config eval/injection_config.yaml

Exit codes:
    0 — Suite completed successfully (even if some targets failed)
    1 — Fatal error (e.g., config not found, disk full)

Outputs (written to --output-dir, default: eval/results/):
    injection_recovery_trials.csv            — per-trial detailed results
    injection_recovery_summary.csv           — overall aggregate metrics
    injection_recovery_by_snr.csv            — metrics binned by SNR
    injection_recovery_by_depth.csv          — metrics binned by depth
    injection_recovery_by_period.csv         — metrics binned by period
    injection_recovery_by_noise.csv          — metrics binned by noise RMS
    injection_recovery_heatmap.csv           — period × depth × noise heatmap
    false_positive_controls.csv              — per-trial control results
    false_positive_summary.csv               — FP summary by control type
    alias_recovery_summary.csv               — alias/harmonic statistics
    confidence_calibration_injection.csv     — confidence binned calibration
    phase4_injection_recovery_report.md      — full markdown report
    plots/
        snr_recall_curve.png
        depth_recall_curve.png
        period_recovery_heatmap.png
        parameter_error_vs_snr.png
        false_positive_controls.png
        confidence_reliability.png
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time as _time
from pathlib import Path

import numpy as np

# ── Ensure repo root on sys.path ─────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.injection_recovery import run_injection_recovery_suite

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_injection_recovery")


VALID_MODES = ("quick", "standard", "full")

# Mode descriptions for help text
MODE_DESCRIPTIONS = {
    "quick": "~75 injections + 25 controls (~100 total). Development check only.",
    "standard": "~500 injections + 100 controls (~600 total). Minimum judge-grade evidence.",
    "full": "~2000 injections + 200 controls (~2200 total). Strong evidence benchmark.",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval.run_injection_recovery",
        description=(
            "TransitLens Phase 4 — Injection-Recovery Benchmark Suite.\n\n"
            "Runs synthetic transit injection trials and control trials\n"
            "through the real TransitLens pipeline to measure:\n"
            "  - Detection recall by SNR, depth, period, noise\n"
            "  - Period recovery accuracy (1% and 5% tolerance)\n"
            "  - Alias (P/2, 2P) rates\n"
            "  - False-positive rates on noise/variability controls\n"
            "  - Confidence score calibration\n\n"
            "Results are written to CSV files and a markdown report.\n"
            "Plots are generated if matplotlib is available.\n\n"
            "EVIDENCE LEVEL: This is synthetic injection evidence only.\n"
            "Do not conflate with real TESS sector screening (Level 4)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default="quick",
        help=(
            "Run mode. "
            + " | ".join(f"{m}: {MODE_DESCRIPTIONS[m]}" for m in VALID_MODES)
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help=(
            "Path to injection_config.yaml. "
            "Defaults to eval/injection_config.yaml relative to repo root."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help=(
            "Output directory for CSVs and plots. "
            "Defaults to eval/results/ relative to repo root."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="INT",
        help="Random seed override (default: from config, usually 42).",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        metavar="N",
        help="Hard cap on injection trials (does not include control trials).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation even if matplotlib is available.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


def print_summary(summary: dict, mode: str) -> None:
    """Print a concise human-readable summary to stdout."""
    sep = "=" * 68
    print(f"\n{sep}")
    print("  TransitLens Phase 4 -- Injection-Recovery Summary")
    print(f"{sep}")
    print(f"  Mode:           {mode}")
    print(f"  Total trials:   {summary.get('n_trials', '?')}")
    print(f"  Injections:     {summary.get('n_injected', '?')}")
    print(f"  Controls:       {summary.get('n_controls', '?')}")
    print(f"  Failures (inj): {summary.get('n_injection_failures', 0)}")
    print(f"  Failures (ctrl):{summary.get('n_control_failures', 0)}")
    print(f"\n  --- Overall Metrics ---")

    def _p(val):
        if val is None:
            return "  N/A"
        try:
            if np.isnan(val):
                return "  N/A"
            return f" {val * 100:5.1f}%"
        except Exception:
            return "  N/A"

    def _f(val, fmt=".3f"):
        if val is None:
            return "  N/A"
        try:
            if np.isnan(val):
                return "  N/A"
            return f" {val:{fmt}}"
        except Exception:
            return "  N/A"

    print(f"  Detection recall (all):      {_p(summary.get('detection_recall'))}")
    print(f"  Detection recall (SNR>=7):   {_p(summary.get('detection_recall_high_snr'))}")
    print(f"  Period recovery +-1% (all):  {_p(summary.get('period_recovery_rate_1pct'))}")
    print(f"  Period recovery +-1% (>=7):  {_p(summary.get('period_recovery_1pct_high_snr'))}")
    print(f"  Period recovery +-5% (all):  {_p(summary.get('period_recovery_rate_5pct'))}")
    print(f"  Period recovery +-5% (>=7):  {_p(summary.get('period_recovery_5pct_high_snr'))}")
    print(f"  FP rate (controls):          {_p(summary.get('false_positive_rate_controls'))}")
    print(f"  Median period error:         {_f(summary.get('median_period_error_pct'))} %")
    print(f"  Median depth error:          {_f(summary.get('median_depth_error_pct'))} %")
    print(f"  Median duration error:       {_f(summary.get('median_duration_error_pct'))} %")
    print(f"  Half-period alias rate:      {_p(summary.get('half_period_alias_rate'))}")
    print(f"  Double-period alias rate:    {_p(summary.get('double_period_alias_rate'))}")
    print(f"  Mean runtime/trial:          {_f(summary.get('mean_runtime_ms'), '.1f')} ms")
    print(f"\n  --- Strict Targets (SNR >= 7) ---")

    def _check(val, thresh, ge=True):
        if val is None:
            return "?? N/A"
        try:
            if np.isnan(val):
                return "?? N/A"
            passed = val >= thresh if ge else val < thresh
            return f"{'[PASS]' if passed else '[FAIL]'} ({val * 100:.1f}%)"
        except Exception:
            return "?? N/A"

    def _check_raw(val, thresh, ge=False):
        if val is None:
            return "?? N/A"
        try:
            if np.isnan(val):
                return "?? N/A"
            passed = val < thresh if not ge else val >= thresh
            return f"{'[PASS]' if passed else '[FAIL]'} ({val * 100:.1f}%)"
        except Exception:
            return "?? N/A"

    print(f"  Recall >= 90%:   {_check(summary.get('detection_recall_high_snr'), 0.90)}")
    print(f"  PRR+-1% >= 90%:  {_check(summary.get('period_recovery_1pct_high_snr'), 0.90)}")
    print(f"  PRR+-5% >= 95%:  {_check(summary.get('period_recovery_5pct_high_snr'), 0.95)}")
    print(f"  FP < 15%:        {_check_raw(summary.get('false_positive_rate_controls'), 0.15)}")
    print(f"\n{sep}\n")


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("TransitLens Phase 4 -- Injection-Recovery Suite")
    logger.info("Mode: %s -- %s", args.mode, MODE_DESCRIPTIONS[args.mode])

    # Resolve config path
    cfg_path = args.config
    if cfg_path is None:
        default_cfg = _REPO_ROOT / "eval" / "injection_config.yaml"
        if default_cfg.exists():
            cfg_path = str(default_cfg)
        else:
            logger.error(
                "injection_config.yaml not found at %s. "
                "Pass --config PATH or run from transitlens-ml-core/.",
                default_cfg
            )
            return 1

    # Resolve output dir
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(_REPO_ROOT / "eval" / "results")

    t_start = _time.perf_counter()

    # Patch: disable plots if --no-plots requested
    if args.no_plots:
        # Monkey-patch generate_plots to a no-op
        import eval.injection_recovery as _ir
        _ir.generate_plots = lambda df, plots_path, cfg: None
        logger.info("Plot generation disabled (--no-plots)")

    try:
        summary = run_injection_recovery_suite(
            mode=args.mode,
            cfg_path=cfg_path,
            output_dir=output_dir,
            seed=args.seed,
            max_trials=args.max_trials,
        )
    except FileNotFoundError as exc:
        logger.error("Config or output path error: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Injection-recovery suite crashed: %s", exc)
        return 1

    elapsed = _time.perf_counter() - t_start
    logger.info("Total elapsed time: %.1f seconds", elapsed)

    print_summary(summary, args.mode)

    out_path = Path(output_dir)
    report_path = out_path / "phase4_injection_recovery_report.md"
    trials_path = out_path / "injection_recovery_trials.csv"

    print(f"  Report:        {report_path}")
    print(f"  Trials CSV:    {trials_path}")
    print(f"  All results:   {out_path}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
