"""Update the 3Q W_CH1 15x5 global amplitude matrix from signal counts."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .wch1_excitation_factor import build_wch1_excitation_factor_matrix
    from .wch1_global_amp import (
        DEFAULT_W_CH1_GLOBAL_AMP_SHAPE,
        load_w_ch1_global_amp_matrix,
        save_w_ch1_global_amp_matrix,
    )
except ImportError:
    from wch1_excitation_factor import build_wch1_excitation_factor_matrix
    from wch1_global_amp import (
        DEFAULT_W_CH1_GLOBAL_AMP_SHAPE,
        load_w_ch1_global_amp_matrix,
        save_w_ch1_global_amp_matrix,
    )


GROUP_ROWS, GROUP_COLS = DEFAULT_W_CH1_GLOBAL_AMP_SHAPE
GROUP_WIDTH = GROUP_ROWS // GROUP_COLS


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_sample_dir() -> Path:
    return repo_root() / "ml_recon" / "data" / "feedback_samples"


def default_matrix_path() -> Path:
    return Path(__file__).resolve().parent / "matrix" / "W_CH1_global_amp.txt"


def find_latest_feedback_sample(samples_dir: str | Path, system_type: str = "3qubit") -> Path:
    samples_dir = Path(samples_dir)
    candidates = sorted(samples_dir.glob(f"{system_type}_*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No {system_type}_*.json samples found in {samples_dir}")
    return candidates[-1]


def load_signal_bins(sample_json: str | Path) -> np.ndarray:
    path = Path(sample_json)
    data = json.loads(path.read_text(encoding="utf-8"))
    signal_bins = data.get("counts_snapshot", {}).get("signal_bins")
    if signal_bins is None:
        raise ValueError(f"{path} does not contain counts_snapshot.signal_bins")
    signal = np.asarray(signal_bins, dtype=float)
    if signal.size != GROUP_ROWS * GROUP_COLS:
        raise ValueError(f"Expected 75 signal bins, got {signal.size} from {path}")
    return signal.reshape(DEFAULT_W_CH1_GLOBAL_AMP_SHAPE)


def effective_wch1_factor_by_group() -> np.ndarray:
    """Return the RMS W_CH1 excitation factor for each 15x5 UI group."""

    factors = build_wch1_excitation_factor_matrix(GROUP_ROWS)
    grouped = np.zeros(DEFAULT_W_CH1_GLOBAL_AMP_SHAPE, dtype=float)
    for row in range(GROUP_ROWS):
        for group in range(GROUP_COLS):
            start = group * GROUP_WIDTH
            stop = start + GROUP_WIDTH
            grouped[row, group] = float(np.sqrt(np.mean(factors[row, start:stop] ** 2)))
    return grouped


def matrix_stats(matrix: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(matrix, dtype=float)
    mean = float(np.mean(matrix))
    std = float(np.std(matrix))
    return {
        "min": float(np.min(matrix)),
        "max": float(np.max(matrix)),
        "mean": mean,
        "median": float(np.median(matrix)),
        "std": std,
        "cv": float(std / mean) if mean else 0.0,
        "sum": float(np.sum(matrix)),
    }


def choose_target_count(signal: np.ndarray, method: str = "median") -> float:
    positive = np.asarray(signal, dtype=float)
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if positive.size == 0:
        raise ValueError("Signal bins do not contain any positive finite counts")

    if method == "median":
        return float(np.median(positive))
    if method == "trimmed-mean":
        lo, hi = np.percentile(positive, [10.0, 90.0])
        trimmed = positive[(positive >= lo) & (positive <= hi)]
        return float(np.mean(trimmed))
    raise ValueError(f"Unsupported target method: {method}")


def compute_wch1_global_amp_update(
    signal: np.ndarray,
    current_global_amp: np.ndarray,
    *,
    target_count: float | None = None,
    target_method: str = "median",
    min_correction: float = 0.5,
    max_correction: float = 3.0,
    min_amp: float = 0.25,
    max_amp: float = 3.0,
) -> dict[str, np.ndarray | float | dict[str, float]]:
    """Compute a bounded one-step W_CH1 global amplitude update.

    The correction assumes the local signal approximately scales as amplitude
    squared.  The built-in W_CH1 factor is kept unchanged and reported through
    the effective-amplitude matrices so the 15x5 UI values can be interpreted
    in the actual waveform context.
    """

    signal = np.asarray(signal, dtype=float).reshape(DEFAULT_W_CH1_GLOBAL_AMP_SHAPE)
    current = np.asarray(current_global_amp, dtype=float).reshape(DEFAULT_W_CH1_GLOBAL_AMP_SHAPE)
    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal matrix contains non-finite values")
    if not np.all(np.isfinite(current)):
        raise ValueError("Current W_CH1 global amplitude matrix contains non-finite values")
    if min_correction <= 0.0 or max_correction < min_correction:
        raise ValueError("Correction bounds must satisfy 0 < min <= max")
    if min_amp < 0.0 or max_amp < min_amp:
        raise ValueError("Amplitude bounds must satisfy 0 <= min <= max")

    target = float(target_count) if target_count is not None else choose_target_count(signal, target_method)
    signal_safe = np.where(signal > 0.0, signal, 1.0)
    signal_safe = np.maximum(signal_safe, 1.0)
    current_safe = np.where(current > 0.0, current, 1.0)

    ideal_correction = np.sqrt(target / signal_safe)
    bounded_correction = np.clip(ideal_correction, min_correction, max_correction)
    updated = np.clip(current_safe * bounded_correction, min_amp, max_amp)
    applied_correction = updated / current_safe
    predicted_signal = signal * applied_correction**2

    factor_by_group = effective_wch1_factor_by_group()
    current_effective_amp = current * factor_by_group
    updated_effective_amp = updated * factor_by_group

    return {
        "signal": signal,
        "target_count": target,
        "current_global_amp": current,
        "ideal_correction": ideal_correction,
        "bounded_correction": bounded_correction,
        "applied_correction": applied_correction,
        "updated_global_amp": updated,
        "wch1_factor_by_group": factor_by_group,
        "current_effective_amp": current_effective_amp,
        "updated_effective_amp": updated_effective_amp,
        "predicted_signal_after": predicted_signal,
        "signal_stats_before": matrix_stats(signal),
        "predicted_signal_stats_after": matrix_stats(predicted_signal),
        "current_effective_amp_stats": matrix_stats(current_effective_amp),
        "updated_effective_amp_stats": matrix_stats(updated_effective_amp),
    }


def report_json_dict(
    result: dict[str, Any],
    *,
    sample_path: Path,
    matrix_path: Path,
    backup_path: Path | None,
    dry_run: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    def value_for_json(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return np.round(value.astype(float), 6).tolist()
        if isinstance(value, np.floating):
            return float(value)
        return value

    return {
        "source_sample": str(sample_path),
        "matrix_path": str(matrix_path),
        "backup_path": str(backup_path) if backup_path else None,
        "dry_run": dry_run,
        "assumptions": {
            "signal_scales_with_amplitude_squared": True,
            "target_count_method": args.target_method,
            "wch1_factor_group_method": "rms_over_three_factor_columns",
        },
        "bounds": {
            "min_correction": args.min_correction,
            "max_correction": args.max_correction,
            "min_amp": args.min_amp,
            "max_amp": args.max_amp,
        },
        "result": {key: value_for_json(value) for key, value in result.items()},
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten 3Q signal counts by updating matrix/W_CH1_global_amp.txt."
    )
    parser.add_argument("--sample-json", type=Path, help="Feedback sample JSON. Defaults to newest 3qubit_*.json.")
    parser.add_argument("--samples-dir", type=Path, default=default_sample_dir())
    parser.add_argument("--matrix-path", type=Path, default=default_matrix_path())
    parser.add_argument("--report-dir", type=Path, default=default_matrix_path().parent / "calibration_reports")
    parser.add_argument("--target-count", type=float, default=None)
    parser.add_argument("--target-method", choices=("median", "trimmed-mean"), default="median")
    parser.add_argument("--min-correction", type=float, default=0.5)
    parser.add_argument("--max-correction", type=float, default=3.0)
    parser.add_argument("--min-amp", type=float, default=0.25)
    parser.add_argument("--max-amp", type=float, default=3.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sample_path = args.sample_json or find_latest_feedback_sample(args.samples_dir)
    signal = load_signal_bins(sample_path)
    current = load_w_ch1_global_amp_matrix(args.matrix_path)
    result = compute_wch1_global_amp_update(
        signal,
        current,
        target_count=args.target_count,
        target_method=args.target_method,
        min_correction=args.min_correction,
        max_correction=args.max_correction,
        min_amp=args.min_amp,
        max_amp=args.max_amp,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = None
    if not args.dry_run:
        if not args.no_backup:
            backup_path = args.matrix_path.with_name(
                f"{args.matrix_path.stem}_{timestamp}_before_signal_flatten{args.matrix_path.suffix}"
            )
            args.matrix_path.parent.mkdir(parents=True, exist_ok=True)
            if args.matrix_path.exists():
                shutil.copy2(args.matrix_path, backup_path)
            else:
                save_w_ch1_global_amp_matrix(backup_path, current)
        save_w_ch1_global_amp_matrix(args.matrix_path, result["updated_global_amp"])

    args.report_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report_dir / f"wch1_signal_flatten_{timestamp}.json"
    report = report_json_dict(
        result,
        sample_path=sample_path,
        matrix_path=args.matrix_path,
        backup_path=backup_path,
        dry_run=args.dry_run,
        args=args,
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    before = result["signal_stats_before"]
    after = result["predicted_signal_stats_after"]
    print(f"sample={sample_path}")
    print(f"matrix={args.matrix_path}")
    print(f"backup={backup_path}")
    print(f"report={report_path}")
    print(f"target_count={result['target_count']:.6g}")
    print(f"signal_cv_before={before['cv']:.6f}")
    print(f"predicted_signal_cv_after={after['cv']:.6f}")
    print(
        "updated_global_amp_range="
        f"{float(np.min(result['updated_global_amp'])):.6g}.."
        f"{float(np.max(result['updated_global_amp'])):.6g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
