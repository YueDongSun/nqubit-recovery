"""Flatten 4Q signal-cell excitation by updating four 16x4 W_CH1 matrices."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from .calibrate_wch1_global_amp_from_signal import choose_target_count, matrix_stats
except ImportError:
    from calibrate_wch1_global_amp_from_signal import choose_target_count, matrix_stats


SHAPE_4Q = (4, 16, 4)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def matrix_paths(matrix_dir: Path) -> list[Path]:
    return [matrix_dir / f"W_CH1_global_amp_{cycle}_4qubit.txt" for cycle in range(1, 5)]


def find_latest_feedback_sample(samples_dir: Path) -> Path:
    candidates = sorted(samples_dir.glob("4qubit_*.json"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No 4qubit_*.json samples found in {samples_dir}")
    return candidates[-1]


def load_signal_bins(sample_json: Path) -> np.ndarray:
    payload = json.loads(sample_json.read_text(encoding="utf-8"))
    signal = payload.get("counts_snapshot", {}).get("signal_bins")
    if signal is None:
        raise ValueError(f"{sample_json} does not contain counts_snapshot.signal_bins")
    values = np.asarray(signal, dtype=float)
    if values.size != 256:
        raise ValueError(f"Expected 256 signal bins, got {values.size}")
    return values.reshape(SHAPE_4Q)


def load_current(paths: list[Path]) -> np.ndarray:
    matrices = []
    for path in paths:
        matrix = np.loadtxt(path, dtype=float) if path.exists() else np.ones((16, 4), dtype=float)
        if matrix.shape != (16, 4):
            raise ValueError(f"{path} must have shape 16x4, got {matrix.shape}")
        matrices.append(matrix)
    return np.asarray(matrices, dtype=float)


def compute_update(
    signal: np.ndarray,
    current: np.ndarray,
    *,
    target_count: float | None = None,
    target_method: str = "median",
    min_correction: float = 0.5,
    max_correction: float = 3.0,
    min_amp: float = 0.25,
    max_amp: float = 3.0,
) -> dict:
    signal = np.asarray(signal, dtype=float).reshape(SHAPE_4Q)
    current = np.asarray(current, dtype=float).reshape(SHAPE_4Q)
    if not np.all(np.isfinite(signal)) or not np.all(np.isfinite(current)):
        raise ValueError("Signal and amplitude matrices must be finite")
    target = float(target_count) if target_count is not None else choose_target_count(signal, target_method)
    safe_signal = np.maximum(np.where(signal > 0.0, signal, 1.0), 1.0)
    safe_current = np.where(current > 0.0, current, 1.0)
    ideal = np.sqrt(target / safe_signal)
    bounded = np.clip(ideal, min_correction, max_correction)
    updated = np.clip(safe_current * bounded, min_amp, max_amp)
    applied = updated / safe_current
    predicted = signal * applied**2
    return {
        "signal": signal,
        "target_count": target,
        "current_global_amp": current,
        "ideal_correction": ideal,
        "bounded_correction": bounded,
        "applied_correction": applied,
        "updated_global_amp": updated,
        "predicted_signal_after": predicted,
        "signal_stats_before": matrix_stats(signal),
        "predicted_signal_stats_after": matrix_stats(predicted),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Flatten 4Q signal counts into four 16x4 UI matrices.")
    parser.add_argument("--sample-json", type=Path)
    parser.add_argument("--samples-dir", type=Path, default=repo_root() / "ml_recon" / "data" / "feedback_samples")
    parser.add_argument("--matrix-dir", type=Path, default=Path(__file__).resolve().parent / "matrix")
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--target-count", type=float)
    parser.add_argument("--target-method", choices=("median", "trimmed-mean"), default="median")
    parser.add_argument("--min-correction", type=float, default=0.5)
    parser.add_argument("--max-correction", type=float, default=3.0)
    parser.add_argument("--min-amp", type=float, default=0.25)
    parser.add_argument("--max-amp", type=float, default=3.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return np.round(value, 6).tolist()
    if isinstance(value, np.floating):
        return float(value)
    return value


def main(argv=None) -> int:
    args = parse_args(argv)
    sample = args.sample_json or find_latest_feedback_sample(args.samples_dir)
    paths = matrix_paths(args.matrix_dir)
    result = compute_update(
        load_signal_bins(sample), load_current(paths), target_count=args.target_count,
        target_method=args.target_method, min_correction=args.min_correction,
        max_correction=args.max_correction, min_amp=args.min_amp, max_amp=args.max_amp,
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backups = []
    if not args.dry_run:
        args.matrix_dir.mkdir(parents=True, exist_ok=True)
        for cycle, path in enumerate(paths):
            backup = path.with_name(f"{path.stem}_{stamp}_before_signal_flatten{path.suffix}")
            if path.exists():
                shutil.copy2(path, backup)
            else:
                np.savetxt(backup, np.ones((16, 4)), fmt="%.6f")
            np.savetxt(path, result["updated_global_amp"][cycle], fmt="%.6f")
            backups.append(str(backup))
    report_dir = args.report_dir or args.matrix_dir / "calibration_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"wch1_signal_flatten_4qubit_{stamp}.json"
    report = {
        "system_type": "4qubit",
        "source_sample": str(sample),
        "matrix_paths": [str(path) for path in paths],
        "backup_paths": backups,
        "dry_run": bool(args.dry_run),
        "mapping": "signal_bins.reshape(4 cycles, 16 rows, 4 operation groups)",
        "assumptions": {"signal_scales_with_amplitude_squared": True, "target_count_method": args.target_method},
        "bounds": {"min_correction": args.min_correction, "max_correction": args.max_correction,
                   "min_amp": args.min_amp, "max_amp": args.max_amp},
        "result": {key: _json_value(value) for key, value in result.items()},
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report={report_path}")
    print(f"signal_cv_before={result['signal_stats_before']['cv']:.6f}")
    print(f"predicted_signal_cv_after={result['predicted_signal_stats_after']['cv']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
