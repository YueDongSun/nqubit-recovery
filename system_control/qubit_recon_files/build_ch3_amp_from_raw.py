"""Build 3-qubit CH3 amp matrices from 15x15 and 15x5 raw matrices.

Defaults:
    matrix/Wmat15raw.txt + matrix/Wmat5raw.txt -> matrix/W_CH3_amp.txt
    matrix/Rmat15raw.txt + matrix/Rmat5raw.txt -> matrix/R_CH3_amp.txt

The CH3 sequence code later uses sqrt(amp_matrix[row, col]). For that reason
the expanded 15x5 factors are squared before being multiplied into the raw
15x15 matrix, so the final waveform amplitude is scaled by the raw 15x5 value.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
MATRIX_DIR = SCRIPT_DIR / "matrix"
RAW15_SHAPE = (15, 15)
RAW5_SHAPE = (15, 5)
OUTPUT_SHAPE = (15, 15)
REPEAT_PER_GROUP = 3


def expand_15x5_to_15x15(raw5: np.ndarray) -> np.ndarray:
    """Repeat each raw5 column three times: 1,2,3,4,5 -> 1,1,1,2,2,2,..."""
    raw5 = np.asarray(raw5, dtype=float)
    if raw5.shape != RAW5_SHAPE:
        raise ValueError(f"raw5 matrix must have shape {RAW5_SHAPE}, got {raw5.shape}")
    return np.repeat(raw5, REPEAT_PER_GROUP, axis=1)


def build_amp_matrix(raw15: np.ndarray, raw5: np.ndarray) -> np.ndarray:
    """Return raw15 multiplied by the squared expanded 15x5 scale matrix."""
    raw15 = np.asarray(raw15, dtype=float)
    if raw15.shape != RAW15_SHAPE:
        raise ValueError(f"raw15 matrix must have shape {RAW15_SHAPE}, got {raw15.shape}")

    expanded = expand_15x5_to_15x15(raw5)
    amp_matrix = raw15 * np.square(expanded)
    if amp_matrix.shape != OUTPUT_SHAPE:
        raise ValueError(f"output matrix must have shape {OUTPUT_SHAPE}, got {amp_matrix.shape}")
    if not np.all(np.isfinite(amp_matrix)):
        raise ValueError("output matrix contains NaN or Inf values")
    return amp_matrix


def load_matrix(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing input file: {path}")
    return np.loadtxt(path)


def build_and_save(raw15_path: Path, raw5_path: Path, output_path: Path, fmt: str) -> None:
    amp_matrix = build_amp_matrix(load_matrix(raw15_path), load_matrix(raw5_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, amp_matrix, fmt=fmt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build W_CH3_amp.txt and R_CH3_amp.txt from 15x15 and 15x5 raw matrices."
    )
    parser.add_argument("--matrix-dir", type=Path, default=MATRIX_DIR)
    parser.add_argument("--w15", type=Path, default=None, help="W 15x15 raw matrix path.")
    parser.add_argument("--w5", type=Path, default=None, help="W 15x5 raw matrix path.")
    parser.add_argument("--r15", type=Path, default=None, help="R 15x15 raw matrix path.")
    parser.add_argument("--r5", type=Path, default=None, help="R 15x5 raw matrix path.")
    parser.add_argument("--w-out", type=Path, default=None, help="Output path for W_CH3 amp matrix.")
    parser.add_argument("--r-out", type=Path, default=None, help="Output path for R_CH3 amp matrix.")
    parser.add_argument("--fmt", default="%.6f", help="numpy.savetxt numeric format.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix_dir = args.matrix_dir

    w15 = args.w15 or matrix_dir / "Wmat15raw.txt"
    w5 = args.w5 or matrix_dir / "Wmat5raw.txt"
    r15 = args.r15 or matrix_dir / "Rmat15raw.txt"
    r5 = args.r5 or matrix_dir / "Rmat5raw.txt"
    w_out = args.w_out or matrix_dir / "W_CH3_amp.txt"
    r_out = args.r_out or matrix_dir / "R_CH3_amp.txt"

    build_and_save(w15, w5, w_out, args.fmt)
    build_and_save(r15, r5, r_out, args.fmt)

    print(f"Saved {w_out}")
    print(f"Saved {r_out}")


if __name__ == "__main__":
    main()
