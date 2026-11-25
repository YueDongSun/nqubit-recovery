"""Build 4-qubit CH3 amp matrices from 16x16 and 16x4 raw matrices.

Defaults:
    matrix/Wmat16raw.txt + matrix/Wmat4raw.txt -> matrix/W_CH3_amp_4qubit.txt
    matrix/Rmat16raw.txt + matrix/Rmat4raw.txt -> matrix/R_CH3_amp_4qubit.txt

The CH3 sequence code later uses sqrt(amp_matrix[row, col]). For that reason
the expanded 16x4 factors are squared before being multiplied into the raw
16x16 matrix, so the final waveform amplitude is scaled by the raw 16x4 value.
This builder only generates amp matrices; phase matrices are not modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
MATRIX_DIR = SCRIPT_DIR / "matrix"
RAW16_SHAPE = (16, 16)
RAW4_SHAPE = (16, 4)
OUTPUT_SHAPE = (16, 16)
REPEAT_PER_GROUP = 4


def expand_16x4_to_16x16(raw4: np.ndarray) -> np.ndarray:
    """Repeat each raw4 column four times: 1,2,3,4 -> 1,1,1,1,2,2,2,2,..."""
    raw4 = np.asarray(raw4, dtype=float)
    if raw4.shape != RAW4_SHAPE:
        raise ValueError(f"raw4 matrix must have shape {RAW4_SHAPE}, got {raw4.shape}")
    return np.repeat(raw4, REPEAT_PER_GROUP, axis=1)


def build_amp_matrix_4qubit(raw16: np.ndarray, raw4: np.ndarray) -> np.ndarray:
    """Return raw16 multiplied by the squared expanded 16x4 scale matrix."""
    raw16 = np.asarray(raw16, dtype=float)
    if raw16.shape != RAW16_SHAPE:
        raise ValueError(f"raw16 matrix must have shape {RAW16_SHAPE}, got {raw16.shape}")

    expanded = expand_16x4_to_16x16(raw4)
    amp_matrix = raw16 * np.square(expanded)
    if amp_matrix.shape != OUTPUT_SHAPE:
        raise ValueError(f"output matrix must have shape {OUTPUT_SHAPE}, got {amp_matrix.shape}")
    if not np.all(np.isfinite(amp_matrix)):
        raise ValueError("output matrix contains NaN or Inf values")
    return amp_matrix


def load_matrix(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing input file: {path}")
    return np.loadtxt(path)


def build_and_save(raw16_path: Path, raw4_path: Path, output_path: Path, fmt: str) -> None:
    amp_matrix = build_amp_matrix_4qubit(load_matrix(raw16_path), load_matrix(raw4_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, amp_matrix, fmt=fmt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build 4-qubit W_CH3_amp and R_CH3_amp matrices from 16x16 and 16x4 raw matrices."
    )
    parser.add_argument("--matrix-dir", type=Path, default=MATRIX_DIR)
    parser.add_argument("--w16", type=Path, default=None, help="W 16x16 raw matrix path.")
    parser.add_argument("--w4", type=Path, default=None, help="W 16x4 raw matrix path.")
    parser.add_argument("--r16", type=Path, default=None, help="R 16x16 raw matrix path.")
    parser.add_argument("--r4", type=Path, default=None, help="R 16x4 raw matrix path.")
    parser.add_argument("--w-out", type=Path, default=None, help="Output path for W_CH3 4-qubit amp matrix.")
    parser.add_argument("--r-out", type=Path, default=None, help="Output path for R_CH3 4-qubit amp matrix.")
    parser.add_argument("--fmt", default="%.6f", help="numpy.savetxt numeric format.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix_dir = args.matrix_dir

    w16 = args.w16 or matrix_dir / "Wmat16raw.txt"
    w4 = args.w4 or matrix_dir / "Wmat4raw.txt"
    r16 = args.r16 or matrix_dir / "Rmat16raw.txt"
    r4 = args.r4 or matrix_dir / "Rmat4raw.txt"
    w_out = args.w_out or matrix_dir / "W_CH3_amp_4qubit.txt"
    r_out = args.r_out or matrix_dir / "R_CH3_amp_4qubit.txt"

    build_and_save(w16, w4, w_out, args.fmt)
    build_and_save(r16, r4, r_out, args.fmt)

    print(f"Saved {w_out}")
    print(f"Saved {r_out}")


if __name__ == "__main__":
    main()
