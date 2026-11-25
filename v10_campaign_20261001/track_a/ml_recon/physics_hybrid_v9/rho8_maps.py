"""V9 canonical rho8 compensation maps (charter sections 9.2-9.3).

Four maps per system:

- ``retrieval_raw`` / ``retrieval_sqrt`` from ``retrieval_efficiency_{system}.txt``
  (3Q 15x15, 4Q 16x16);
- ``idler_raw`` / ``idler_sqrt`` from ``histogram_2026-06-24_22-01-37.csv`` row
  "Idler Absolute" (256 values -> 16x16; 3Q takes the top-left 15x15).

Canonical normalization (identical for every map):

    positive_max = max(M[M>0]);  M_norm = M / positive_max
    M_sqrt       = sqrt(M_norm); M_sqrt = M_sqrt / max(M_sqrt[M_sqrt>0])

Every map stores ``raw_file_sha256`` / ``raw_matrix_sha256`` /
``normalized_matrix_sha256`` provenance. Unlike V8, the sqrt map is re-normalized and
all three hashes are recorded, so the normalized matrix is explicit and versioned.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .contracts import V9RhoViewProvenanceV1

MAP_KEYS = ("retrieval_raw", "retrieval_sqrt", "idler_raw", "idler_sqrt")
REPO_ROOT = Path(__file__).resolve().parents[2]
QUBIT_RECON_DIR = REPO_ROOT / "system_control" / "qubit_recon_files"
MATRIX_DIR = QUBIT_RECON_DIR / "matrix"
DEFAULT_RETRIEVAL_PATHS = {
    "3qubit": MATRIX_DIR / "retrieval_efficiency_3qubit.txt",
    "4qubit": MATRIX_DIR / "retrieval_efficiency_4qubit.txt",
}
DEFAULT_IDLER_CSV = REPO_ROOT / "histogram_2026-06-24_22-01-37.csv"
RETRIEVAL_SHAPES = {"3qubit": (15, 15), "4qubit": (16, 16)}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matrix_sha256(matrix: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(matrix).tobytes()).hexdigest()


def normalize_map(matrix: np.ndarray, transform: str) -> np.ndarray:
    """Apply canonical normalization: raw or sqrt (with sqrt re-normalization)."""
    values = np.asarray(matrix, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("map must be finite and nonnegative")
    positive = values[values > 0.0]
    if positive.size == 0:
        raise ValueError("map has no positive support (positive_max must be > 0)")
    positive_max = float(np.max(positive))
    normalized = values / positive_max
    if transform == "raw":
        return normalized
    if transform == "sqrt":
        sqrt_values = np.sqrt(normalized)
        sqrt_positive = sqrt_values[sqrt_values > 0.0]
        if sqrt_positive.size == 0:
            raise ValueError("sqrt map has no positive support")
        return sqrt_values / float(np.max(sqrt_positive))
    raise ValueError(f"transform must be 'raw' or 'sqrt', got {transform}")


def load_retrieval_map(
    system_type: str,
    path: str | Path | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load and canonically normalize the retrieval-efficiency TXT map."""
    if system_type not in RETRIEVAL_SHAPES:
        raise ValueError(f"system_type must be one of {list(RETRIEVAL_SHAPES)}")
    path = Path(path or DEFAULT_RETRIEVAL_PATHS[system_type])
    if not path.is_file():
        raise FileNotFoundError(path)
    raw_file_sha = file_sha256(path)
    matrix = np.loadtxt(path, dtype=np.float64)
    if matrix.shape != RETRIEVAL_SHAPES[system_type]:
        raise ValueError(
            f"retrieval map for {system_type} must have shape "
            f"{RETRIEVAL_SHAPES[system_type]}, got {matrix.shape}"
        )
    raw_matrix_sha = matrix_sha256(matrix)
    normalized_raw = normalize_map(matrix, "raw")
    normalized_sqrt = normalize_map(matrix, "sqrt")
    return normalized_raw, {
        "raw_file_sha256": raw_file_sha,
        "raw_matrix_sha256": raw_matrix_sha,
        "normalized_raw_sha256": matrix_sha256(normalized_raw),
        "normalized_sqrt_sha256": matrix_sha256(normalized_sqrt),
        "source_path": str(path.resolve()),
    }


def load_idler_csv_map(
    system_type: str,
    path: str | Path | None = None,
    row_label: str = "Idler Absolute",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load and canonically normalize the idler CSV map.

    Reads the full 256 values, normalizes to 16x16, and for 3Q takes the top-left
    15x15 (charter section 9.3).
    """
    if system_type not in RETRIEVAL_SHAPES:
        raise ValueError(f"system_type must be one of {list(RETRIEVAL_SHAPES)}")
    path = Path(path or DEFAULT_IDLER_CSV)
    if not path.is_file():
        raise FileNotFoundError(path)
    raw_file_sha = file_sha256(path)
    target = row_label.strip().casefold()
    values = None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            if row[0].strip().casefold() == target:
                values = [float(item) for item in row[1:] if str(item).strip()]
                break
    if values is None:
        raise ValueError(f"CSV row {row_label!r} not found in {path}")
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size != 256:
        raise ValueError(f"{row_label!r} row must contain 256 idler counts, got {arr.size}")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError("idler CSV must be finite and nonnegative")
    matrix_16 = arr.reshape(16, 16)
    if system_type == "3qubit":
        matrix = matrix_16[:15, :15].copy()
    else:
        matrix = matrix_16
    raw_matrix_sha = matrix_sha256(matrix)
    normalized_raw = normalize_map(matrix, "raw")
    normalized_sqrt = normalize_map(matrix, "sqrt")
    return normalized_raw, {
        "raw_file_sha256": raw_file_sha,
        "raw_matrix_sha256": raw_matrix_sha,
        "normalized_raw_sha256": matrix_sha256(normalized_raw),
        "normalized_sqrt_sha256": matrix_sha256(normalized_sqrt),
        "source_path": str(path.resolve()),
    }


def build_rho8_maps(
    system_type: str,
    *,
    retrieval_map_path: str | Path | None = None,
    idler_csv_path: str | Path | None = None,
) -> Dict[str, Tuple[np.ndarray, V9RhoViewProvenanceV1]]:
    """Build the four canonical normalized maps with provenance.

    Returns ``{map_key: (matrix, V9RhoViewProvenanceV1)}``.
    """
    retrieval_raw, r_meta = load_retrieval_map(system_type, retrieval_map_path)
    idler_raw, i_meta = load_idler_csv_map(system_type, idler_csv_path)

    maps = {
        "retrieval_raw": retrieval_raw,
        "idler_raw": idler_raw,
    }
    provenance: Dict[str, V9RhoViewProvenanceV1] = {}

    def _prov(map_key: str, transform: str, meta: Dict[str, Any], shape: Tuple[int, int]) -> V9RhoViewProvenanceV1:
        return V9RhoViewProvenanceV1(
            map_key=map_key,
            system_type=system_type,
            source_path=meta["source_path"],
            transform=transform,
            raw_file_sha256=meta["raw_file_sha256"],
            raw_matrix_sha256=meta["raw_matrix_sha256"],
            normalized_matrix_sha256=meta[f"normalized_{transform}_sha256"],
            shape=shape,
        )

    for key, transform in (("retrieval_raw", "raw"), ("retrieval_sqrt", "sqrt")):
        provenance[key] = _prov(key, transform, r_meta, RETRIEVAL_SHAPES[system_type])
    for key, transform in (("idler_raw", "raw"), ("idler_sqrt", "sqrt")):
        provenance[key] = _prov(key, transform, i_meta, RETRIEVAL_SHAPES[system_type])

    maps["retrieval_sqrt"] = normalize_map(retrieval_raw, "sqrt")
    maps["idler_sqrt"] = normalize_map(idler_raw, "sqrt")
    return {key: (maps[key], provenance[key]) for key in MAP_KEYS}
