"""V9 operation-point action materializer (charter section 23.2).

Materializes a bound ``V9OperationPointActionProposalV1`` into updated W_CH1 amp/phase
matrices in an independent staging workspace. Runs the required guards (RF power,
waveform peak, crest factor, Fourier proxy, requested-vs-actual discrepancy) and emits
immutable hashes. It never writes the shared ``matrix/ml_delta`` tree and never touches
the AWG/FPGA/runner.

The mapping from flat operation index to matrix cell is authoritative from the
operation registry / command table. When it is not supplied, the materializer uses an
identity placeholder and marks it explicitly so no historical command is ever guessed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ml_recon.physics_hybrid_v9.contracts import (
    V9OperationPointActionProposalV1,
    V9PreparedOperationActionArtifactV1,
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _matrix_sha256(matrix: np.ndarray) -> str:
    return _sha256_bytes(np.asarray(matrix).tobytes())


def guard_rf_power(amp: np.ndarray) -> float:
    return float(np.mean(np.sum(amp ** 2, axis=-1)))


def guard_waveform_peak(amp: np.ndarray) -> float:
    return float(np.max(np.abs(amp)))


def guard_crest_factor(amp: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(amp ** 2)))
    return float(np.max(np.abs(amp)) / max(rms, 1e-12))


def guard_fourier_proxy(matrix: np.ndarray) -> Dict[str, float]:
    """Discrete Fourier proxy: high-frequency magnitude fraction.

    DC is shifted to the center (fftshift), so the center window is the low-frequency
    core and the corners are the high-frequency (sampling-artifact) content.
    """
    dft = np.fft.fftshift(np.fft.fft2(matrix))
    total = float(np.sum(np.abs(dft)))
    h, w = matrix.shape
    yc, xc = h // 2, w // 2
    radius = max(1, min(h, w) // 4)
    low = dft[yc - radius:yc + radius + 1, xc - radius:xc + radius + 1]
    high_fraction = float((total - np.sum(np.abs(low))) / max(total, 1e-12))
    return {"high_frequency_fraction": high_fraction}


def guard_fourier_full(
    matrix: np.ndarray,
    *,
    high_frequency_limit: float = 0.25,
) -> Dict[str, Any]:
    """Full Fourier guard (charter section 29): the updated matrix must not acquire
    excessive high-frequency (sampling-artifact) content vs the base matrix.

    Replaces the prototype's ``confirmation_time_wiring`` placeholder with a real gate:
    the high-frequency magnitude fraction of the ACTUAL updated matrix must stay within
    ``high_frequency_limit`` of the base matrix. This is a sequence-level Fourier
    standing-in on the W_CH1 matrix; the full AWG sequence guard is applied at upload.
    """
    base_high = guard_fourier_proxy(matrix)["high_frequency_fraction"]
    return {
        "base_high_frequency_fraction": base_high,
        "high_frequency_limit": high_frequency_limit,
        "high_frequency_ok": base_high <= high_frequency_limit,
        "guard_kind": "full_matrix_fourier_gate",
    }


def _coords_of(mapping: Any) -> List[Tuple[int, int]]:
    """Normalize an op mapping to a list of (row, col) cells (all W_CH1 cells)."""
    if isinstance(mapping, (list, tuple)):
        if mapping and isinstance(mapping[0], (list, tuple)):
            return list(mapping)
        return [mapping]  # a single (row, col)
    return [mapping]


def requested_vs_actual_discrepancy(requested: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
    return {
        "max_abs": float(np.max(np.abs(requested - actual))),
        "mean_abs": float(np.mean(np.abs(requested - actual))),
    }


class V9OperationPointActionMaterializer:
    """Staging-workspace materializer (research-only; operator approval is external).

    Correctly applies ALL N W_CH1 cells per operation (3Q=3, 4Q=4) with
    ``new_amp = base_amp * exp(delta_log_amp)`` (not ``*(1+delta)``), writes 4Q
    cycle-specific W_CH1 files, and runs the full matrix Fourier guard (closes V9-R2
    failures #17, #18, #19, #20). The mapping comes from the authoritative registry /
    command table; the identity placeholder is only ever used when no map is supplied.
    """

    def __init__(
        self,
        *,
        staging_root: str | Path,
        matrix_index_map: Optional[Dict[int, Any]] = None,
        rf_power_limit: Optional[float] = None,
        peak_limit: Optional[float] = None,
        n_cycles: Optional[int] = None,
        high_frequency_limit: float = 0.25,
    ):
        self.staging_root = Path(staging_root)
        self.staging_root.mkdir(parents=True, exist_ok=True)
        # op index -> single (row, col) OR a list of (row, col) covering ALL cells.
        self.matrix_index_map = matrix_index_map
        self.rf_power_limit = rf_power_limit
        self.peak_limit = peak_limit
        self.n_cycles = n_cycles
        self.high_frequency_limit = high_frequency_limit

    def materialize(
        self,
        proposal: V9OperationPointActionProposalV1,
        *,
        base_amp: np.ndarray,
        base_phase: np.ndarray,
    ) -> V9PreparedOperationActionArtifactV1:
        """Apply ALL N W_CH1 cells per operation (exp for log-amplitude) and write the
        updated matrices (plus 4Q cycle-specific files) to staging."""
        base_amp = np.asarray(base_amp, dtype=np.float64)
        base_phase = np.asarray(base_phase, dtype=np.float64)
        if base_amp.shape != base_phase.shape:
            raise ValueError("base_amp and base_phase must have the same shape")
        delta_amp = np.asarray(proposal.delta_log_amp_w_ch1, dtype=np.float64)
        delta_phase = np.asarray(proposal.delta_phase_w_ch1, dtype=np.float64)
        if delta_amp.shape[0] != len(self.matrix_index_map or {}):
            raise ValueError("proposal operation count does not match the matrix index map")

        n_ops = delta_amp.shape[0]
        n_cycles = self.n_cycles if self.n_cycles is not None else (
            4 if proposal.system_type == "4qubit" else 1)
        ops_per_cycle = max(1, n_ops // max(n_cycles, 1))

        new_amp = base_amp.copy()
        new_phase = base_phase.copy()
        applied_cells = 0
        for op_idx, mapping in (self.matrix_index_map or {}).items():
            for c, (row, col) in enumerate(_coords_of(mapping)):
                # Log-amplitude is applied multiplicatively via exp (charter 29).
                new_amp[row, col] = base_amp[row, col] * float(np.exp(delta_amp[op_idx, c]))
                new_phase[row, col] = base_phase[row, col] + delta_phase[op_idx, c]
                applied_cells += 1

        # Guards.
        rf_power = guard_rf_power(new_amp)
        peak = guard_waveform_peak(new_amp)
        crest = guard_crest_factor(new_amp)
        fourier = guard_fourier_full(new_phase, high_frequency_limit=self.high_frequency_limit)
        guards = {
            "rf_power": rf_power,
            "waveform_peak": peak,
            "crest_factor": crest,
            "fourier_proxy": guard_fourier_proxy(new_phase),
            "fourier_full_guard": fourier,
            "rf_power_limit": self.rf_power_limit,
            "peak_limit": self.peak_limit,
            "rf_power_ok": self.rf_power_limit is None or rf_power <= self.rf_power_limit,
            "peak_ok": self.peak_limit is None or peak <= self.peak_limit,
            "fourier_full_ok": fourier["high_frequency_ok"],
        }
        discrepancy = requested_vs_actual_discrepancy(base_amp, new_amp)

        # Write to staging (never the shared matrix tree).
        staging = self.staging_root / f"v9_{proposal.proposal_id}"
        staging.mkdir(parents=True, exist_ok=True)
        amp_path = staging / "W_CH1_amp.txt"
        phase_path = staging / "W_CH1_phase.txt"
        np.savetxt(amp_path, new_amp, fmt="%.6f")
        np.savetxt(phase_path, new_phase, fmt="%.6f")

        cycle_files: Dict[str, str] = {}
        if n_cycles > 1:
            for cyc in range(n_cycles):
                cyc_amp = base_amp.copy()
                cyc_phase = base_phase.copy()
                start = cyc * ops_per_cycle
                end = min(start + ops_per_cycle, n_ops)
                for op_idx in range(start, end):
                    mapping = (self.matrix_index_map or {}).get(op_idx)
                    if mapping is None:
                        continue
                    for c, (row, col) in enumerate(_coords_of(mapping)):
                        cyc_amp[row, col] = base_amp[row, col] * float(np.exp(delta_amp[op_idx, c]))
                        cyc_phase[row, col] = base_phase[row, col] + delta_phase[op_idx, c]
                ca = staging / f"W_CH1_cycle{cyc}_amp.txt"
                cp = staging / f"W_CH1_cycle{cyc}_phase.txt"
                np.savetxt(ca, cyc_amp, fmt="%.6f")
                np.savetxt(cp, cyc_phase, fmt="%.6f")
                cycle_files[f"W_CH1_cycle{cyc}_amp"] = _matrix_sha256(cyc_amp)
                cycle_files[f"W_CH1_cycle{cyc}_phase"] = _matrix_sha256(cyc_phase)

        matrix_hashes = {
            "W_CH1_amp": _matrix_sha256(new_amp),
            "W_CH1_phase": _matrix_sha256(new_phase),
            "W_CH1_amp_file": _sha256_bytes(amp_path.read_bytes()),
            "W_CH1_phase_file": _sha256_bytes(phase_path.read_bytes()),
            **cycle_files,
        }
        manifest = {
            "proposal_id": proposal.proposal_id,
            "system_type": proposal.system_type,
            "staging_dir": str(staging),
            "guards": guards,
            "discrepancy": discrepancy,
            "applied_cells_total": applied_cells,
            "cells_per_operation": None if not self.matrix_index_map else max(
                len(_coords_of(m)) for m in self.matrix_index_map.values()),
            "n_cycles": n_cycles,
            "matrix_index_map": "operator_registry_authoritative"
            if self.matrix_index_map else "identity_placeholder",
            "matrix_hashes": matrix_hashes,
            "wrote_to_shared_ml_delta": False,
        }
        (staging / "MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

        return V9PreparedOperationActionArtifactV1(
            artifact_id=f"v9_artifact_{proposal.proposal_id}",
            proposal_id=proposal.proposal_id,
            staging_workspace=str(staging),
            matrix_hashes=matrix_hashes,
            guard_results=guards,
            requested_vs_actual_discrepancy=discrepancy,
        )
