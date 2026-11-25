"""V9 real-data seam (charter sections 5, 44).

Adapts the authoritative V8 DLCZ measured data (V2.3 trusted corpus) into the same
per-operation ``signal_counts/idler_counts/coin_counts/read_count`` sample format that
the V9 synthetic stream produces, so the F1/F2 real static pretraining and the
classifier real static non-regression evaluation can consume real rows with the same
code path.

Key mappings (verified against the V8 loader):
- padded ``(B, 256)`` counts are reduced to per-operation arrays via ``operation_mask``
  (3Q = 75, 4Q = 256 effective operations);
- ``read_count`` is a per-row scalar in V8 and is broadcast to per-operation;
- global multi pairs/counts map to the V9 global-multi dict;
- target amplitudes/phases (4-col, masked) map to gauge-invariant rho features.

Real rows provide count likelihood / rho8 weak consistency / calibration / OOD analysis
and the F1/F2 feedback adapter data. They never provide family/error ground-truth
accuracy and never set real action supervision masks.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from .operation_target_law import target_rho_features


def _target_coefficients(row_idx: int, system_type: str,
                         amplitudes: np.ndarray, phases: np.ndarray,
                         target_mask: np.ndarray) -> Optional[np.ndarray]:
    """Extract the normalized target W coefficients for one row."""
    n = 3 if system_type == "3qubit" else 4
    amp = np.asarray(amplitudes[row_idx], dtype=np.float64)
    phase = np.asarray(phases[row_idx], dtype=np.float64)
    mask = np.asarray(target_mask[row_idx], dtype=bool)
    if np.sum(mask) < n:
        return None
    c = amp[:n] * np.exp(1j * phase[:n])
    norm = float(np.sum(np.abs(c) ** 2))
    if norm <= 1e-12:
        return None
    return c / np.sqrt(norm)


class V9RealDataSeam:
    """Loads V8 DLCZ measured data and converts rows to the V9 synthetic format."""

    def __init__(
        self,
        *,
        manifest_dir: str = "ml_recon/reports/measured_data_v2_3",
        data_root: str = "H:/N_qubit/data",
        integrity_check_mode: str = "quick_manifest_entry",
        include_rho_payloads: bool = False,
    ):
        from ml_recon.physics_hybrid_v8.dlcz_noise_data import (
            load_v8_dlcz_development_data,
        )

        self.data = load_v8_dlcz_development_data(
            manifest_dir=manifest_dir,
            data_root=data_root,
            include_rho_payloads=include_rho_payloads,
            integrity_check_mode=integrity_check_mode,
        )
        self.system_types = list(self.data.system_types)
        self.sample_ids = list(self.data.sample_ids)

    def row_to_sample(self, row_idx: int) -> Dict[str, Any]:
        """Convert one real row to a V9 synthetic-format sample dict."""
        system_type = self.system_types[row_idx]
        n_ops = 75 if system_type == "3qubit" else 256
        mask = np.asarray(self.data.operation_mask[row_idx], dtype=bool)
        if int(np.sum(mask)) != n_ops:
            raise ValueError(
                f"operation_mask for row {row_idx} ({system_type}) has {int(np.sum(mask))} "
                f"effective ops, expected {n_ops}"
            )
        signal = np.asarray(self.data.signal_counts[row_idx], dtype=np.float64)[mask]
        idler = np.asarray(self.data.idler_counts[row_idx], dtype=np.float64)[mask]
        coin = np.asarray(self.data.coin_counts[row_idx], dtype=np.float64)[mask]
        read_scalar = int(self.data.read_count[row_idx])
        read = np.full(n_ops, float(read_scalar), dtype=np.float64)

        gm = self.data.global_multi_counts[row_idx]  # [multi2, multi3, multi4]
        pairs = self.data.global_multi2_pairs[row_idx]  # [6]
        global_multi = {
            "multi2": float(gm[0]), "multi3": float(gm[1]), "multi4": float(gm[2]),
            "pair12": float(pairs[0]), "pair13": float(pairs[1]), "pair14": float(pairs[2]),
            "pair23": float(pairs[3]), "pair24": float(pairs[4]), "pair34": float(pairs[5]),
        }

        c = _target_coefficients(
            row_idx, system_type,
            self.data.target_amplitudes, self.data.target_phases_rad, self.data.target_mask,
        )
        target_features = None
        born = None
        if c is not None:
            from .rho8_reconstruction import _load_calibrated_reconstruction_module

            rho = np.outer(c, c.conj())
            target_features = target_rho_features(rho)
            module = _load_calibrated_reconstruction_module()
            entries = module.raw_reconstruction.load_basis_entries(
                system_type,
                f"system_control/qubit_recon_files/matrix/basis_config_{system_type}.json")
            born = np.asarray([
                float(np.real(np.vdot(e["amp"] * np.exp(1j * np.asarray(e.get("phase", 0))),
                                    rho @ (e["amp"] * np.exp(1j * np.asarray(e.get("phase", 0))))))
                / max(float(np.sum(np.abs(e["amp"]) ** 2)), 1e-12))
                for e in entries
            ])

        return {
            "sample_id": self.sample_ids[row_idx],
            "system_type": system_type,
            "counts": {"signal_counts": signal, "idler_counts": idler,
                       "coin_counts": coin, "read_count": read},
            "global_multi": global_multi,
            "target_coefficients": c,
            "target_features": target_features,
            "born_prior": born,
            "exposure_s": float(self.data.exposure_s[row_idx]),
            "exposure_known_mask": bool(self.data.exposure_known_mask[row_idx]),
            "real_action_supervision_mask": 0.0,
        }

    def iter_rows(self) -> Iterator[Dict[str, Any]]:
        for i in range(len(self.sample_ids)):
            yield self.row_to_sample(i)

    def system_rows(self, system_type: str) -> List[int]:
        return [i for i, s in enumerate(self.system_types) if s == system_type]

    def development_train_indices(self, system_type: str) -> List[int]:
        """V9 development train rows (3Q 739 + 4Q 1240 = 1979, contract section 5.2).

        The V8 loader encodes the V9 dev-train as split names ``train`` (3Q),
        ``train_target_family`` + ``probe_train`` (4Q). Validation and probe-validation
        rows are excluded.
        """
        dev_train_splits = {"train", "train_target_family", "probe_train"}
        return [
            i for i in self.system_rows(system_type)
            if self.data.split_names[i] in dev_train_splits
        ]

    def iter_development_train(self, system_type: str) -> Iterator[Dict[str, Any]]:
        for i in self.development_train_indices(system_type):
            yield self.row_to_sample(i)

    def grouped_adapter_indices(
        self,
        system_type: str,
        target_count: int,
        seed: int = 0,
    ) -> List[int]:
        """F2 grouped adapter rows (contract 5.2): sample whole repeat groups without
        splitting them, up to ``target_count`` rows. Repeat/acquisition groups are never
        torn apart (3Q groups are mostly size 1; 4Q mostly size 2)."""
        dev_idx = self.development_train_indices(system_type)
        groups: Dict[str, List[int]] = {}
        for i in dev_idx:
            g = self.data.repeat_group_ids[i]
            groups.setdefault(g, []).append(i)
        rng = np.random.default_rng(seed)
        group_ids = list(groups.keys())
        rng.shuffle(group_ids)
        chosen: List[int] = []
        for g in group_ids:
            if len(chosen) + len(groups[g]) > target_count:
                continue  # do not split the group
            chosen.extend(groups[g])
        return chosen

    def iter_adapter_rows(self, system_type: str, target_count: int,
                          seed: int = 0) -> Iterator[Dict[str, Any]]:
        for i in self.grouped_adapter_indices(system_type, target_count, seed):
            yield self.row_to_sample(i)

    def summary(self) -> Dict[str, Any]:
        from collections import Counter

        return {
            "n_rows": len(self.sample_ids),
            "by_system": dict(Counter(self.system_types)),
            "exposure_known_fraction": float(np.mean(self.data.exposure_known_mask)),
            "split_names": list(dict.fromkeys(self.data.split_names)),
            "stable_hash": self.data.stable_hash,
        }
