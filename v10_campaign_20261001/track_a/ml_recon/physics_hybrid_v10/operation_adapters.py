"""Read-only adapters to verified V9/V2.3 seams."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from ml_recon.physics_hybrid_v9.operation_registry import build_default_registry
from ml_recon.physics_hybrid_v9.contracts import V9RhoEightViewContractV1

from .contracts import V10FourCellCounts, V10GlobalMultiObservation, sha256_of


V23_DEVELOPMENT_SPLITS = {"train", "train_target_family", "probe_train"}


@dataclass(frozen=True)
class V10AdaptedOperationBatch:
    system_type: str
    sample_ids: Tuple[str, ...]
    session_ids: Tuple[str, ...]
    trajectory_ids: Tuple[str, ...]
    split_names: Tuple[str, ...]
    operation_ids: Tuple[str, ...]
    operation_registry_hash: str
    signal_counts: np.ndarray
    idler_counts: np.ndarray
    coin_counts: np.ndarray
    read_counts: np.ndarray
    four_cell_counts: np.ndarray
    operation_mask: np.ndarray
    global_multi: Tuple[V10GlobalMultiObservation, ...]
    global_multi_mask: np.ndarray
    temperature_features: np.ndarray
    temperature_mask: np.ndarray
    exposure_s: np.ndarray
    exposure_known_mask: np.ndarray
    real_action_supervision_mask: np.ndarray
    source_manifest_hash: str
    source_split_hash: str
    adapter_sha: str

    def validate(self) -> None:
        row_count = len(self.sample_ids)
        operation_count = len(self.operation_ids)
        if row_count <= 0 or len(set(self.sample_ids)) != row_count:
            raise ValueError("adapted V2.3 batch requires unique sample IDs")
        if not all(len(values) == row_count for values in (
                self.session_ids, self.trajectory_ids, self.split_names, self.global_multi)):
            raise ValueError("adapted row identity lengths disagree")
        expected = (row_count, operation_count)
        for name in ("signal_counts", "idler_counts", "coin_counts", "read_counts",
                     "operation_mask"):
            if np.asarray(getattr(self, name)).shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
        if np.asarray(self.four_cell_counts).shape != (*expected, 4):
            raise ValueError("four_cell_counts must have shape [rows,operations,4]")
        if not np.asarray(self.operation_mask).all():
            raise ValueError("system-specific adapted batches must contain only real operations")
        if np.any(np.asarray(self.four_cell_counts) < -1e-9):
            raise ValueError("adapted four-cell counts cannot be clipped or negative")
        if set(self.split_names) - V23_DEVELOPMENT_SPLITS:
            raise ValueError("locked validation/test rows entered V10 real training adapter")
        if np.any(np.asarray(self.real_action_supervision_mask) != 0.0):
            raise ValueError("V2.3 rows do not provide real action supervision")
        if np.asarray(self.temperature_features).shape != (row_count, 7) \
                or np.asarray(self.temperature_mask).shape != (row_count, 7):
            raise ValueError("temperature features/masks must retain the seven-field weak context")
        if np.any(np.asarray(self.temperature_mask)):
            raise ValueError("V2.3 operation payload has no bound temperature source; mask must stay false")


def v10_operation_registry(system_type: str):
    basis_path = Path("system_control/qubit_recon_files/matrix") / f"basis_config_{system_type}.json"
    if not basis_path.is_file():
        raise ValueError(f"authoritative operation command table is missing: {basis_path}")
    import hashlib
    command_table_hash = hashlib.sha256(basis_path.read_bytes()).hexdigest()
    return build_default_registry(system_type, command_table_hash=command_table_hash)


def v10_rho8_contract() -> Dict[str, Any]:
    contract = V9RhoEightViewContractV1()
    result = contract.to_dict()
    result.update({"v10_adapter": "V10Rho8ReadOnlyAdapterV1",
                   "derived_views_are_independent": False,
                   "weak_loss_max_fraction": 0.05})
    return result


def v10_real_data_seam(*, manifest_dir: str, data_root: str,
                       integrity_check_mode: str = "quick_manifest_entry"):
    """Open immutable V2.3 development data and verify the exact 739/1240 split."""
    from ml_recon.physics_hybrid_v9.v9_real_data_seam import V9RealDataSeam
    seam = V9RealDataSeam(manifest_dir=manifest_dir, data_root=data_root,
                          integrity_check_mode=integrity_check_mode,
                          include_rho_payloads=False)
    counts = {system: len(seam.development_train_indices(system))
              for system in ("3qubit", "4qubit")}
    if counts != {"3qubit": 739, "4qubit": 1240}:
        raise ValueError(f"V2.3 development split mismatch: {counts}")
    return seam


def adapt_v23_development_batch(seam: Any, system_type: str) -> V10AdaptedOperationBatch:
    """Adapt immutable V2.3 dev-train rows without normalization or row loss."""
    if system_type not in ("3qubit", "4qubit"):
        raise ValueError("system_type must be 3qubit or 4qubit")
    all_development = tuple(seam.development_train_indices("3qubit")) + tuple(
        seam.development_train_indices("4qubit"))
    if len(all_development) != 1979 or len(set(all_development)) != 1979:
        raise ValueError("V2.3 development population must be exactly 1979 unique rows")
    indices = tuple(seam.development_train_indices(system_type))
    expected_rows = 739 if system_type == "3qubit" else 1240
    if len(indices) != expected_rows:
        raise ValueError(f"V2.3 {system_type} development row mismatch")
    data = seam.data
    registry = v10_operation_registry(system_type)
    operation_count = registry.n_operations
    sample_ids = tuple(data.sample_ids[index] for index in indices)
    session_ids = tuple(data.acquisition_run_ids[index] for index in indices)
    trajectory_ids = tuple(data.repeat_group_ids[index] for index in indices)
    split_names = tuple(data.split_names[index] for index in indices)
    signal_rows = []
    idler_rows = []
    coin_rows = []
    read_rows = []
    four_rows = []
    global_multi = []
    global_multi_mask = []
    for index in indices:
        mask = np.asarray(data.operation_mask[index], dtype=np.bool_)
        if int(mask.sum()) != operation_count:
            raise ValueError("V2.3 operation mask count disagrees with registry")
        signal = np.asarray(data.signal_counts[index], dtype=np.float64)[mask]
        idler = np.asarray(data.idler_counts[index], dtype=np.float64)[mask]
        coin = np.asarray(data.coin_counts[index], dtype=np.float64)[mask]
        read = np.full(operation_count, float(data.read_count[index]), dtype=np.float64)
        cells = np.stack((read - signal - idler + coin,
                          signal - coin, idler - coin, coin), axis=-1)
        if np.any(cells < -1e-9):
            raise ValueError("V2.3 row violates four-cell count inclusion-exclusion")
        signal_rows.append(signal)
        idler_rows.append(idler)
        coin_rows.append(coin)
        read_rows.append(read)
        four_rows.append(cells)
        multi = np.asarray(data.global_multi_counts[index], dtype=np.int64)
        pairs = tuple(int(value) for value in data.global_multi2_pairs[index])
        observation = V10GlobalMultiObservation(
            multi2=int(multi[0]), pair_counts=pairs,
            multi3=int(multi[1]), multi4=int(multi[2]),
            scope=str(data.global_multi_scope[index]))
        observation.validate()
        global_multi.append(observation)
        global_multi_mask.append(bool(data.global_multi_observation_mask[index]))
    arrays = {
        "signal_counts": np.stack(signal_rows),
        "idler_counts": np.stack(idler_rows),
        "coin_counts": np.stack(coin_rows),
        "read_counts": np.stack(read_rows),
        "four_cell_counts": np.stack(four_rows),
    }
    adapter_payload = {
        "system_type": system_type,
        "sample_ids": sample_ids,
        "operation_registry_hash": registry.registry_hash(),
        "source_manifest_hash": data.manifest_hash,
        "source_split_hash": data.split_hash,
        "array_hashes": {key: sha256_of(value.tobytes()) for key, value in arrays.items()},
    }
    result = V10AdaptedOperationBatch(
        system_type=system_type,
        sample_ids=sample_ids,
        session_ids=session_ids,
        trajectory_ids=trajectory_ids,
        split_names=split_names,
        operation_ids=registry.flat_order,
        operation_registry_hash=registry.registry_hash(),
        signal_counts=arrays["signal_counts"],
        idler_counts=arrays["idler_counts"],
        coin_counts=arrays["coin_counts"],
        read_counts=arrays["read_counts"],
        four_cell_counts=arrays["four_cell_counts"],
        operation_mask=np.ones((len(indices), operation_count), dtype=np.bool_),
        global_multi=tuple(global_multi),
        global_multi_mask=np.asarray(global_multi_mask, dtype=np.bool_),
        temperature_features=np.zeros((len(indices), 7), dtype=np.float64),
        temperature_mask=np.zeros((len(indices), 7), dtype=np.bool_),
        exposure_s=np.asarray(data.exposure_s, dtype=np.float64)[list(indices)],
        exposure_known_mask=np.asarray(data.exposure_known_mask, dtype=np.bool_)[list(indices)],
        real_action_supervision_mask=np.zeros(len(indices), dtype=np.float64),
        source_manifest_hash=data.manifest_hash,
        source_split_hash=data.split_hash,
        adapter_sha=sha256_of(adapter_payload),
    )
    result.validate()
    return result


def v10_build_rho8_view_bundle(**kwargs: Any):
    """Use the verified V9-R2 implementation while preserving V10 weak-view semantics."""
    from ml_recon.physics_hybrid_v9.rho8_contract import build_rho8_view_bundle
    bundle = build_rho8_view_bundle(**kwargs)
    return bundle


def v10_count_mle_estimator(*args: Any, **kwargs: Any):
    """Construct the verified multi-start/bootstrap/Laplace count-MLE estimator."""
    from ml_recon.physics_hybrid_v9.count_mle_rho import V9CountLikelihoodRhoEstimator
    return V9CountLikelihoodRhoEstimator(*args, **kwargs)
