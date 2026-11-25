"""Executable multi-command dataset for the V6 joint feedback pipeline.

Samples store only immutable command-table references.  Static command tables
are hash-deduplicated and contain compact physical commands, never 1 GHz raw
waveforms.  Counterfactual pairs share target/current-state error while using
different command contexts and inverse-teacher actions.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Sequence, Tuple

import numpy as np

from system_control.experiment_control.data_protocol import atomic_write_json, stable_json_hash

from .command_contract import (
    COMMAND_CHANNELS,
    CommandTableCache,
    PhysicsHybridCommandTableV1,
)
from .command_normalization import fit_command_table_normalization
from .projective_action import (
    action_dimensions,
    amplitude_to_coordinates,
    coordinates_to_amplitude,
    coordinates_to_phase,
    phase_to_coordinates,
)


JOINT_DATASET_SCHEMA = "PhysicsHybridV6JointFeedbackDataset"
JOINT_DATASET_VERSION = 1
SAMPLES_FILE_NAME = "joint_samples.npz"
MANIFEST_FILE_NAME = "manifest.json"


@dataclass(frozen=True)
class CommandPerturbationFamilyV1:
    """Bounded compact-command perturbations in explicit physical units."""

    global_relative_amplitude_fraction: float = 0.005
    global_relative_phase_rad: float = 0.02
    per_channel_power_compression_fraction: float = 0.02
    absolute_frequency_shift_mhz: float = 0.5
    write_read_path_imbalance_fraction: float = 0.005
    cycle_read_variation_rad: float = 0.01
    temperature_delta_c: float = 2.0
    maximum_time_blocks: int = 4
    pairwise_tone_interaction_rad: float = 0.005
    calibration_residual_fraction: float = 0.003

    def sample(self, rng: np.random.Generator, family_index: int) -> Dict[str, Any]:
        return {
            "family_index": int(family_index),
            "global_relative_amplitude_fraction": float(
                rng.uniform(-self.global_relative_amplitude_fraction, self.global_relative_amplitude_fraction)
            ),
            "global_relative_phase_rad": float(
                rng.uniform(-self.global_relative_phase_rad, self.global_relative_phase_rad)
            ),
            "per_channel_power_compression_fraction": float(
                rng.uniform(0.0, self.per_channel_power_compression_fraction)
            ),
            "absolute_frequency_shift_mhz": float(
                rng.uniform(-self.absolute_frequency_shift_mhz, self.absolute_frequency_shift_mhz)
            ),
            "write_read_path_imbalance_fraction": float(
                rng.uniform(-self.write_read_path_imbalance_fraction, self.write_read_path_imbalance_fraction)
            ),
            "cycle_read_variation_rad": float(
                rng.uniform(-self.cycle_read_variation_rad, self.cycle_read_variation_rad)
            ),
            "temperature_delta_c": float(
                rng.uniform(-self.temperature_delta_c, self.temperature_delta_c)
            ),
            "time_block_index": int(rng.integers(0, max(1, self.maximum_time_blocks))),
            "pairwise_tone_interaction_rad": float(
                rng.uniform(-self.pairwise_tone_interaction_rad, self.pairwise_tone_interaction_rad)
            ),
            "calibration_residual_fraction": float(
                rng.uniform(-self.calibration_residual_fraction, self.calibration_residual_fraction)
            ),
            "units": {
                "amplitude": "relative_fraction",
                "phase": "rad",
                "frequency": "MHz",
                "temperature": "degC",
            },
        }

    def apply(
        self,
        base_table: PhysicsHybridCommandTableV1,
        perturbation: Mapping[str, Any],
    ) -> PhysicsHybridCommandTableV1:
        static = {
            name: np.array(value, copy=True)
            for name, value in base_table.static_arrays.items()
        }
        global_log_gain = float(np.log1p(perturbation["global_relative_amplitude_fraction"]))
        path_imbalance = float(np.log1p(perturbation["write_read_path_imbalance_fraction"]))
        calibration_residual = float(perturbation["calibration_residual_fraction"])
        phase_offset = float(perturbation["global_relative_phase_rad"])
        pair_offset = float(perturbation["pairwise_tone_interaction_rad"])
        frequency_shift = float(perturbation["absolute_frequency_shift_mhz"])
        power_compression = float(perturbation["per_channel_power_compression_fraction"])

        write_indices = [COMMAND_CHANNELS.index(channel) for channel in COMMAND_CHANNELS if channel.startswith("W_")]
        read_indices = [COMMAND_CHANNELS.index(channel) for channel in COMMAND_CHANNELS if channel.startswith("R_")]
        for channel_index in write_indices:
            static["x_log_amplitude"][..., channel_index] += global_log_gain + path_imbalance + calibration_residual
            static["y_amplitude"][..., channel_index] *= np.exp(global_log_gain + path_imbalance + calibration_residual)
            _rotate_phase(static, channel_index, phase_offset + pair_offset)
        for channel_index in read_indices:
            static["x_log_amplitude"][..., channel_index] += global_log_gain - path_imbalance - calibration_residual
            static["y_amplitude"][..., channel_index] *= np.exp(global_log_gain - path_imbalance - calibration_residual)
            _rotate_phase(static, channel_index, -phase_offset + pair_offset)

        static["absolute_frequency_mhz"] += frequency_shift
        static["y_frequency_mhz"] += frequency_shift
        static["per_channel_rf_power"] *= max(0.0, 1.0 - power_compression)
        static["total_rf_power"] = np.sum(static["per_channel_rf_power"], axis=-1)
        if base_table.system_type == "4qubit":
            read_variation = float(perturbation["cycle_read_variation_rad"])
            cycle_offsets = np.linspace(-read_variation, read_variation, 4)
            for cycle_index, cycle_offset in enumerate(cycle_offsets):
                for channel_index in read_indices:
                    _rotate_phase(static, channel_index, float(cycle_offset), cycle_index=cycle_index)

        perturbation_hash = stable_json_hash(dict(perturbation))
        return dataclasses.replace(
            base_table,
            matrix_snapshot_hash=stable_json_hash({
                "base": base_table.matrix_snapshot_hash,
                "command_perturbation": perturbation_hash,
            }),
            awg_state_hash=stable_json_hash({
                "research_compact_command": True,
                "uploaded": False,
                "perturbation": perturbation_hash,
            }),
            static_arrays=static,
            table_hash="",
        )


def _rotate_phase(
    static: Dict[str, np.ndarray],
    channel_index: int,
    offset_rad: float,
    *,
    cycle_index: int | None = None,
) -> None:
    prefix = () if cycle_index is None else (int(cycle_index),)
    for sin_name, cos_name in (("x_phase_sin", "x_phase_cos"), ("y_phase_sin", "y_phase_cos")):
        sin_values = static[sin_name][prefix + (..., channel_index)]
        cos_values = static[cos_name][prefix + (..., channel_index)]
        phase = np.arctan2(sin_values, cos_values) + float(offset_rad)
        static[sin_name][prefix + (..., channel_index)] = np.sin(phase)
        static[cos_name][prefix + (..., channel_index)] = np.cos(phase)


def deduplicate_command_tables(
    tables: Sequence[PhysicsHybridCommandTableV1],
) -> Tuple[Tuple[PhysicsHybridCommandTableV1, ...], Tuple[int, ...]]:
    unique: list[PhysicsHybridCommandTableV1] = []
    indices: Dict[str, int] = {}
    remap = []
    for table in tables:
        if table.table_hash not in indices:
            indices[table.table_hash] = len(unique)
            unique.append(table)
        remap.append(indices[table.table_hash])
    return tuple(unique), tuple(remap)


def build_joint_feedback_dataset(
    base_table: PhysicsHybridCommandTableV1,
    output_dir: str | Path,
    *,
    command_table_count: int = 4,
    sample_count: int = 32,
    seed: int = 20260805,
    perturbation_family: CommandPerturbationFamilyV1 | None = None,
) -> Path:
    """Generate a reproducible compact dataset with paired counterfactuals."""

    if int(command_table_count) < 2:
        raise ValueError("joint feedback training requires at least two command tables")
    if int(sample_count) < 2:
        raise ValueError("joint feedback training requires at least two samples")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(seed))
    family = perturbation_family or CommandPerturbationFamilyV1()
    perturbations: list[Dict[str, Any]] = [{
        **family.sample(np.random.default_rng(int(seed)), 0),
        "global_relative_amplitude_fraction": 0.0,
        "global_relative_phase_rad": 0.0,
        "per_channel_power_compression_fraction": 0.0,
        "absolute_frequency_shift_mhz": 0.0,
        "write_read_path_imbalance_fraction": 0.0,
        "cycle_read_variation_rad": 0.0,
        "temperature_delta_c": 0.0,
        "time_block_index": 0,
        "pairwise_tone_interaction_rad": 0.0,
        "calibration_residual_fraction": 0.0,
    }]
    tables = [base_table]
    while len(tables) < int(command_table_count):
        perturbation = family.sample(rng, len(perturbations))
        candidate = family.apply(base_table, perturbation)
        if all(candidate.table_hash != table.table_hash for table in tables):
            tables.append(candidate)
            perturbations.append(perturbation)
    unique_tables, _ = deduplicate_command_tables(tables)
    if len(unique_tables) != int(command_table_count):
        raise RuntimeError("command perturbation family failed to produce unique tables")

    table_records: Dict[str, Any] = {}
    for index, table in enumerate(unique_tables):
        manifest_path = table.save(root / "command_tables")
        table_records[str(index)] = {
            "table_id": table.table_id,
            "table_hash": table.table_hash,
            "manifest_path": str(manifest_path.relative_to(root)).replace("\\", "/"),
            "perturbation": perturbations[index],
        }
    normalization = fit_command_table_normalization(unique_tables)
    normalization_path = root / "command_normalization.json"
    atomic_write_json(normalization_path, normalization)

    arrays = _generate_samples(unique_tables, perturbations, int(sample_count), rng)
    samples_path = root / SAMPLES_FILE_NAME
    np.savez_compressed(samples_path, **arrays)
    samples_sha256 = hashlib.sha256(samples_path.read_bytes()).hexdigest()
    dataset_identity = {
        "schema": JOINT_DATASET_SCHEMA,
        "schema_version": JOINT_DATASET_VERSION,
        "system_type": base_table.system_type,
        "seed": int(seed),
        "sample_count": int(sample_count),
        "command_table_hashes": [table.table_hash for table in unique_tables],
        "normalization_sha256": normalization["sha256"],
        "samples_sha256": samples_sha256,
    }
    manifest = {
        **dataset_identity,
        "dataset_sha256": stable_json_hash(dataset_identity),
        "samples_path": SAMPLES_FILE_NAME,
        "command_tables": table_records,
        "command_normalization": {
            "path": normalization_path.name,
            "sha256": normalization["sha256"],
        },
        "static_table_repeated_in_samples": False,
        "sample_payload_policy": "table_reference_only",
        "counterfactual_pair_count": int(len(np.unique(arrays["counterfactual_pair_id"]))),
        "held_out_axes": [
            "command_group", "frequency_group", "cycle_index", "time_block_index",
        ],
        "raw_waveform_loaded": False,
        "research_only": True,
        "promotion_ready": False,
    }
    manifest_path = root / MANIFEST_FILE_NAME
    atomic_write_json(manifest_path, manifest)
    return manifest_path


def _generate_samples(
    tables: Sequence[PhysicsHybridCommandTableV1],
    perturbations: Sequence[Mapping[str, Any]],
    sample_count: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    system_type = tables[0].system_type
    cell_count = 3 if system_type == "3qubit" else 4
    dimensions = action_dimensions(system_type)
    rows: Dict[str, list[Any]] = {
        name: [] for name in (
            "command_table_index", "target_amplitude", "target_phase_rad",
            "latent_amplitude_error", "latent_phase_error", "observation_features",
            "point_counts", "teacher_action", "projected_teacher_action",
            "teacher_actionable_mask", "amplitude_empirical_evidence_mask",
            "counterfactual_pair_id", "command_group", "frequency_group",
            "held_out_cycle_index", "time_block_index", "temperature_c",
            "exposure_s", "holdout_command_group", "holdout_frequency_group",
            "holdout_cycle", "holdout_time_block",
        )
    }
    sample_index = 0
    pair_index = 0
    while sample_index < sample_count:
        target_amplitude = rng.uniform(0.5, 1.5, size=cell_count)
        target_amplitude /= np.linalg.norm(target_amplitude)
        target_phase = rng.uniform(-np.pi, np.pi, size=cell_count)
        target_phase -= np.mean(target_phase)
        latent_amp = rng.normal(0.0, 0.015, size=dimensions["amplitude"])
        latent_phase = rng.normal(0.0, 0.05, size=dimensions["phase"])
        table_a = pair_index % len(tables)
        table_b = (table_a + 1 + pair_index % (len(tables) - 1)) % len(tables)
        for table_index in (table_a, table_b):
            if sample_index >= sample_count:
                break
            perturbation = perturbations[table_index]
            teacher = _synthetic_inverse_teacher(
                system_type, latent_amp, latent_phase, perturbation,
            )
            current_amplitude = coordinates_to_amplitude(
                amplitude_to_coordinates(target_amplitude) + latent_amp
            )
            current_phase = coordinates_to_phase(
                phase_to_coordinates(target_phase) + latent_phase
            )
            temperature = 25.0 + float(perturbation["temperature_delta_c"])
            time_block = int(perturbation["time_block_index"])
            observation_features = np.concatenate((
                current_amplitude,
                np.sin(current_phase),
                np.asarray([(temperature - 25.0) / 10.0, time_block / 3.0]),
            ))
            rows["command_table_index"].append(table_index)
            rows["target_amplitude"].append(target_amplitude)
            rows["target_phase_rad"].append(target_phase)
            rows["latent_amplitude_error"].append(latent_amp)
            rows["latent_phase_error"].append(latent_phase)
            rows["observation_features"].append(observation_features)
            rows["point_counts"].append(_synthetic_point_counts(
                tables[table_index], current_amplitude, current_phase, perturbation,
            ))
            rows["teacher_action"].append(teacher)
            rows["projected_teacher_action"].append(teacher)
            rows["teacher_actionable_mask"].append(np.ones(dimensions["total"], dtype=bool))
            rows["amplitude_empirical_evidence_mask"].append(
                np.zeros(dimensions["amplitude"], dtype=bool)
            )
            rows["counterfactual_pair_id"].append(pair_index)
            rows["command_group"].append(table_index)
            rows["frequency_group"].append(int(abs(float(perturbation["absolute_frequency_shift_mhz"])) * 10.0))
            rows["held_out_cycle_index"].append(sample_index % (4 if system_type == "4qubit" else 1))
            rows["time_block_index"].append(time_block)
            rows["temperature_c"].append(temperature)
            rows["exposure_s"].append(1.0)
            rows["holdout_command_group"].append(table_index == len(tables) - 1)
            maximum_frequency_group = max(
                int(abs(float(item["absolute_frequency_shift_mhz"])) * 10.0)
                for item in perturbations
            )
            rows["holdout_frequency_group"].append(
                rows["frequency_group"][-1] == maximum_frequency_group
            )
            rows["holdout_cycle"].append(
                system_type == "4qubit" and rows["held_out_cycle_index"][-1] == 3
            )
            rows["holdout_time_block"].append(
                time_block == max(int(item["time_block_index"]) for item in perturbations)
            )
            sample_index += 1
        pair_index += 1
    return {name: np.asarray(values) for name, values in rows.items()}


def _synthetic_inverse_teacher(
    system_type: str,
    latent_amplitude_error: np.ndarray,
    latent_phase_error: np.ndarray,
    perturbation: Mapping[str, Any],
) -> np.ndarray:
    dimensions = action_dimensions(system_type)
    amp_command = float(perturbation["write_read_path_imbalance_fraction"])
    phase_command = float(perturbation["global_relative_phase_rad"])
    amp = -np.asarray(latent_amplitude_error, dtype=float) - np.linspace(
        -amp_command, amp_command, dimensions["amplitude"],
    )
    phase = -np.asarray(latent_phase_error, dtype=float) - np.linspace(
        -phase_command, phase_command, dimensions["phase"],
    )
    return np.concatenate((
        np.clip(amp, -np.log1p(0.03), np.log1p(0.03)),
        np.clip(phase, -0.20, 0.20),
    ))


def _synthetic_point_counts(
    table: PhysicsHybridCommandTableV1,
    current_amplitude: np.ndarray,
    current_phase: np.ndarray,
    perturbation: Mapping[str, Any],
) -> np.ndarray:
    basis_amplitude = np.array(table.static_arrays["basis_amplitude"], dtype=float, copy=True)
    basis_phase = np.arctan2(
        np.asarray(table.static_arrays["basis_phase_sin"], dtype=float),
        np.asarray(table.static_arrays["basis_phase_cos"], dtype=float),
    )
    basis_amplitude /= np.maximum(np.linalg.norm(basis_amplitude, axis=-1, keepdims=True), 1e-12)
    state = current_amplitude * np.exp(1j * current_phase)
    projector = basis_amplitude * np.exp(1j * basis_phase)
    born = np.abs(np.sum(np.conjugate(projector) * state, axis=-1)) ** 2
    command_scale = 1.0 - float(perturbation["per_channel_power_compression_fraction"])
    signal = 100.0 * command_scale * np.exp(0.03 * born)
    idler = 80.0 * command_scale * np.exp(0.02 * born)
    coin = 25.0 * command_scale * born + 1e-3
    return np.stack((signal, idler, coin), axis=-1).reshape(-1, 3)


class JointFeedbackDataset:
    """Memory-mapped sample arrays plus hash-verified command-table cache."""

    def __init__(self, dataset_dir: str | Path, *, maximum_cached_tables: int = 8) -> None:
        self.root = Path(dataset_dir)
        self.manifest = json.loads((self.root / MANIFEST_FILE_NAME).read_text(encoding="utf-8"))
        if self.manifest.get("schema") != JOINT_DATASET_SCHEMA:
            raise ValueError("unsupported joint feedback dataset")
        samples_path = self.root / str(self.manifest["samples_path"])
        if hashlib.sha256(samples_path.read_bytes()).hexdigest() != self.manifest["samples_sha256"]:
            raise ValueError("joint feedback samples hash mismatch")
        self.arrays = np.load(samples_path, mmap_mode="r", allow_pickle=False)
        self.command_cache = CommandTableCache(max_tables=maximum_cached_tables)
        self.table_records = dict(self.manifest["command_tables"])
        normalization_path = self.root / self.manifest["command_normalization"]["path"]
        self.normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
        if self.normalization.get("sha256") != self.manifest["normalization_sha256"]:
            raise ValueError("joint command normalization hash mismatch")

    def __len__(self) -> int:
        return int(self.arrays["command_table_index"].shape[0])

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return {name: np.asarray(self.arrays[name][index]) for name in self.arrays.files}

    def command_table(self, table_index: int) -> PhysicsHybridCommandTableV1:
        record = self.table_records[str(int(table_index))]
        manifest_path = self.root / record["manifest_path"]
        return self.command_cache.load(manifest_path)

    def counterfactual_pairs(self) -> Iterator[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]:
        by_pair: Dict[int, list[int]] = {}
        for index, pair_id in enumerate(np.asarray(self.arrays["counterfactual_pair_id"], dtype=int)):
            by_pair.setdefault(int(pair_id), []).append(index)
        for indices in by_pair.values():
            if len(indices) >= 2:
                yield self[indices[0]], self[indices[1]]

    def close(self) -> None:
        self.arrays.close()
        self.command_cache.close()

    def __enter__(self) -> "JointFeedbackDataset":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


__all__ = [
    "CommandPerturbationFamilyV1",
    "JointFeedbackDataset",
    "build_joint_feedback_dataset",
    "deduplicate_command_tables",
    "JOINT_DATASET_SCHEMA",
    "JOINT_DATASET_VERSION",
]
