"""Registry-aware spatial, frequency, cycle, and temporal operation graph.

The graph replaces flat-index total variation for the opt-in V5 research path.
Legacy fixed-lag dynamics remain available to V4 when this feature is disabled.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from system_control.experiment_control.operation_points import OperationPointSpec, operation_registry_hash

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


REGISTRY_GRAPH_SCHEMA = "registry_operation_graph"
REGISTRY_GRAPH_VERSION = 1
EDGE_TYPES: Tuple[str, ...] = (
    "adjacent_group",
    "adjacent_row",
    "frequency_overlap",
    "shared_cell",
    "same_local_cross_cycle",
    "shared_aod_channel",
    "temporal_predecessor",
    "cycle_transition",
)
EDGE_TYPE_TO_ID = {name: index for index, name in enumerate(EDGE_TYPES)}
EDGE_FEATURE_NAMES: Tuple[str, ...] = (
    "delta_row",
    "delta_group",
    "delta_cycle",
    "delta_frequency_mhz",
    "shared_tone_count",
    "command_similarity",
    "physical_channel_relation",
    "temporal_distance",
)


def registry_strides(system_type: str) -> Dict[str, int]:
    if system_type == "3qubit":
        return {"group": 1, "row": 5}
    if system_type == "4qubit":
        return {"group": 1, "row": 4, "cycle": 64}
    raise ValueError(f"unsupported system_type: {system_type!r}")


def _command_vector(operation: OperationPointSpec) -> np.ndarray:
    values: List[float] = []
    for channel in sorted(operation.channel_commands):
        command = operation.channel_commands[channel]
        amp = [0.0 if value is None else float(value) for value in command.get("amplitudes", [])]
        phase = [0.0 if value is None else float(value) for value in command.get("phases_rad", [])]
        values.extend(amp)
        values.extend(np.sin(phase).tolist())
        values.extend(np.cos(phase).tolist())
    return np.asarray(values, dtype=np.float64)


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    size = max(left.size, right.size)
    if size == 0:
        return 0.0
    a = np.pad(left, (0, size - left.size))
    b = np.pad(right, (0, size - right.size))
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 0.0 else 0.0


@dataclass(frozen=True)
class RegistryOperationGraph:
    system_type: str
    operation_registry_hash: str
    source: np.ndarray
    target: np.ndarray
    edge_type: np.ndarray
    edge_features: np.ndarray
    edge_weights: np.ndarray
    schema: str = REGISTRY_GRAPH_SCHEMA
    version: int = REGISTRY_GRAPH_VERSION
    graph_hash: str = ""

    def __post_init__(self) -> None:
        source = np.asarray(self.source, dtype=np.int64)
        target = np.asarray(self.target, dtype=np.int64)
        edge_type = np.asarray(self.edge_type, dtype=np.int64)
        features = np.asarray(self.edge_features, dtype=np.float64)
        weights = np.asarray(self.edge_weights, dtype=np.float64)
        count = source.size
        if target.shape != (count,) or edge_type.shape != (count,) or weights.shape != (count,):
            raise ValueError("registry graph edge arrays disagree")
        if features.shape != (count, len(EDGE_FEATURE_NAMES)):
            raise ValueError("registry graph edge_features has the wrong shape")
        if np.any(source == target):
            raise ValueError("registry graph cannot contain self edges")
        if np.any(edge_type < 0) or np.any(edge_type >= len(EDGE_TYPES)):
            raise ValueError("unknown registry graph edge type")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "edge_type", edge_type)
        object.__setattr__(self, "edge_features", features)
        object.__setattr__(self, "edge_weights", weights)
        expected = self.compute_hash()
        if self.graph_hash and self.graph_hash != expected:
            raise ValueError("registry graph hash mismatch")
        object.__setattr__(self, "graph_hash", expected)

    def compute_hash(self) -> str:
        digest = hashlib.sha256()
        header = {
            "schema": self.schema,
            "version": int(self.version),
            "system_type": self.system_type,
            "operation_registry_hash": self.operation_registry_hash,
            "edge_types": list(EDGE_TYPES),
            "edge_feature_names": list(EDGE_FEATURE_NAMES),
        }
        digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        for value in (self.source, self.target, self.edge_type, self.edge_features, self.edge_weights):
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
            digest.update(array.tobytes())
        return digest.hexdigest()

    def edges_of_type(self, edge_type: str) -> np.ndarray:
        return np.flatnonzero(self.edge_type == EDGE_TYPE_TO_ID[edge_type])

    def manifest(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "version": int(self.version),
            "system_type": self.system_type,
            "operation_registry_hash": self.operation_registry_hash,
            "graph_hash": self.graph_hash,
            "edge_types": list(EDGE_TYPES),
            "edge_feature_names": list(EDGE_FEATURE_NAMES),
            "edge_count": int(self.source.size),
            "strides": registry_strides(self.system_type),
        }


def build_registry_graph(
    registry: Sequence[OperationPointSpec],
    *,
    edge_type_weights: Mapping[str, float] | None = None,
) -> RegistryOperationGraph:
    if not registry:
        raise ValueError("registry must not be empty")
    system_type = registry[0].system_type
    if any(item.system_type != system_type for item in registry):
        raise ValueError("registry contains mixed system types")
    weights = {name: 1.0 for name in EDGE_TYPES}
    if edge_type_weights:
        for name, value in edge_type_weights.items():
            if name not in weights:
                raise ValueError(f"unknown graph edge type {name!r}")
            weights[name] = float(value)

    commands = [_command_vector(operation) for operation in registry]
    edge_keys = set()
    source: List[int] = []
    target: List[int] = []
    types: List[int] = []
    features: List[List[float]] = []
    edge_weights: List[float] = []

    def add(left: int, right: int, edge_name: str) -> None:
        if left == right:
            return
        a, b = sorted((int(left), int(right)))
        key = (a, b, edge_name)
        if key in edge_keys:
            return
        edge_keys.add(key)
        op_a, op_b = registry[a], registry[b]
        shared_cells = len(set(op_a.cell_indices).intersection(op_b.cell_indices))
        shared_frequencies = len(set(round(value, 9) for value in op_a.frequencies_mhz).intersection(
            (round(value, 9) for value in op_b.frequencies_mhz),
        ))
        mean_a = float(np.mean(op_a.frequencies_mhz))
        mean_b = float(np.mean(op_b.frequencies_mhz))
        physical_relation = 1.0 if edge_name in {
            "frequency_overlap", "shared_cell", "same_local_cross_cycle", "shared_aod_channel",
        } else 0.0
        source.append(a)
        target.append(b)
        types.append(EDGE_TYPE_TO_ID[edge_name])
        features.append([
            float(op_b.row - op_a.row),
            float(op_b.group - op_a.group),
            float(op_b.cycle_index - op_a.cycle_index),
            mean_b - mean_a,
            float(max(shared_cells, shared_frequencies)),
            _cosine_similarity(commands[a], commands[b]),
            physical_relation,
            float(abs(op_b.global_index - op_a.global_index)),
        ])
        edge_weights.append(float(weights[edge_name]))

    # Local grid relations and exact temporal order.
    by_key = {(item.cycle_index, item.row, item.group): item.global_index for item in registry}
    for item in registry:
        current = int(item.global_index)
        for delta, edge_name in (((0, 0, 1), "adjacent_group"), ((0, 1, 0), "adjacent_row")):
            key = (item.cycle_index + delta[0], item.row + delta[1], item.group + delta[2])
            if key in by_key:
                add(current, by_key[key], edge_name)
        if current > 0:
            add(current - 1, current, "temporal_predecessor")

    # Shared-local and cycle-transition relations for the four conditional R views.
    by_local: Dict[int, List[int]] = {}
    for item in registry:
        by_local.setdefault(int(item.local_index), []).append(int(item.global_index))
    for indices in by_local.values():
        ordered = sorted(indices, key=lambda index: registry[index].cycle_index)
        if len(ordered) > 1:
            for left_index in range(len(ordered)):
                for right_index in range(left_index + 1, len(ordered)):
                    add(ordered[left_index], ordered[right_index], "same_local_cross_cycle")
            for left, right in zip(ordered[:-1], ordered[1:]):
                add(left, right, "cycle_transition")

    # Frequency/cell relations are derived from the registry, not flat index.
    for left in range(len(registry)):
        cells_left = set(registry[left].cell_indices)
        frequencies_left = set(round(value, 9) for value in registry[left].frequencies_mhz)
        for right in range(left + 1, len(registry)):
            cells_shared = cells_left.intersection(registry[right].cell_indices)
            frequencies_shared = frequencies_left.intersection(round(value, 9) for value in registry[right].frequencies_mhz)
            if frequencies_shared:
                add(left, right, "frequency_overlap")
            if cells_shared:
                add(left, right, "shared_cell")
            # The physical X-channel path is shared inside a cycle; keep this
            # graph sparse by connecting grid neighbors only.
            if registry[left].cycle_index == registry[right].cycle_index and (
                (registry[left].row == registry[right].row and abs(registry[left].group - registry[right].group) == 1)
                or (registry[left].group == registry[right].group and abs(registry[left].row - registry[right].row) == 1)
            ):
                add(left, right, "shared_aod_channel")

    return RegistryOperationGraph(
        system_type=system_type,
        operation_registry_hash=operation_registry_hash(registry),
        source=np.asarray(source, dtype=np.int64),
        target=np.asarray(target, dtype=np.int64),
        edge_type=np.asarray(types, dtype=np.int64),
        edge_features=np.asarray(features, dtype=np.float64),
        edge_weights=np.asarray(edge_weights, dtype=np.float64),
    )


def graph_action_smoothness(action: Any, graph: RegistryOperationGraph, *, reduction: str = "mean") -> Any:
    """Return ``sum_edges w ||a_o-a_o'||^2`` for NumPy or torch arrays."""
    if torch is not None and isinstance(action, torch.Tensor):
        source = torch.as_tensor(graph.source, dtype=torch.long, device=action.device)
        target = torch.as_tensor(graph.target, dtype=torch.long, device=action.device)
        weight = torch.as_tensor(graph.edge_weights, dtype=action.dtype, device=action.device)
        difference = action[..., source, :] - action[..., target, :]
        per_edge = difference.square().sum(dim=-1) * weight
        if reduction == "sum":
            return per_edge.sum()
        if reduction == "none":
            return per_edge
        return per_edge.sum() / torch.clamp(weight.sum(), min=torch.finfo(action.dtype).eps)
    array = np.asarray(action, dtype=float)
    difference = array[..., graph.source, :] - array[..., graph.target, :]
    per_edge = np.sum(np.square(difference), axis=-1) * graph.edge_weights
    if reduction == "sum":
        return float(np.sum(per_edge))
    if reduction == "none":
        return per_edge
    return float(np.sum(per_edge) / max(float(np.sum(graph.edge_weights)), np.finfo(float).eps))


__all__ = [
    "RegistryOperationGraph",
    "EDGE_TYPES",
    "EDGE_FEATURE_NAMES",
    "build_registry_graph",
    "graph_action_smoothness",
    "registry_strides",
]
