"""V9 operation registry (charter sections 17.6, 20.4).

The flat hardware operation order is authoritative and comes from the operation
registry / command table / AWG sequence order — never from array reshape. 3Q order is
11 special + 64 informational (75 total); 4Q is four cycles (256 total). The registry
carries an immutable hash so any proposal binds to the exact operation layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .contracts import sha256_of_dict

SYSTEM_OPERATION_COUNTS = {"3qubit": 75, "4qubit": 256}
SYSTEM_INTERNAL_SHAPES = {"3qubit": (15, 5, 3), "4qubit": (4, 16, 4, 4)}


@dataclass(frozen=True)
class V9OperationRegistryV1:
    """Authoritative flat operation order for a system."""

    system_type: str
    n_operations: int
    flat_order: Tuple[str, ...]  # operation ids in hardware/AWG order
    internal_shape: Tuple[int, ...]
    special_ids: Tuple[str, ...] = ()  # 3Q: 11 special
    informational_ids: Tuple[str, ...] = ()  # 3Q: 64 informational
    cycles: Tuple[str, ...] = ()  # 4Q: four cycle ids
    command_table_hash: str = ""

    def registry_hash(self) -> str:
        payload = {
            "system_type": self.system_type,
            "n_operations": self.n_operations,
            "flat_order": list(self.flat_order),
            "internal_shape": list(self.internal_shape),
            "special_ids": list(self.special_ids),
            "informational_ids": list(self.informational_ids),
            "cycles": list(self.cycles),
            "command_table_hash": self.command_table_hash,
        }
        return sha256_of_dict(payload)

    def validate(self) -> None:
        if self.system_type not in SYSTEM_OPERATION_COUNTS:
            raise ValueError(f"system_type must be one of {list(SYSTEM_OPERATION_COUNTS)}")
        if self.n_operations != SYSTEM_OPERATION_COUNTS[self.system_type]:
            raise ValueError("n_operations mismatch with system")
        if len(self.flat_order) != self.n_operations:
            raise ValueError("flat_order length must equal n_operations")
        if self.internal_shape != SYSTEM_INTERNAL_SHAPES[self.system_type]:
            raise ValueError("internal_shape mismatch with system")
        if self.system_type == "3qubit":
            if len(self.special_ids) != 11 or len(self.informational_ids) != 64:
                raise ValueError("3Q registry requires 11 special + 64 informational")
            if tuple(self.flat_order[:11]) != self.special_ids:
                raise ValueError("3Q flat order must start with the 11 special operations")
            if tuple(self.flat_order[11:]) != self.informational_ids:
                raise ValueError("3Q flat order must continue with the 64 informational operations")
        else:  # 4qubit
            if len(self.cycles) != 4:
                raise ValueError("4Q registry requires four cycles")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_type": self.system_type,
            "n_operations": self.n_operations,
            "flat_order": list(self.flat_order),
            "internal_shape": list(self.internal_shape),
            "special_ids": list(self.special_ids),
            "informational_ids": list(self.informational_ids),
            "cycles": list(self.cycles),
            "command_table_hash": self.command_table_hash,
            "registry_hash": self.registry_hash(),
        }


def build_default_registry(system_type: str, command_table_hash: str = "") -> V9OperationRegistryV1:
    """Build the default registry for a system.

    3Q: 11 special operations (``sp00``..``sp10``) followed by 64 informational
    operations (``info00``..``info63``) in AWG order. 4Q: four cycles
    (``cycle0``..``cycle3``) each holding 64 operations in flat order.
    """
    if system_type == "3qubit":
        special = tuple(f"sp{i:02d}" for i in range(11))
        informational = tuple(f"info{i:02d}" for i in range(64))
        flat = special + informational
        registry = V9OperationRegistryV1(
            system_type="3qubit", n_operations=75, flat_order=flat,
            internal_shape=(15, 5, 3), special_ids=special,
            informational_ids=informational, command_table_hash=command_table_hash,
        )
        registry.validate()
        return registry
    if system_type == "4qubit":
        cycles = tuple(f"cycle{i}" for i in range(4))
        flat = tuple(f"{c}_op{j:02d}" for c in cycles for j in range(64))
        registry = V9OperationRegistryV1(
            system_type="4qubit", n_operations=256, flat_order=flat,
            internal_shape=(4, 16, 4, 4), cycles=cycles,
            command_table_hash=command_table_hash,
        )
        registry.validate()
        return registry
    raise ValueError(f"system_type must be 3qubit or 4qubit")


def operation_mask_for_groups(
    registry: V9OperationRegistryV1,
    hidden_groups: Sequence[str],
) -> np.ndarray:
    """Boolean mask (True = visible) hiding whole registered groups.

    ``hidden_groups`` are prefixes of operation ids, e.g. ``("info",)`` hides all
    informational operations for 3Q, or ``("cycle1",)`` hides a 4Q cycle.
    """
    mask = np.ones(registry.n_operations, dtype=bool)
    for i, op_id in enumerate(registry.flat_order):
        for group in hidden_groups:
            if op_id.startswith(group):
                mask[i] = False
                break
    return mask


def registry_hash_mismatch(a: V9OperationRegistryV1, b: V9OperationRegistryV1) -> bool:
    """Fail-closed registry identity check."""
    return a.registry_hash() != b.registry_hash()
