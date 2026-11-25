"""Physical basis descriptors and small-N local-Pauli registries."""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Optional, Tuple

LOCAL_PAULI_AXES = ("X", "Y", "Z")


@dataclass(frozen=True)
class V10BasisDescriptor:
    setting_id: str
    N: int
    subset: Tuple[int, ...]
    pauli_axes: Tuple[str, ...]
    projector_amplitudes: Tuple[float, ...] = ()
    projector_phases: Tuple[float, ...] = ()
    aod_positions: Tuple[Tuple[float, float], ...] = ()
    physical_group: str = ""

    @property
    def setting_hash_embedding(self) -> Tuple[float, ...]:
        digest = hashlib.sha256(self.setting_id.encode("utf-8")).digest()
        return tuple((b / 127.5) - 1.0 for b in digest[:8])

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def full_local_pauli_registry(N: int) -> Tuple[V10BasisDescriptor, ...]:
    if N not in (3, 4):
        raise ValueError("full local-Pauli measurement is reserved for N=3,4")
    result = []
    for axes in itertools.product(LOCAL_PAULI_AXES, repeat=N):
        sid = "pauli_" + "".join(axes)
        result.append(V10BasisDescriptor(sid, N, tuple(range(N)), tuple(axes)))
    return tuple(result)


def descriptor_vector(d: V10BasisDescriptor, max_modes: int = 256) -> Tuple[float, ...]:
    """Fixed-width physics vector; no learned operation-index embedding."""
    axis_code = {"I": 0.0, "X": 1.0, "Y": 2.0, "Z": 3.0}
    subset_fraction = len(d.subset) / max(d.N, 1)
    axis_hist = tuple(d.pauli_axes.count(a) / max(len(d.pauli_axes), 1) for a in ("X", "Y", "Z"))
    index_moments = (
        sum(d.subset) / max(len(d.subset) * max_modes, 1),
        max(d.subset, default=0) / max_modes,
    )
    return (d.N / max_modes, subset_fraction, *axis_hist, *index_moments,
            *d.setting_hash_embedding)
