"""V2.4 hardware measurement basis (DECISIONS C1, H1, J18).

The V2.4 shards store `flat_order` labels but no operation descriptors. The
basis is rebuilt from the legacy construction in
`system_control/qubit_recon_files/basis_configs.py` (physics only):

3Q, 75 operations: `sp00..sp10` are the 11 special entries (three single-cell
reads, pairs (1,2), (1,3), (2,3) at relative phase 0 and pi, all three cells at
phases (0, 0, 0) and (0, pi/2, pi)); `info00..info63` are the 64 tensor-product
phase entries over {0, pi/2, pi, 3pi/2}^3 with amplitudes 1.0 or 0.6 from the
three low bits of the entry index.

4Q, 256 operations: the tensor-product entries over {0, pi/2, pi, 3pi/2}^4 with
amplitudes from the four low bits; `cycle{c}_op{e}` is entry 64 c + e.

beta is the amplitude vector normalized to unit read power. theta is the stored
read phase; phase_sign (-1 in the lab) is applied once, in the forward map.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

PHASE_GRID = (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0)
LOW_AMPLITUDE = 0.6


def _pack(labels, amps, phases) -> dict:
    amps = np.asarray(amps, dtype=np.float64)
    phases = np.asarray(phases, dtype=np.float64)
    beta = amps / np.linalg.norm(amps, axis=-1, keepdims=True)
    return {"labels": list(labels), "beta": beta, "theta": phases, "n_settings": len(labels), "n_modes": amps.shape[1]}


def _info_amplitudes(index: int, n_modes: int) -> list[float]:
    bits = index % (2**n_modes)
    return [1.0 if (bits >> q) & 1 else LOW_AMPLITUDE for q in range(n_modes)]


def three_qubit_basis() -> dict:
    amps, phases, labels = [], [], []
    for cell in range(3):
        amp = [0.0, 0.0, 0.0]
        amp[cell] = 1.0
        amps.append(amp)
        phases.append([0.0, 0.0, 0.0])
    for i, j in ((0, 1), (0, 2), (1, 2)):
        for relative in (0.0, math.pi):
            amp = [0.0, 0.0, 0.0]
            phase = [0.0, 0.0, 0.0]
            amp[i] = amp[j] = 1.0
            phase[j] = relative
            amps.append(amp)
            phases.append(phase)
    amps.append([1.0, 1.0, 1.0])
    phases.append([0.0, 0.0, 0.0])
    amps.append([1.0, 1.0, 1.0])
    phases.append([0.0, math.pi / 2.0, math.pi])
    labels.extend(f"sp{index:02d}" for index in range(11))
    for index, triple in enumerate(itertools.product(PHASE_GRID, repeat=3)):
        if index >= 64:
            break
        amps.append(_info_amplitudes(index, 3))
        phases.append(list(triple))
        labels.append(f"info{index:02d}")
    return _pack(labels, amps, phases)


def four_qubit_basis() -> dict:
    amps, phases, labels = [], [], []
    for index, quad in enumerate(itertools.product(PHASE_GRID, repeat=4)):
        amps.append(_info_amplitudes(index, 4))
        phases.append(list(quad))
        labels.append(f"cycle{index // 64}_op{index % 64:02d}")
    return _pack(labels, amps, phases)


def hardware_basis(n_modes: int) -> dict:
    if n_modes == 3:
        return three_qubit_basis()
    if n_modes == 4:
        return four_qubit_basis()
    raise ValueError(f"the V2.4 hardware basis exists for N = 3 and 4, got {n_modes}")
