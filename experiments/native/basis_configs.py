"""Pure reference basis used to validate the native analyzer registry."""

import math
from itertools import product


def _entry(amp, phase, label):
    return {
        "amp": [float(value) for value in amp],
        "phase": [float(value) for value in phase],
        "label": label,
    }


def build_3qubit_basis_config():
    entries = []
    pair_indices = [(0, 1), (0, 2), (1, 2)]

    # Keep 11 special focus bases (single-qubit excitations, pair special phases,
    # and two triple modes) unchanged so downstream code relying on these
    # labels/modes remains compatible.
    for idx in range(3):
        amp = [0.0, 0.0, 0.0]
        amp[idx] = 1.0
        entries.append(_entry(amp, [0.0, 0.0, 0.0], f"single_{idx + 1}"))

    for i, j in pair_indices:
        for phase_value, suffix in [(0.0, "0"), (math.pi, "pi")]:
            amp = [0.0, 0.0, 0.0]
            phase = [0.0, 0.0, 0.0]
            amp[i] = 1.0
            amp[j] = 1.0
            phase[j] = phase_value
            entries.append(_entry(amp, phase, f"pair_{i + 1}{j + 1}_{suffix}"))

    entries.append(_entry([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], "triple_all_0"))
    entries.append(
        _entry([1.0, 1.0, 1.0], [0.0, math.pi / 2.0, math.pi], "triple_all_phase")
    )

    # Build 64 informational entries using tensor-product of single-qubit
    # phase grid, but vary amplitudes moderately (1.0 vs 0.6) to avoid
    # degenerate all-ones vectors.
    phase_grid = [0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0]
    info_entries = []
    for idx, (p1, p2, p3) in enumerate(product(phase_grid, repeat=3)):
        bits = idx % 8  # three-bit amplitude pattern cycles through 0..7
        amp = [1.0 if ((bits >> q) & 1) else 0.6 for q in range(3)]
        info_entries.append(_entry(amp, [p1, p2, p3], f"info_{idx + 1:03d}"))

    # Append the 64 info entries after the 11 special entries -> total 75 entries
    entries.extend(info_entries[:64])

    return {
        "system_type": "3qubit",
        "operation_duration_us": 3,
        "scan_order": "row-major",
        "entries": entries,
    }


def _build_4qubit_full_entries():
    """Generate 256 tensor-product full-basis entries from single-qubit
    phase grid while varying amplitudes using a small set of templates.
    """
    entries = []
    phase_grid = [0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0]
    for idx, phases in enumerate(product(phase_grid, repeat=4)):
        # Use 4-bit pattern to vary amplitudes between 1.0 and 0.6
        bits = idx % 16
        amp = [1.0 if ((bits >> q) & 1) else 0.6 for q in range(4)]
        entries.append(_entry(amp, [float(ph) for ph in phases], f"full_{idx + 1:03d}"))
    return entries


def build_4qubit_basis_config():
    # Create 256 informationally-complete tensor-product bases and split
    # them into 4 cycles of 64 entries each (keeps downstream code expecting
    # a 'cycles' structure).
    all_entries = _build_4qubit_full_entries()
    cycles = []
    for cycle_index in range(4):
        start = cycle_index * 64
        cycles.append(
            {
                "cycle": cycle_index + 1,
                "entries": all_entries[start : start + 64],
            }
        )

    return {
        "system_type": "4qubit",
        "operation_duration_us": 4,
        "scan_order": "row-major",
        "cycles": cycles,
    }
