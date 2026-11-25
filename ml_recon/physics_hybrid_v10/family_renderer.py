"""Operational 3Q/4Q fine-family renderer."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .synthetic_manifest import V10CounterBasedRng, V10SyntheticSampleKey
from ml_recon.physics_hybrid_v9.full_family_renderer import family_labels, render_family_state

FINE_FAMILY_COUNTS = {3: 14, 4: 24}


class V10FamilyRenderer:
    def render(self, key: V10SyntheticSampleKey, rng: V10CounterBasedRng) -> Dict[str, object]:
        if key.N not in FINE_FAMILY_COUNTS:
            raise ValueError("fine-family rendering is defined only for N=3,4")
        stream = rng.stream("family")
        labels = family_labels(key.N)
        family_id = int(stream.integers(FINE_FAMILY_COUNTS[key.N]))
        rendered = render_family_state(key.N, labels[family_id], seed=key.state_seed)
        return {"sample_id": key.sample_id, "N": key.N, "fine_family_id": family_id,
                "fine_family_label": rendered.family,
                "coarse_group": rendered.coarse_group,
                "fine_family_count": FINE_FAMILY_COUNTS[key.N], "rho_clean": rendered.rho,
                "taxonomy_semantics": "project_operational_not_exhaustive_SLOCC_LU"}
