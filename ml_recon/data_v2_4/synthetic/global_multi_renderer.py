"""Integer-valued global multi-click synthetic renderer."""

from __future__ import annotations

import numpy as np


def render_global_multi(rng: np.random.Generator, mean_multi2: float = 4.0) -> np.ndarray:
    multi2 = int(rng.poisson(max(mean_multi2, 0.0)))
    pair_probabilities = rng.dirichlet(np.ones(6))
    pairs = rng.multinomial(multi2, pair_probabilities).astype(np.uint32)
    return np.asarray((multi2, *pairs, int(rng.poisson(0.5)), int(rng.poisson(0.1))),
                      dtype=np.uint32)
