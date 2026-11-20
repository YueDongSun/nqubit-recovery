"""Fresh population reads from the archived round-6 E5 fixed-cell model.

The generator retains the exposure assigned to the N population settings in a
4N-3 CHAIN/trine design. Static E5 has identical probabilities on every round,
so its pooled counts have a single multinomial law with k times the exposure.
The public calibration template is measured independently of the per-row
distortion strength. Only `population` is a supervision/scoring field.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np


SEED_NAMESPACE = 2026111805
FAMILY_NAMESPACE = {"e5": SEED_NAMESPACE, "structured": 2026111816}
FAMILY_SCHEMA = {
    "e5": "round6_e5_pooled_population_v1",
    "structured": "structured_amplitude_pooled_population_v1",
}
GRID_SIDE = 16
STRUCTURED_PRIOR = {
    "gradient_coefficient_range": [-0.65, 0.65],
    "patch_depth_range": [0.35, 1.75],
    "patch_width_cells_range": [1.0, 3.0],
    "patch_center": "uniform_active_cell",
    "coordinate_scale": "2*physical_grid_index/15-1",
    "residual": "independent log-amplitude slopes x,y minus a Gaussian patch",
}
ROLE_IDS = {
    "engineering_train": 1, "engineering_eval": 2, "qualification": 3,
    "train": 4, "dev": 5,
    "long_train": 21, "long_validation": 22,
    "long_qualification_train": 23, "long_qualification_validation": 24,
}


@lru_cache(maxsize=8)
def _design(nqrec_root: str, n: int):
    root = Path(nqrec_root).resolve(strict=True)
    src = root / "src"
    required_config = root / "configs" / "round6_device.json"
    if not required_config.is_file() or not (src / "nqrec/sim/round6.py").is_file():
        raise FileNotFoundError(f"round-6 E5 source/config missing in {root}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    sim = importlib.import_module("nqrec.sim.round6")
    for name, module in tuple(sys.modules.items()):
        if name.startswith("nqrec.") and getattr(module, "__file__", None):
            if not Path(module.__file__).resolve().is_relative_to(src):
                raise RuntimeError(f"{name} is already bound to another source root")
    device = sim.load_device(required_config)
    design = sim.make_design("fixed", n)
    if (design.n_settings != 4*n - 3 or
            not np.array_equal(design.support_modes[:n, 0], np.arange(n)) or
            not np.all(design.beta[:n, 0] == 1) or not np.all(design.beta[:n, 1] == 0)):
        raise ValueError("Expected ordered individual population reads followed by chain trines")
    return sim, device, design


@lru_cache(maxsize=16)
def _cell_polynomial(chi, eta_signal, background, background_idler, n_max):
    """P(cell|x) in powers of x=eta*p, for the calibrated single-cell read."""
    ratio = chi / (1 + chi)
    sector = np.arange(n_max + 1)
    weights = (1 - ratio) * ratio**sector
    weights[-1] += ratio**(n_max + 1)  # archived forward folds the upper tail
    no_signal = (1 - background) * (1 - eta_signal)**sector
    # q_s=1-(1-x)^s, except the same-mode double reads as 2x-2x^2.
    q = np.zeros((n_max + 1, n_max + 1))
    for s in range(1, n_max + 1):
        for power in range(1, s + 1):
            q[s, power] = (-1)**(power + 1) * math.comb(s, power)
    q[2, 2] = -2.0
    click = (1 - background_idler) * q
    click[:, 0] = background_idler
    no_click = -click.copy()
    no_click[:, 0] += 1
    w0, w1 = weights * no_signal, weights * (1 - no_signal)
    return np.stack([w0 @ no_click, w1 @ no_click, w0 @ click, w1 @ click])


def population_probabilities(population, efficiency, metadata):
    """Return [..., 00/10/01/11] probabilities for candidate per-mode populations."""
    p, eta = np.asarray(population, dtype=np.float64), np.asarray(efficiency, dtype=np.float64)
    if not (np.isfinite(p).all() and np.isfinite(eta).all() and
            np.all((p >= 0) & (p <= 1)) and np.all((eta > 0) & (eta <= 1))):
        raise ValueError("Population or calibrated efficiency outside its finite support")
    coefficients = _cell_polynomial(*(metadata[key] for key in
        ("chi", "eta_signal", "background", "background_idler", "n_max")))
    return np.moveaxis(np.polynomial.polynomial.polyval(eta*p, coefficients.T), 0, -1)


def generate(n: int, rows: int, role: str, *, nqrec_root: str | Path,
             start: int = 0, shots: int = 300, rounds: int = 16, family: str = "e5") -> dict:
    """Generate stable CPU arrays; training may move the observations to its own device.

    Row IDs are invariant to batch/worker order. Within one role and N, budget
    variants deliberately share latent rows and have distinct count streams.
    `metadata["trials"]` is pooled opportunities *per population setting*.
    """
    for name, value in (("n", n), ("rows", rows), ("start", start),
                        ("shots", shots), ("rounds", rounds)):
        if type(value) is not int or value < int(name != "start"):
            raise ValueError(f"Invalid {name}: {value!r}")
    if family not in FAMILY_NAMESPACE:
        raise ValueError(f"Unknown amplitude family {family!r}")
    namespace = FAMILY_NAMESPACE[family]
    if role not in ROLE_IDS or not 2 <= n <= 256:
        raise ValueError("Use a registered engineering/development role and N between 2 and 256")
    sim, device, design = _design(str(Path(nqrec_root).resolve()), n)
    per_round = sim.trials_per_setting(device, design, shots)
    trials = rounds * per_round
    meta = {
        "family": family, "seed_namespace": namespace, "role": role, "role_id": ROLE_IDS[role],
        "start": start, "rows": rows, "n_modes": n, "shots": shots, "rounds": rounds,
        "cell_order": ["00", "10", "01", "11"], "n_settings": design.n_settings,
        "trials_per_setting_per_round": per_round, "trials": trials,
        "population_read_opportunities_per_row": n * trials,
        "full_chain_read_opportunities_per_row": design.n_settings * trials,
        "chi": device.chi, "eta_signal": device.chain.eta_signal,
        "background": device.background, "background_idler": device.background_idler,
        "n_max": device.n_max, "e5_lambda": list(device.priors["e5_lambda"]),
        "generator": FAMILY_SCHEMA[family],
    }
    p = np.empty((rows, n), dtype=np.float64)
    templates, efficiency = np.empty_like(p), np.empty_like(p)
    extras = {}
    if family == "structured":
        meta["structured_prior"] = STRUCTURED_PRIOR.copy()
        extras["cell_index"] = np.empty((rows, n), dtype=np.int16)
    log_map = device.log_weight(design.system)
    eta_map = device.eta_map(design.system)
    label = np.array([sim.ALL_CLASSES.index("e5_amplitude")])
    for i in range(rows):
        key = [namespace, ROLE_IDS[role], n, start+i, 0]
        rng = np.random.default_rng(np.random.SeedSequence(key))
        cells = sim.fixed_cell_block(n, rng)
        mechanism = sim.draw_mechanisms(label, rounds, device.nominal(), device.priors, rng)
        templates[i], efficiency[i] = log_map[cells], eta_map[cells]
        log_population = mechanism["amplitude_lambda"][0] * templates[i]
        if family == "structured":
            extras["cell_index"][i] = cells
            y, x = np.divmod(cells, GRID_SIDE)
            center = rng.integers(n)
            slopes = rng.uniform(*STRUCTURED_PRIOR["gradient_coefficient_range"], size=2)
            depth = rng.uniform(*STRUCTURED_PRIOR["patch_depth_range"])
            width = rng.uniform(*STRUCTURED_PRIOR["patch_width_cells_range"])
            distance_squared = (x - x[center])**2 + (y - y[center])**2
            drift = slopes[0] * (2*x/(GRID_SIDE-1) - 1) + slopes[1] * (2*y/(GRID_SIDE-1) - 1)
            attenuation = depth * np.exp(-distance_squared/(2 * width**2))
            log_population += 2 * (drift - attenuation)
        weights = np.exp(log_population - log_population.max())
        p[i] = weights / weights.sum()
    pi = population_probabilities(p, efficiency, meta)
    if not np.isfinite(pi).all() or np.min(pi) < 0 or not np.allclose(
        pi.sum(axis=-1), 1, rtol=0, atol=2e-14
    ):
        raise ArithmeticError("Invalid probabilities in round-6 E5 marginal")
    counts = np.empty((rows, n, 4), dtype=np.int64)
    for i in range(rows):
        key = [namespace, ROLE_IDS[role], n, start+i, 1, shots, rounds]
        rng = np.random.default_rng(np.random.SeedSequence(key))
        counts[i] = rng.multinomial(trials, pi[i])
    if not np.all(counts.sum(-1) == trials):
        raise ArithmeticError("Population counts do not conserve READ opportunities")
    return {"counts": counts, "efficiency": efficiency, "log_weight": templates,
            "population": p, "metadata": meta, **extras}


def save_bank(path, batch):
    """Create a new portable NPZ; retain the separation of observations and labels."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        try:
            keys = ["counts", "efficiency", "log_weight", "population"]
            if batch["metadata"].get("family") == "structured":
                keys.append("cell_index")
            np.savez_compressed(handle, **{k: batch[k] for k in keys},
                metadata_json=json.dumps(batch["metadata"], sort_keys=True, allow_nan=False))
        except BaseException:
            handle.close()
            destination.unlink()
            raise


def load_bank(path):
    """Load a finished bank without nqrec, torch or pickle."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(data["metadata_json"].item())
        family = meta.setdefault("family", "e5")
        if (family not in FAMILY_SCHEMA or meta["generator"] != FAMILY_SCHEMA[family] or
                meta["seed_namespace"] != FAMILY_NAMESPACE[family] or
                meta["role_id"] != ROLE_IDS[meta["role"]] or
                meta["cell_order"] != ["00", "10", "01", "11"]):
            raise ValueError("Unexpected E5 bank schema or row role")
        shape = (meta["rows"], meta["n_modes"])
        arrays = {k: data[k] for k in ("counts", "efficiency", "log_weight", "population")}
        if family == "structured":
            cells = data["cell_index"]
            if (cells.shape != shape or not np.issubdtype(cells.dtype, np.integer)
                    or cells.min() < 0 or cells.max() >= GRID_SIDE**2
                    or meta.get("structured_prior") != STRUCTURED_PRIOR):
                raise ValueError("Structured calibration geometry or prior is inconsistent")
            arrays["cell_index"] = cells
    c = arrays["counts"]
    if (c.shape != (*shape, 4) or not np.issubdtype(c.dtype, np.integer)
            or (c < 0).any() or not np.all(c.sum(-1) == meta["trials"])):
        raise ValueError("Bank counts disagree with shape or READ exposure")
    for k in ("efficiency", "log_weight", "population"):
        if arrays[k].shape != shape or not np.isfinite(arrays[k]).all():
            raise ValueError(f"Invalid bank array {k!r}")
    if ((arrays["efficiency"] <= 0).any() or (arrays["efficiency"] > 1).any()
            or (arrays["population"] <= 0).any()
            or not np.allclose(arrays["population"].sum(-1), 1, rtol=0, atol=2e-14)):
        raise ValueError("Invalid E5 calibration or full-support label")
    return {**arrays, "metadata": meta}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nqrec-root", required=True, type=Path)
    parser.add_argument("--n", required=True, type=int)
    parser.add_argument("--rows", required=True, type=int)
    parser.add_argument("--role", required=True, choices=ROLE_IDS)
    parser.add_argument("--family", default="e5", choices=FAMILY_SCHEMA)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--shots", default=300, type=int)
    parser.add_argument("--rounds", default=16, type=int)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    result = generate(arguments.n, arguments.rows, arguments.role, start=arguments.start,
                      shots=arguments.shots, rounds=arguments.rounds, nqrec_root=arguments.nqrec_root,
                      family=arguments.family)
    save_bank(arguments.output, result)
    print(json.dumps(result["metadata"], sort_keys=True))
