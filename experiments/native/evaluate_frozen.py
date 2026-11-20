"""Evaluate a predeclared corrected core using disjoint native operation folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize_scalar
import torch

from .evaluate_static import DRAWS, SEED, score
from ml_recon.physics_hybrid_v10.classifier_n_full import V10ClassifierNFull
from ml_recon.physics_hybrid_v10.effective_projector_law import effective_probabilities
from .observations import (
    NOMINAL,
    native_public_observation,
    read_panel,
    site_schedule,
)
from ml_recon.physics_hybrid_v10.projector_core_export import (
    CLASSIFIER_SCHEMA,
    classifier_configuration,
)
from ml_recon.physics_hybrid_v10.training_ideas_objective import normalized_state


def transfer_probs(panel, schedule, q, net):
    ps = {
        k: np.empty(panel.counts.shape, dtype=np.float64) for k in ("core", "command")
    }
    n = schedule.n
    parity = torch.arange(len(schedule.labels)) % 2
    with torch.no_grad():
        for start in range(0, len(panel.counts), 8):
            ids = np.arange(start, min(start + 8, len(panel.counts)))
            for fold in (0, 1):
                observed = (parity != fold).expand(len(ids), -1)
                x = native_public_observation(
                    panel,
                    schedule,
                    ids,
                    signal_rate=q,
                    idler_gain=1.0,
                    idler_background=schedule.idler_background,
                    visible_mask=observed,
                )
                if x["counts"][~observed].any() or not x["counts"][observed].sum():
                    raise ValueError(
                        "Independent query counts were not removed from public features"
                    )
                output = net(x, N=n)
                rho = normalized_state(output)
                factor = torch.complex(
                    output["R1_factor_re"].double(), output["R1_factor_im"].double()
                )
                gram = factor @ factor.mH
                gram = (
                    gram / gram.diagonal(dim1=-2, dim2=-1).real.sum(-1)[:, None, None]
                )
                analyzer = torch.complex(x["analyzer_re"], x["analyzer_im"])
                cmd = torch.complex(x["prepared_re"], x["prepared_im"])
                pure = cmd.unsqueeze(-1) * cmd.conj().unsqueeze(-2)
                query = parity == fold
                cmd64 = cmd.to(torch.complex128)
                pure64 = cmd64.unsqueeze(-1) * cmd64.conj().unsqueeze(-2)
                for key, state, stable in (
                    ("core", rho, gram),
                    ("command", pure, pure64),
                ):
                    p = (
                        effective_probabilities(
                            state, analyzer, x["effective_calibration"]
                        )
                        .double()
                        .numpy()
                    )
                    # The float32 model produces a normalized law to float32 precision.
                    # Re-evaluate the density and transfer in double for the exposure-weighted score.
                    stable = (
                        stable
                        / stable.diagonal(dim1=-2, dim2=-1).sum(-1).real[:, None, None]
                    )
                    if (state.to(torch.complex128) - stable).abs().max() > 2e-6:
                        raise ValueError(
                            "Exported factors do not reconstruct the public density"
                        )
                    a = analyzer.to(torch.complex128)
                    a = a / torch.linalg.vector_norm(a, dim=-1, keepdim=True)
                    checked = effective_probabilities(
                        stable, a, x["effective_calibration"].double()
                    ).numpy()
                    if np.max(np.abs(checked - p)) > 2e-6:
                        raise ValueError(
                            "Precision changed native predicted probabilities"
                        )
                    ps[key][ids[:, None], np.flatnonzero(query)[None, :]] = checked[
                        :, query
                    ]
    return ps


def run(data_root, path, n, calibration_dir, histogram):
    data = torch.load(path, map_location="cpu", weights_only=True)
    ident = data["identity"]
    if (
        data["schema"] != CLASSIFIER_SCHEMA
        or data["law"] != "effective_single_event_v1"
        or data["sector_family"] != "K0_K1_reference"
        or data["updates"] != ident["steps"]
        or [ident[k] for k in ("n", "seed", "arm", "rank")] != [n, 61, "baseline", 4]
    ):
        raise ValueError("Wrong identity for the predeclared frozen transfer")
    return evaluate_checkpoint(data_root, path, n, data, calibration_dir, histogram)


def evaluate_checkpoint(data_root, path, n, data, calibration_dir, histogram):
    """Score an already identified, frozen constructor and complete state on native folds."""
    ident = data["identity"]
    cfg, state = classifier_configuration(data)
    net = V10ClassifierNFull(**cfg).eval()
    net.load_state_dict(state, strict=True)
    schedule = site_schedule(n, calibration_dir, histogram)
    tr = read_panel(data_root, schedule, NOMINAL[n][0])
    op_counts = tr.counts.sum(0).astype(float)
    q = (op_counts[:, 1] + op_counts[:, 3]).sum() / op_counts.sum()
    address = op_counts + 0.5
    address /= address.sum(-1, keepdims=True)
    train = transfer_probs(tr, schedule, q, net)

    def obj(a):
        p = a * train["core"] + (1 - a) * train["command"]
        return -(tr.counts * np.log(p)).sum() / tr.counts.sum()

    optimum = minimize_scalar(obj, bounds=(0, 1), method="bounded")
    alpha = min((0.0, 1.0, float(optimum.x)), key=obj)
    val = read_panel(data_root, schedule, NOMINAL[n][1])
    if set(val.trajectory) & set(tr.trajectory):
        raise ValueError("TRAIN/validation trajectory overlap")
    pred = transfer_probs(val, schedule, q, net)
    pred["mixture"] = alpha * pred["core"] + (1 - alpha) * pred["command"]
    pred["address"] = address
    groups = np.unique(val.trajectory)
    draws = np.random.default_rng(SEED + n).integers(
        len(groups), size=(DRAWS, len(groups))
    )
    out = score(
        val,
        pred,
        groups,
        draws,
        contrasts=(
            ("core", "command"),
            ("mixture", "command"),
            ("core", "address"),
            ("mixture", "address"),
        ),
    )
    return dict(
        n=n,
        core_file=path.name,
        training_identity=ident,
        updates=data["updates"],
        native_train_rows=len(tr.counts),
        core_weight=alpha,
        hardware_read_efficiency_scale=1.0,
        total_signal_rate=float(q),
        validation=out,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--core-3", type=Path, required=True)
    parser.add_argument("--core-4", type=Path, required=True)
    parser.add_argument(
        "--calibration-dir",
        required=True,
        type=Path,
        help="directory of round6_device_v24_{3,4}q.yaml",
    )
    parser.add_argument(
        "--histogram",
        required=True,
        type=Path,
        help="external calibration CSV containing Idler Absolute",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(1)
    t = time.perf_counter()
    report = {
        "schema": "native_frozen_core_transfer_v1",
        "law": "effective_single_event_v1",
        "folds": "query by native operation index parity; complementary counts alone visible",
        "systems": {
            str(n): run(args.data_root, c, n, args.calibration_dir, args.histogram)
            for n, c in ((3, args.core_3), (4, args.core_4))
        },
    }
    report["elapsed_seconds"] = time.perf_counter() - t
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )
    print(
        f"Frozen native transfer: both systems scored in {report['elapsed_seconds']:.2f} s."
    )


if __name__ == "__main__":
    main()
