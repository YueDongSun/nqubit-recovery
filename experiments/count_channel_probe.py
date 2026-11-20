"""Frozen, exploratory decomposition of count-source information and shot-noise variance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ml_recon.physics_hybrid_v10.classifier_n_full import V10ClassifierNFull
from ml_recon.physics_hybrid_v10.effective_inference_metrics import command_and_truth
from ml_recon.physics_hybrid_v10.effective_projector_law import effective_probabilities
from ml_recon.physics_hybrid_v10.effective_projector_observation import (
    EFFECTIVE_LAW,
    make_public_observation,
)
from experiments.protocol import file_digest, register as register_once
from ml_recon.physics_hybrid_v10.training_ideas_data import (
    allocate_integer_budget,
    make_projector_training_batch,
)
from ml_recon.physics_hybrid_v10.training_ideas_objective import normalized_state


NAMESPACE = 202611201823091
BUDGETS = (1024, 4096, 16384)
ARMS = ("command", "population", "normalized_coherence", "true")
CAPACITIES = ("cs", "cm", "cl")
ROWS, REPEATS, STEP = 384, 8, 32768


def sources(command, truth):
    """Combine either population vector with either unit-diagonal coherence matrix."""
    q = command.diagonal(dim1=-2, dim2=-1).real
    p = truth.diagonal(dim1=-2, dim2=-1).real
    if (q <= 0).any() or (p <= 0).any():
        raise ValueError(
            "The registered positive-population generator produced an unsupported row"
        )

    def replace(rho, scale):
        return scale[:, :, None] * rho * scale[:, None, :]

    return (
        command,
        replace(command, (p / q).sqrt()),
        replace(truth, (q / p).sqrt()),
        truth,
    )


def prepare(folder: Path, n: int, rows=ROWS, repeats=REPEATS):
    # The fresh engineering-independent namespace is never passed as a CONFIRMATION split.
    base = make_projector_training_batch(
        n,
        rows,
        BUDGETS[0],
        NAMESPACE + n,
        observation_law=EFFECTIVE_LAW,
        include_teachers=False,
    )
    command, truth = command_and_truth(base)
    analyzer = torch.complex(base["analyzer_re"], base["analyzer_im"])
    prepared = torch.complex(base["prepared_re"], base["prepared_im"])
    calibration = base["effective_calibration"]
    family = torch.where(
        base["target_is_nominal"], -1, base["target_operational_noise"]
    )
    visible = base["setting_mask"]
    probabilities = []
    for state in sources(command, truth):
        probs = effective_probabilities(state, analyzer, calibration).double().numpy()
        # Source validation is at float32 precision; use an exact numerical unit sum for sampling.
        probabilities.append(probs / probs.sum(axis=-1, keepdims=True))
    shots = np.stack(
        [allocate_integer_budget(budget, analyzer.shape[1]) for budget in BUDGETS]
    )
    increments = np.diff(shots, axis=0, prepend=np.zeros_like(shots[:1]))
    counts = np.empty(
        (repeats, len(ARMS), len(BUDGETS), rows, analyzer.shape[1], 4), dtype=np.int32
    )
    for rep in range(repeats):
        for arm, probs in enumerate(probabilities):
            cumulative = np.zeros(probs.shape, dtype=np.int32)
            for level, added in enumerate(increments):
                # Independent between repeat draws, paired by source arm and shared between models.
                rng = np.random.default_rng(
                    np.random.SeedSequence([NAMESPACE, n, rep, level, 223])
                )
                for setting, paid in enumerate(added):
                    cumulative[:, setting] += rng.multinomial(
                        int(paid), probs[:, setting]
                    ).astype(np.int32)
                counts[rep, arm, level] = cumulative
    if not np.array_equal(
        counts.sum(-1), np.broadcast_to(shots[None, None, :, None], counts.shape[:-1])
    ):
        raise ValueError("count draws disagree with integer exposure")
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f"bank_n{n}.partial.npz"
    np.savez_compressed(
        tmp,
        counts=counts,
        shots=shots,
        truth=truth.numpy(),
        command=prepared.numpy(),
        analyzer=analyzer.numpy(),
        calibration=calibration.numpy(),
        visible=visible.numpy(),
        families=family.numpy(),
    )
    tmp.replace(folder / f"bank_n{n}.npz")


@torch.inference_mode()
def score(folder: Path, n: int, cap: str, core: Path, expected_hash: str, device: str):
    with core.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != expected_hash:
            raise ValueError("core and exploratory registration differ")
    run = f"joint_{cap}_32768_s7301"
    payload = torch.load(core, map_location="cpu", weights_only=True)
    if (
        payload["law"] != EFFECTIVE_LAW
        or payload["identity"]["seed"] != 7301
        or payload["identity"]["train_states_per_n"] != 32768
        or payload["identity"]["sizes"] != [3, 4, 8]
        or payload["identity"]["lane"] != run
        or payload["updates"] != STEP
    ):
        raise ValueError("unexpected frozen checkpoint")
    model = V10ClassifierNFull(**payload["classifier_config"]).to(device).eval()
    model.load_state_dict(payload["classifier_state_dict"], strict=True)
    with np.load(folder / f"bank_n{n}.npz") as bank:
        arrays = {k: torch.from_numpy(bank[k]) for k in bank.files}
    counts, truth = arrays["counts"], arrays["truth"]
    repeats, _, _, rows = counts.shape[:4]
    risk = np.empty((repeats, len(ARMS), len(BUDGETS), rows, 2), dtype=np.float64)
    means = torch.zeros((len(BUDGETS), rows, n, n), dtype=torch.complex128)
    for rep in range(repeats):
        for a in range(len(ARMS)):
            for b in range(len(BUDGETS)):
                obs = make_public_observation(
                    counts=counts[rep, a, b].long(),
                    shots=arrays["shots"][b].expand(rows, -1),
                    analyzers=arrays["analyzer"],
                    prepared=arrays["command"],
                    calibration=arrays["calibration"],
                    visible_mask=arrays["visible"],
                )
                if obs["counts"][~arrays["visible"]].any():
                    raise ValueError("hidden counts entered the network")
                pred = []
                for start in range(0, rows, 32):
                    segment = {
                        k: v[start : start + 32].to(device) for k, v in obs.items()
                    }
                    pred.append(
                        normalized_state(model(segment, N=n)).cpu().to(torch.complex128)
                    )
                estimate = torch.cat(pred)
                dif = (estimate - truth).abs().square()
                diag = dif.diagonal(dim1=-2, dim2=-1).sum(-1)
                risk[rep, a, b, :, 0] = diag.numpy()
                risk[rep, a, b, :, 1] = (dif.sum((-1, -2)) - diag).numpy()
                if a == 3:
                    means[b] += estimate / repeats
    native = risk[:, 3].sum(-1).mean(0)
    centre = (means - truth).abs().square().sum((-2, -1)).numpy()
    variance = (native - centre) * repeats / (repeats - 1)
    bias2 = centre - variance / repeats
    if variance.min() < -1e-10 or not np.allclose(
        native, bias2 + variance, rtol=1e-10, atol=1e-12
    ):
        raise ArithmeticError("finite-repeat bias and variance accounting disagrees")
    tmp = folder / f"n{n}_{cap}.partial.npz"
    np.savez_compressed(
        tmp,
        risk=risk,
        bias_squared_unbiased=bias2,
        variance=variance,
        family=arrays["families"].numpy(),
        state_ids=np.arange(rows),
        weight=expected_hash,
        run=run,
        step=STEP,
        budget=BUDGETS,
        arms=ARMS,
    )
    tmp.replace(folder / f"n{n}_{cap}.npz")


def summarize(folder: Path, registration: dict):
    table = []
    for n in (3, 4, 8):
        for cap in CAPACITIES:
            with np.load(folder / f"n{n}_{cap}.npz") as arr:
                if (
                    str(arr["weight"]) != registration["weights"][cap]
                    or len(arr["state_ids"]) != ROWS
                ):
                    raise ValueError("cannot pool unmatched exploratory cases")
                risk = arr["risk"].mean(0)
                for b, budget in enumerate(BUDGETS):
                    per_source = risk[:, b].sum(-1)
                    pop = (
                        (per_source[1] - per_source[0])
                        + (per_source[3] - per_source[2])
                    ) / 2
                    coh = (
                        (per_source[2] - per_source[0])
                        + (per_source[3] - per_source[1])
                    ) / 2
                    interaction = (
                        per_source[3] - per_source[2] - per_source[1] + per_source[0]
                    )
                    endpoints = np.stack(
                        [
                            pop,
                            coh,
                            interaction,
                            per_source[3] - per_source[0],
                            arr["bias_squared_unbiased"][b],
                            arr["variance"][b],
                        ],
                        axis=1,
                    )
                    for family in (None, *np.unique(arr["family"])):
                        ids = (
                            np.arange(ROWS)
                            if family is None
                            else np.flatnonzero(arr["family"] == family)
                        )
                        rng = np.random.default_rng(
                            [
                                NAMESPACE,
                                n,
                                b,
                                323,
                                20 if family is None else int(family) + 1,
                            ]
                        )
                        groups = [
                            ids[arr["family"][ids] == f]
                            for f in np.unique(arr["family"][ids])
                        ]
                        draws = np.concatenate(
                            [rng.choice(g, size=(4000, len(g))) for g in groups], axis=1
                        )
                        ci = np.quantile(
                            endpoints[draws].mean(1), [0.025, 0.975], axis=0
                        )
                        names = [
                            "population_effect",
                            "coherence_effect",
                            "interaction",
                            "native_minus_command",
                            "native_bias_squared",
                            "native_count_variance",
                        ]
                        row = dict(
                            n=n,
                            size=cap,
                            run_id=str(arr["run"]),
                            step=STEP,
                            read_budget=budget,
                            family="all" if family is None else int(family),
                            states=len(ids),
                            repeats=REPEATS,
                        )
                        row["mean_risk_parts_diagonal_offdiagonal"] = (
                            risk[:, b, ids].mean(1).tolist()
                        )
                        row["endpoints"] = {
                            k: [float(endpoints[ids, j].mean()), *ci[:, j].tolist()]
                            for j, k in enumerate(names)
                        }
                        table.append(row)
    register_once(folder / "result.json", dict(registration=registration, rows=table))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("register", "prepare", "score", "summarize"))
    cli.add_argument("--folder", type=Path, required=True)
    cli.add_argument("--n", type=int, choices=(3, 4, 8))
    cli.add_argument("--cap", choices=CAPACITIES)
    cli.add_argument("--core", type=Path)
    cli.add_argument("--device", default="cuda")
    cli.add_argument(
        "--cores",
        nargs=3,
        type=Path,
        metavar=("CS", "CM", "CL"),
        help="matched 32768-step exports for registration",
    )
    args = cli.parse_args()
    if args.phase == "register":
        if not args.cores:
            cli.error("register requires --cores CS CM CL")
        weights = {cap: file_digest(path) for cap, path in zip(CAPACITIES, args.cores)}
        register_once(
            args.folder / "registration.json",
            dict(
                schema="frozen_count_channels_exploratory_v1",
                namespace=NAMESPACE,
                rows_per_n=ROWS,
                repeats=REPEATS,
                weights=weights,
                n=[3, 4, 8],
                budgets=BUDGETS,
                seed=7301,
                step=STEP,
                training_states_per_n=32768,
            ),
        )
        raise SystemExit(0)
    if args.phase in ("prepare", "score") and args.n is None:
        cli.error("select --n")
    if args.phase == "score" and (args.cap is None or args.core is None):
        cli.error("scoring requires --cap and --core")
    torch.set_num_threads(1)
    reg = json.loads((args.folder / "registration.json").read_text())
    if (
        reg["namespace"] != NAMESPACE
        or reg["rows_per_n"] != ROWS
        or reg["repeats"] != REPEATS
    ):
        raise ValueError("unexpected exploratory registration")
    if args.phase == "prepare":
        prepare(args.folder, args.n)
    elif args.phase == "score":
        score(
            args.folder,
            args.n,
            args.cap,
            args.core,
            reg["weights"][args.cap],
            args.device,
        )
    else:
        summarize(args.folder, reg)
