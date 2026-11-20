"""Fit native static predictors on TRAIN; read and score the named validation roles once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit, softmax
from scipy.stats import binom
import torch

from .observations import (
    NOMINAL,
    SCHEMA,
    native_public_observation,
    read_panel,
    site_schedule,
    verify_packed,
)
from ml_recon.physics_hybrid_v10.effective_projector_law import effective_probabilities
from nqrec.physics.single_event_mc import calibrate_source_weights

DRAWS = 2000
SEED = 2026112027


def optical_terms(command, schedule):
    transfer = schedule.transfer.numpy()
    coherent = np.abs(command @ transfer.conj().T) ** 2
    diagonal = np.abs(command) ** 2 @ (np.abs(transfer) ** 2).T
    return coherent, diagonal


def physical(terms, joint, params):
    coherent, diagonal = terms
    gain, background, vis = np.exp(params[0]), expit(params[1]), params[2]
    retrieval = gain * (vis * coherent + (1 - vis) * diagonal)
    marginal = joint.sum(0)
    idler = marginal * background + (1 - background) * retrieval[..., None] * joint[1]
    return np.concatenate((marginal - idler, idler), axis=-1)


def fit(panel, schedule):
    counts = panel.counts
    op_counts = counts.sum(0).astype(float)
    pooled = op_counts.sum(0) + 0.5
    pooled /= pooled.sum()
    op_freq = op_counts + 0.5
    op_freq /= op_freq.sum(-1, keepdims=True)
    signal_rate = (op_counts[:, 1] + op_counts[:, 3]).sum() / op_counts.sum()
    as_d = lambda x: torch.tensor(x, dtype=torch.float64)
    joint = calibrate_source_weights(
        as_d(signal_rate),
        as_d(schedule.eta_signal),
        as_d(schedule.signal_background),
        double_probability=as_d(0.0),
    )["joint"].numpy()
    terms = optical_terms(panel.command, schedule)
    marg = joint.sum(0)

    def objective(theta):
        gain, bg, v = np.exp(theta[0]), expit(theta[1]), theta[2]
        ret = gain * (v * terms[0] + (1 - v) * terms[1])
        probs = physical(terms, joint, theta)
        contrast = counts[..., 2:] / probs[..., 2:] - counts[..., :2] / probs[..., :2]
        score_1 = (contrast * joint[1]).sum(-1)
        scale = 1e4 / counts.sum()
        grad = [
            ((1 - bg) * ret * score_1).sum(),
            bg * (1 - bg) * ((contrast * marg).sum() - (ret * score_1).sum()),
            ((1 - bg) * gain * (terms[0] - terms[1]) * score_1).sum(),
        ]
        return -scale * (counts * np.log(probs)).sum(), -scale * np.array(grad)

    maximum_gain = 0.999 / schedule.transfer.abs().square().sum(-1).max().item()
    optimization = minimize(
        objective,
        [0.0, logit(schedule.idler_background), 0.7],
        jac=True,
        method="L-BFGS-B",
        bounds=[(-15.0, np.log(maximum_gain)), (-16.0, 0.0), (0.0, 1.0)],
        options={"maxiter": 80, "ftol": 1e-13, "gtol": 1e-8},
    )
    if not optimization.success:
        raise RuntimeError(optimization.message)
    predictions = physical(terms, joint, optimization.x)
    mean_pred = np.average(predictions, axis=0, weights=panel.reads)

    def tilt_loss(weight):
        pred = softmax(
            np.log(op_freq)[None] + weight * (np.log(predictions) - np.log(mean_pred)),
            axis=-1,
        )
        return -(counts * np.log(pred)).sum() / counts.sum()

    opt = minimize_scalar(
        tilt_loss, bounds=(0, 1), method="bounded", options={"xatol": 1e-7}
    )
    choices = (0.0, 1.0, float(opt.x))
    weight = min(choices, key=tilt_loss)
    # At zero optical gain the TRAIN MLE of idler background is its marginal.
    # Cell probabilities are affine in background and the two nonnegative
    # optical coefficients (coherent and diagonal). Positive inward derivatives
    # certify the convex optimum at zero on the closed physical parameter set.
    b0 = op_counts[:, 2:].sum() / op_counts.sum()
    contrast0 = counts[..., 2:] / (b0 * marg) - counts[..., :2] / ((1 - b0) * marg)
    score0 = (contrast0 * joint[1]).sum(-1)
    slopes = [float(-((1 - b0) * term * score0).sum() / counts.sum()) for term in terms]
    return (
        pooled,
        op_freq,
        joint,
        optimization.x,
        mean_pred,
        weight,
        dict(
            signal_rate=signal_rate,
            signal_efficiency=schedule.eta_signal,
            signal_background=schedule.signal_background,
            idler_site_gain=np.exp(optimization.x[0]),
            idler_background=expit(optimization.x[1]),
            command_visibility=optimization.x[2],
            tilt_exponent=weight,
            optimizer_iterations=int(optimization.nit),
            train_boundary_diagnostic=dict(
                zero_gain_background=b0,
                coherent_gain_derivative=slopes[0],
                dephased_gain_derivative=slopes[1],
                closed_domain_zero_gain_optimal=bool(min(slopes) >= 0.0),
            ),
            optimizer_status=str(optimization.message),
            training_rows=len(counts),
            training_groups=len(set(panel.trajectory)),
            known_exposure_seconds_rows=int(panel.known_seconds.sum()),
        ),
    )


def validate_abi(panel, schedule, theta, signal_rate, expected):
    """Round-trip the full calibrated native setting sequence through the shared #44 interface."""
    checks = 0.0
    delta = 0.0
    for start in range(0, len(panel.counts), 12):
        idx = np.arange(start, min(start + 12, len(panel.counts)))
        vismask = torch.ones((len(idx), len(schedule.labels)), dtype=torch.bool)
        gain, bg, v = np.exp(theta[0]), expit(theta[1]), theta[2]
        public = native_public_observation(
            panel,
            schedule,
            idx,
            signal_rate=signal_rate,
            idler_gain=gain,
            idler_background=bg,
            visible_mask=vismask,
        )
        assert torch.equal(public["counts"], torch.from_numpy(panel.counts[idx]))
        cal = public["effective_calibration"].double()
        analyzer = torch.complex(
            public["analyzer_re"].double(), public["analyzer_im"].double()
        )
        command = torch.complex(
            public["prepared_re"].double(), public["prepared_im"].double()
        )
        # Public tokens serialize kets in float32. Restore unit trace in the same
        # numerical dtype used for physical likelihood evaluation.
        command = command / torch.linalg.vector_norm(command, dim=-1, keepdim=True)
        analyzer = analyzer / torch.linalg.vector_norm(analyzer, dim=-1, keepdim=True)
        pure = command[:, :, None] * command.conj()[:, None, :]
        rho = v * pure + (1 - v) * torch.diag_embed(command.abs().square()).to(
            command.dtype
        )
        p = effective_probabilities(rho, analyzer, cal).numpy()
        checks = max(checks, float(np.max(np.abs(p - expected[idx]))))
        delta = max(
            delta,
            float(
                (
                    cal[:, :, 3].sqrt()[:, :, None] * analyzer
                    - schedule.transfer[None] * np.sqrt(gain)
                )
                .abs()
                .max()
            ),
        )
    if checks > 2e-7 or delta > 1e-6:
        raise ValueError("Per-operation projector changed through the public ABI")
    return dict(
        max_probability_difference=checks, max_complex_transfer_difference=delta
    )


def score(
    panel,
    preds,
    groups,
    draw,
    contrasts=(
        ("address", "pooled"),
        ("command", "pooled"),
        ("address_command_tilt", "address"),
        ("address", "command"),
    ),
):
    _, inv = np.unique(panel.trajectory, return_inverse=True)
    counts = panel.counts
    reads = np.broadcast_to(panel.reads[:, None], counts.shape[:2])
    denom = np.bincount(inv, weights=reads.sum(-1))
    size = np.bincount(inv, weights=np.full(len(inv), counts.shape[1]))

    def bounds(x, den):
        boot = x[draw].sum(1) / den[draw].sum(1)
        return np.quantile(boot, [0.025, 0.975], axis=0).T.tolist()

    values = {}
    losses = {}
    for key, p in preds.items():
        p = np.broadcast_to(p, counts.shape)
        if np.any(p <= 0) or not np.allclose(p.sum(-1), 1.0, atol=1e-13, rtol=0):
            raise ValueError("Nonphysical native predictions")
        cost = np.bincount(inv, weights=-(counts * np.log(p)).sum((1, 2)))
        lo, hi = binom.interval(0.95, reads[:, :, None], p)
        passed = ((lo <= counts) & (counts <= hi)).sum(1)
        covered = np.stack(
            [np.bincount(inv, weights=passed[:, i]) for i in range(4)], axis=1
        )
        losses[key] = cost
        vals = covered.sum(0) / size.sum()
        values[key] = dict(
            nll_per_read=cost.sum() / denom.sum(),
            nll_per_read_ci95=bounds(cost, denom),
            marginal_95_coverage=vals.tolist(),
            coverage_ci95=bounds(covered, size[:, None]),
            expected_cell_rate=(reads[:, :, None] * p)
            .sum((0, 1))
            .__truediv__(reads.sum())
            .tolist(),
        )
    paired = {}
    for arm, ref in contrasts:
        delta = losses[ref] - losses[arm]
        paired[f"{arm}_improvement_over_{ref}"] = dict(
            mean=delta.sum() / denom.sum(), ci95=bounds(delta, denom)
        )
    return dict(
        validation_rows=len(inv),
        validation_groups=len(groups),
        paid_reads=int(reads.sum()),
        known_exposure_seconds_rows=int(panel.known_seconds.sum()),
        observed_cell_rate=(counts.sum((0, 1)) / reads.sum()).tolist(),
        models=values,
        paired=paired,
    )


def run(data_root, packed_root, n, calibration_dir, histogram):
    schedule = site_schedule(n, calibration_dir, histogram)
    train = read_panel(data_root, schedule, NOMINAL[n][0])
    pool, op, joint, theta, mean_pred, weight, model = fit(train, schedule)
    val = read_panel(data_root, schedule, NOMINAL[n][1])
    groups = np.unique(val.trajectory)
    if set(groups) & set(train.trajectory):
        raise ValueError("Trajectory overlap across admitted roles")
    with np.load(packed_root / f"packed_n{n}/panel.npz", allow_pickle=False) as bank:
        verify_packed(schedule, val, bank)
        # Both mandatory counterexamples must fail in the same converter.
        packed = {
            key: bank[key]
            for key in (
                "cells",
                "beta",
                "theta_read",
                "eta_cells",
                "counts",
                "read_trials",
                "target",
                "trajectory_group",
            )
        }
        rejected = []
        for key, change in (
            ("theta_read", packed["theta_read"] + 0.25),
            ("counts", packed["counts"][:, :, [0, 2, 1, 3]]),
        ):
            try:
                verify_packed(schedule, val, {**packed, key: change})
            except ValueError as exc:
                if key not in str(exc):
                    raise
                rejected.append(key)
            else:
                raise AssertionError(f"Failed to reject changed {key}")
    pc = physical(optical_terms(val.command, schedule), joint, theta)
    abi = validate_abi(val, schedule, theta, model["signal_rate"], pc)
    nominal_theta = np.array([0.0, logit(schedule.idler_background), 1.0])
    nominal = physical(optical_terms(val.command, schedule), joint, nominal_theta)
    nominal_abi = validate_abi(
        val, schedule, nominal_theta, model["signal_rate"], nominal
    )
    joined = softmax(
        np.log(op)[None] + weight * (np.log(pc) - np.log(mean_pred)), axis=-1
    )
    draws = np.random.default_rng(SEED + n).integers(
        len(groups), size=(DRAWS, len(groups))
    )
    scores = score(
        val,
        dict(pooled=pool, address=op, command=pc, address_command_tilt=joined),
        groups,
        draws,
    )
    eta = (schedule.transfer.abs().square().sum(-1) * model["idler_site_gain"]).numpy()
    return dict(
        n=n,
        train_split=train.split,
        validation_split=val.split,
        operations=len(schedule.labels),
        registry_source="V2.4 flat order, J18 descriptor, normalized repository sequence basis",
        phase_sign=-1,
        trajectory_overlap=0,
        model=model,
        validation=scores,
        packed_mismatch_rejections=rejected,
        validation_abi=abi,
        nominal_calibration_abi=nominal_abi,
        per_operation_effective_efficiency_range=[float(eta.min()), float(eta.max())],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--packed-root", required=True, type=Path)
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
    start = time.perf_counter()
    report = dict(
        schema=SCHEMA,
        model_law="effective_single_event_v1",
        composite_law="native_empirical_density_ratio_v1",
        statistical_unit="trajectory; retain row multiplicity and READ denominators",
        bootstrap_draws=DRAWS,
        cells=["00", "10", "01", "11"],
        validation_role="historical development, no fresh confirmation",
        systems={
            str(n): run(
                args.data_root,
                args.packed_root,
                n,
                args.calibration_dir,
                args.histogram,
            )
            for n in (3, 4)
        },
    )
    report["elapsed_seconds"] = time.perf_counter() - start
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Both nominal validation panels scored in {report['elapsed_seconds']:.2f} s")


if __name__ == "__main__":
    main()
