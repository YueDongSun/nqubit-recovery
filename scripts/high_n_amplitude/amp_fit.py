"""Fit shared population estimators on the aligned E5 population-setting experiment."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch import nn

if __package__:
    from .amp_control import centered_action, conditional_fidelity
    from .amp_data import generate, population_probabilities
else:
    from amp_control import centered_action, conditional_fidelity
    from amp_data import generate, population_probabilities


@dataclass(frozen=True)
class Profile:
    sizes: tuple[int, ...]
    train_rows: int
    evaluation_rows: int
    updates: int
    batch_size: int
    train_role: str
    evaluation_role: str
    seconds: int
    budgets: tuple[tuple[int, int], ...] = ((300, 16),)
    row_start: int = 0


PROFILES = {
    "smoke": Profile((4, 32, 256), 4, 4, 6, 4, "engineering_train", "engineering_eval", 300),
    "dev": Profile((32, 64, 128, 256), 128, 64, 512, 16, "train", "dev", 1800),
    "expanded_dev": Profile((32, 64, 128, 256), 4096, 512, 12000, 32, "train", "dev", 14400,
                            ((30, 1), (300, 16)), 50000),
}
HIDDEN = 32
LEARNING_RATE = 0.003
ACTION_BOUND = 0.25
GRID_POINTS = 241
LOCAL_LOG_POPULATION_SD = 1.5
LOCAL_GRID_POINTS = 193


class PopulationModel(nn.Module):
    """Estimate one row-wide distortion strength with optional shared local residual."""

    def __init__(self, arm, family="e5"):
        super().__init__()
        self.arm = arm
        feature_count = 10 if family == "structured" else 8
        self.encoder = nn.Sequential(nn.Linear(feature_count, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN), nn.Tanh())
        self.strength = nn.Linear(HIDDEN, 1)
        self.residual = nn.Linear(HIDDEN, 1) if arm == "flexible" else None
        if self.residual is not None:
            nn.init.zeros_(self.residual.weight)
            nn.init.zeros_(self.residual.bias)

    def forward(self, features, log_weight, lambda_range):
        latent = self.encoder(features)
        lower, upper = lambda_range
        coefficient = lower + (upper - lower) * self.strength(latent.mean(1)).sigmoid()
        logits = coefficient * log_weight
        if self.residual is not None:
            logits = logits + self.residual(latent).squeeze(-1)
        return logits.log_softmax(-1)


def encode_observations(data):
    """Use public design and residual counts relative to an undistorted W source."""
    counts = data["counts"]
    rows, n_modes, _ = counts.shape
    efficiency, log_weight, metadata = data["efficiency"], data["log_weight"], data["metadata"]
    uniform = np.full((rows, n_modes), 1.0 / n_modes)
    nominal = population_probabilities(uniform, efficiency, metadata)
    expected = float(metadata["trials"]) * nominal
    residual = (counts - expected) / np.sqrt(np.maximum(expected * (1.0 - nominal), 1.0))
    features = [
        np.arcsinh(residual),
        log_weight[..., None],
        np.log(efficiency)[..., None],
        np.full_like(uniform[..., None], math.log(n_modes)),
        np.log1p(expected[..., 3:4]),
    ]
    if metadata.get("family", "e5") == "structured":
        cells = data["cell_index"]
        features.extend([((cells // 16) / 7.5 - 1.)[..., None], ((cells % 16) / 7.5 - 1.)[..., None]])
    return np.concatenate(features, axis=-1)


def local_likelihood_population(counts, efficiency, prior_population, metadata):
    """Independent log-population shrinkage fits, normalized over the full row.

    Each coordinate has a truncated Gaussian working prior in log(N*p_i),
    centered at the no-count E5 population prediction with fixed width 1.5.
    This estimates heterogeneous per-mode residuals with the same four-cell
    likelihood. The post-fit normalization imposes the simplex; this finite
    grid calculation is neither a joint MLE nor a joint simplex posterior.
    """
    n_modes = counts.shape[1]
    log_relative = np.linspace(-12.0, math.log(n_modes), LOCAL_GRID_POINTS)
    candidate = np.exp(log_relative) / n_modes
    candidate[-1] = 1.0
    outputs = []
    for first in range(0, len(counts), 4):
        section = slice(first, first + 4)
        probabilities = population_probabilities(candidate[None, :, None], efficiency[section, None], metadata)
        if np.any(probabilities <= 0.) or not np.isfinite(probabilities).all():
            raise ValueError("Unexpected finite-grid four-cell probability")
        log_likelihood = (counts[section, None] * np.log(probabilities)).sum(-1)
        offset = log_relative[None, :, None] - np.log(n_modes * prior_population[section, None])
        log_weights = log_likelihood - 0.5 * (offset / LOCAL_LOG_POPULATION_SD) ** 2
        weights = np.exp(log_weights - log_weights.max(1, keepdims=True))
        weights[:, [0, -1], :] *= 0.5
        local = (weights * candidate[None, :, None]).sum(1) / weights.sum(1)
        outputs.append(local / local.sum(-1, keepdims=True))
    return np.concatenate(outputs)


def likelihood_estimates(data):
    """Fit the native four-cell likelihood on the public E5 template grid."""
    counts, weight, efficiency = data["counts"], data["log_weight"], data["efficiency"]
    grid = np.linspace(*data["metadata"]["e5_lambda"], GRID_POINTS)
    mle, posterior, prior = [], [], []
    for first in range(0, len(counts), 4):
        section = slice(first, first + 4)
        logits = grid[None, :, None] * weight[section, None, :]
        population = np.exp(logits - logits.max(-1, keepdims=True))
        population /= population.sum(-1, keepdims=True)
        probabilities = population_probabilities(population, efficiency[section, None, :], data["metadata"])
        if np.any(probabilities <= 0) or not np.isfinite(probabilities).all():
            raise ValueError("Expected positive cell probabilities under the registered background law")
        log_likelihood = (counts[section, None] * np.log(probabilities)).sum(axis=(-2, -1))
        mle.append(population[np.arange(len(population)), log_likelihood.argmax(1)])
        prior.append((population[:, 1:-1].sum(1) + 0.5 * (population[:, 0] + population[:, -1]))
                     / (GRID_POINTS - 1))
        weights = np.exp(log_likelihood - log_likelihood.max(1, keepdims=True))
        weights[:, [0, -1]] *= 0.5  # Trapezoidal quadrature on the uniform E5 prior.
        weights /= weights.sum(1, keepdims=True)
        posterior.append((weights[..., None] * population).sum(1))
    calibrated_prior = np.concatenate(prior)
    return {"template_grid_mle": np.concatenate(mle), "template_grid_posterior_mean": np.concatenate(posterior),
            "calibration_prior_no_counts": calibrated_prior,
            "local_grid_shrinkage": local_likelihood_population(counts, efficiency, calibrated_prior, data["metadata"])}


def action_scores(truth, predicted, baseline):
    action = centered_action(predicted, ACTION_BOUND)
    gain = conditional_fidelity(truth, action) - baseline
    return {
        "population_sse": ((truth - predicted) ** 2).sum(1),
        "gain": gain,
        "action_bound_fraction": np.isclose(np.abs(action), ACTION_BOUND, atol=1e-12, rtol=0).mean(1),
        "rms_log_action": np.sqrt(np.square(action).mean(1)),
    }


def record_json(path, content):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_rows(args, cell, count, role):
    profile = PROFILES[args.profile]
    n_modes, shots, rounds = cell
    if args.bank_dir is None:
        return generate(n_modes, count, role, nqrec_root=args.nqrec_root, start=profile.row_start,
                        shots=shots, rounds=rounds, family=args.family)
    if __package__:
        from .amp_data import load_bank
    else:
        from amp_data import load_bank
    suffix = f"_c{shots}_k{rounds}" if len(profile.budgets) > 1 else ""
    data = load_bank(args.bank_dir / f"{role}_n{n_modes}{suffix}.npz")
    expected = {"n_modes": n_modes, "start": profile.row_start, "rows": count, "role": role,
                "rounds": rounds, "shots": shots, "cell_order": ["00", "10", "01", "11"]}
    if (data["metadata"].get("family", "e5") != args.family or
            any(data["metadata"].get(key) != value for key, value in expected.items())):
        raise ValueError("Input bank metadata does not match this registered profile")
    return data


def run(args):
    profile = PROFILES[args.profile]
    cells = [(n, shots, rounds) for n in profile.sizes for shots, rounds in profile.budgets]
    started = time.monotonic()
    def check_time():
        if time.monotonic() - started >= profile.seconds:
            raise TimeoutError(f"{args.profile} exceeded the {profile.seconds}-second budget")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The selected CUDA device is unavailable")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = PopulationModel(args.arm, args.family).to(device=device, dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    def tens(array):
        return torch.as_tensor(array, dtype=torch.float64, device=device)

    recipe = {
        "profile": args.profile, "arm": args.arm, "family": args.family, "seed": args.seed, "sizes_and_budgets": asdict(profile),
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine(),
        "python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
        "hidden": HIDDEN, "parameter_count": sum(p.numel() for p in model.parameters()),
        "optimizer": "Adam", "learning_rate": LEARNING_RATE,
        "log_amplitude_bound": ACTION_BOUND, "likelihood_grid_points": GRID_POINTS, "precision": "float64",
        "loss": "KL(true population || estimated population), synthetic supervised",
        "selection": "terminal weights at fixed cap", "training_schedule": "cycle (N, shots, rounds), sample rows with replacement",
        "input": "portable_banks" if args.bank_dir is not None else "local_generation",
        "negative_gain_tolerance": 1e-12, "likelihood_integration": "equally spaced grid with trapezoidal weights",
        "local_log_population_sd": LOCAL_LOG_POPULATION_SD, "local_grid_points": LOCAL_GRID_POINTS,
        "template_reference_scope": "correct E5 family" if args.family == "e5" else "E5-only misspecified family",
        "study_scope": "fresh development comparison; population settings, aligned pure W, phase action zero",
    }
    record_json(output / "status.json", {"state": "running", "pid": os.getpid(), "recipe": recipe})
    try:
        banks = {}
        for cell in cells:
            data = read_rows(args, cell, profile.train_rows, profile.train_role)
            banks[cell] = (tens(encode_observations(data)), tens(data["log_weight"]), tens(data["population"]), data["metadata"]["e5_lambda"])
            check_time()
        trained = time.monotonic()
        for step in range(profile.updates):
            check_time()
            x, w, truth, prior_range = banks[cells[step % len(cells)]]
            indices = torch.from_numpy(rng.integers(0, len(x), size=profile.batch_size)).to(device)
            log_prediction = model(x[indices], w[indices], prior_range)
            labels = truth[indices]
            loss = nn.functional.kl_div(log_prediction, labels, reduction="batchmean")
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        training_seconds = time.monotonic() - trained
        # A future explicit fork can load weights and optimizer on the same
        # device type, reconstruct the same banks, restore RNG, and start at
        # the recorded zero-based next step in the deterministic cell cycle.
        terminal = {
            "state_dict": {key: val.detach().cpu() for key, val in model.state_dict().items()},
            "optimizer": optimizer.state_dict(), "step": profile.updates,
            "next_cell_index": profile.updates % len(cells), "ordered_cells": cells,
            "numpy_sampling": rng.bit_generator.state, "torch_cpu_rng": torch.get_rng_state(),
            "torch_device_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            "recipe": recipe,
        }
        terminal_path = output / "terminal.pt"
        torch.save(terminal, terminal_path.with_suffix(".tmp"))
        os.replace(terminal_path.with_suffix(".tmp"), terminal_path)
        checkpoint = {"state_dict": model.cpu().state_dict(), "recipe": recipe, "step": profile.updates}
        torch.save(checkpoint, output / "model.pt")
        loaded = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
        restored = PopulationModel(args.arm, args.family).double()
        restored.load_state_dict(loaded["state_dict"])
        model = restored.to(device).eval()
        summary, arrays = {}, {}
        for cell in cells:
            check_time()
            n_modes, shots, rounds = cell
            data = read_rows(args, cell, profile.evaluation_rows, profile.evaluation_role)
            with torch.no_grad():
                pred = model(tens(encode_observations(data)), tens(data["log_weight"]), data["metadata"]["e5_lambda"]).exp().cpu().numpy()
            truth = data["population"]
            before = conditional_fidelity(truth, np.zeros_like(truth))
            predictions = likelihood_estimates(data)
            predictions["network"] = pred
            predictions["truth_matching_reference"] = truth
            size_key = f"n{n_modes}" if len(profile.budgets) == 1 else f"n{n_modes}_c{shots}_k{rounds}"
            arrays[size_key + "_before"] = before
            arrays[size_key + "_minimum_relative_population"] = n_modes * truth.min(axis=-1)
            arrays[size_key + "_centered_log_amplitude_rms"] = np.log(truth).std(axis=-1) * 0.5
            summary[size_key] = {"rows": len(truth), "mean_before": float(before.mean()), "methods": {}, "metadata": data["metadata"]}
            for method, estimate in predictions.items():
                scores = action_scores(truth, estimate, before)
                for key, values in scores.items():
                    arrays[f"{size_key}_{method}_{key}"] = values
                summary[size_key]["methods"][method] = {**{k: float(v.mean()) for k, v in scores.items()},
                    "harmed_rows": int(np.sum(scores["gain"] < -1e-12))}
            paired = arrays[f"{size_key}_network_gain"] - arrays[f"{size_key}_template_grid_posterior_mean_gain"]
            summary[size_key]["network_minus_posterior_gain"] = float(paired.mean())
            summary[size_key]["network_minus_local_shrinkage_gain"] = float(
                (arrays[f"{size_key}_network_gain"] - arrays[f"{size_key}_local_grid_shrinkage_gain"]).mean())
            calibration_gain = arrays[f"{size_key}_calibration_prior_no_counts_gain"]
            summary[size_key]["network_minus_no_count_prior_gain"] = float(
                (arrays[f"{size_key}_network_gain"] - calibration_gain).mean())
            check_time()
        np.savez_compressed(output / "paired_scores.npz", **arrays)
        result = {
            "state": "completed", "recipe": recipe, "updates": profile.updates,
            "updates_by_size": {str(n): profile.updates // len(profile.sizes) for n in profile.sizes},
            "updates_per_size_and_budget": profile.updates // len(cells),
            "elapsed_s": time.monotonic() - started, "training_s": training_seconds,
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
            "results": summary,
        }
        record_json(output / "result.json", result)
        record_json(output / "status.json", {"state": "completed", "elapsed_s": result["elapsed_s"], "updates": profile.updates})
    except BaseException as exc:
        record_json(output / "status.json", {"state": "failed", "error": repr(exc), "elapsed_s": time.monotonic() - started})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--arm", choices=("template", "flexible"), required=True)
    parser.add_argument("--family", choices=("e5", "structured"), default="e5")
    parser.add_argument("--seed", type=int, choices=(61, 62), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--nqrec-root", type=Path)
    inputs.add_argument("--bank-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
