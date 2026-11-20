"""Paired synthetic experiments using the main classifier and registered step.

Fixed updates, public measurement budget, split seeds and the selected profile
are written before optimization. This is a development experiment, not a
reopening of any historical locked/test campaign or an apparatus experiment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Dict

import torch

from .classifier_n_full import V10ClassifierNFull
from .training_ideas import TrainingIdeasConfig
from .training_ideas_data import make_projector_training_batch
from .training_ideas_objective import normalized_state, projector_primary_losses


def _fingerprint(batch: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(batch):
        value = batch[name]
        digest.update(name.encode("utf-8"))
        if isinstance(value, torch.Tensor):
            item = value.detach().cpu().contiguous()
            digest.update(str(item.dtype).encode("ascii"))
            digest.update(str(tuple(item.shape)).encode("ascii"))
            digest.update(item.numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _take(batch: dict, indices: torch.Tensor) -> dict:
    row_count = batch["counts"].shape[0]
    return {
        key: (
            value[indices]
            if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == row_count
            else value
        )
        for key, value in batch.items()
    }


def _candidate_plugin_scores(output: dict, batch: dict) -> torch.Tensor:
    """Compare conditional fidelity after the same public candidate filters."""
    rho = normalized_state(output)
    target_parameters = batch["public_control_target"]
    target = target_parameters[..., 0] * torch.exp(1j * target_parameters[..., 1])
    target = target / target.norm(dim=-1, keepdim=True)
    actions = batch["public_candidate_actions"]
    diagonal = torch.exp(torch.complex(actions[..., 0], actions[..., 1]))
    # Uniform attenuation does not affect conditional fidelity; this rescaling
    # makes each displayed filter trace nonincreasing.
    diagonal = diagonal / diagonal.abs().amax(-1, keepdim=True)
    acted = diagonal[..., :, None] * rho[:, None] * diagonal.conj()[..., None, :]
    survival = acted.diagonal(dim1=-2, dim2=-1).real.sum(-1)
    if torch.any(survival <= 0):
        raise ValueError("candidate filter produced zero conditional support")
    fidelity = torch.einsum("bi,bkij,bj->bk", target.conj(), acted, target).real / survival
    valid = batch["public_candidate_valid_mask"].bool()
    return fidelity.masked_fill(~valid, -torch.inf)


def _rho8_diagnostics(batch: dict) -> dict:
    """Expose correlated-view and PSD-projection limits without changing loss."""
    if "rho8_effective_unique_views" not in batch:
        return {"available": False}
    valid = batch["rho8_view_mask"].bool()

    def histogram(value: torch.Tensor) -> dict:
        labels, counts = value.detach().cpu().unique(return_counts=True)
        return {str(int(label)): int(count) for label, count in zip(labels, counts)}

    def masked_mean(name: str):
        values = batch[name][valid].float()
        return float(values.mean()) if values.numel() else None

    unique_views = batch["rho8_effective_unique_views"]
    return {
        "available": True,
        "valid_sample_fraction": float(batch["rho8_sample_mask"].float().mean()),
        "effective_unique_views_mean": float(unique_views.float().mean()),
        "effective_unique_views_histogram": histogram(unique_views),
        "unique_compensation_maps_histogram": histogram(batch["rho8_unique_compensation_maps"]),
        "unit_map_fraction_by_family": batch["rho8_unit_map_mask"].float().mean(0).cpu().tolist(),
        "map_assumption_code_histogram": histogram(batch["rho8_map_assumption_code"]),
        "uniform_view_weight_fraction": float(batch["rho8_uniform_view_weights"].float().mean()),
        "psd_projected_fraction_of_valid_views": masked_mean("rho8_psd_projected_mask"),
        "discarded_negative_mass_mean": masked_mean("rho8_discarded_negative_mass"),
        "psd_correction_frobenius_mean": masked_mean("rho8_psd_correction_frobenius"),
        "psd_relative_correction_mean": masked_mean("rho8_psd_relative_correction"),
    }


@torch.no_grad()
def evaluate_training_ideas(model, batch: dict, *, N: int) -> tuple[dict, dict]:
    output = model(batch, N=N, fine_identifiable=False)
    losses = projector_primary_losses(output, batch)
    rho = normalized_state(output)
    truth = torch.complex(batch["target_R1_re"], batch["target_R1_im"])
    state_risk = (rho - truth).abs().square().sum((-1, -2))
    plugin_selected = _candidate_plugin_scores(output, batch).argmax(-1)
    selected = plugin_selected
    noop = torch.as_tensor(batch["public_noop_candidate_index"], device=selected.device).long()
    noop = noop.expand_as(selected)
    selector = "physical_plugin_candidate_fidelity"
    if "idea_fd_logits" in output:
        valid = batch["public_candidate_valid_mask"].bool()
        score = output["idea_fd_logits"].masked_fill(~valid, -torch.inf)
        selected = score.argmax(-1)
        estimated_harm = torch.sigmoid(output["idea_fd_harm_logits"])
        selected_harm = estimated_harm.gather(-1, selected[:, None]).squeeze(-1)
        selected = torch.where(selected_harm > 0.5, noop, selected)
        selector = "learned_fd_candidate_with_fixed_half_harm_gate"
    utility = batch["target_fd_utility"].gather(-1, selected[:, None]).squeeze(-1)
    harm = batch["target_fd_harm"].gather(-1, selected[:, None]).squeeze(-1)
    executed = selected != noop
    plugin_utility = batch["target_fd_utility"].gather(-1, plugin_selected[:, None]).squeeze(-1)
    plugin_harm = batch["target_fd_harm"].gather(-1, plugin_selected[:, None]).squeeze(-1)
    plugin_executed = plugin_selected != noop
    valid = batch["public_candidate_valid_mask"].bool()
    oracle = batch["target_fd_utility"].masked_fill(~valid, -torch.inf).amax(-1)
    noise_mask = batch["target_operational_noise_mask"].bool()
    correct = output["operational_noise_logits"].argmax(-1) == batch["target_operational_noise"]
    noise_accuracy = float(correct[noise_mask].float().mean()) if bool(noise_mask.any()) else None
    summary = {key: float(value.detach()) for key, value in losses.items()}
    summary.update(
        state_risk=float(state_risk.mean()),
        selected_conditional_gain=float(utility.mean()),
        selected_harm_rate=float(harm.float().mean()),
        execution_coverage=float(executed.float().mean()),
        harm_given_execution=float(harm[executed].float().mean()) if bool(executed.any()) else None,
        candidate_regret=float((oracle - utility).mean()),
        physical_plugin_conditional_gain=float(plugin_utility.mean()),
        physical_plugin_harm_rate=float(plugin_harm.float().mean()),
        same_model_plugin_coverage=float(plugin_executed.float().mean()),
        physical_plugin_harm_given_execution=(
            float(plugin_harm[plugin_executed].float().mean())
            if bool(plugin_executed.any())
            else None
        ),
        physical_plugin_candidate_regret=float((oracle - plugin_utility).mean()),
        noise_accuracy=noise_accuracy,
        selector=selector,
        available_noise_classes=sorted(
            batch["target_operational_noise"][noise_mask].unique().cpu().tolist()
        ),
    )
    pair_mask = batch["target_pair_mask"].bool()
    pair_error = (output["pair_entanglement_probs"] - batch["target_pair"]).square()
    summary["esm_pair_brier"] = float(pair_error[pair_mask].mean())
    if "idea_csm_gram" in output:
        teacher = batch["target_csm_gram"]
        prediction = output["idea_csm_gram"]
        mask = batch["target_csm_mask"].bool()
        error = (prediction - teacher).square().sum((-1, -2))
        scale = teacher.square().sum((-1, -2))
        summary["csm_relative_squared_error"] = float((error[mask] / scale[mask]).mean())
    samples = {
        "state_risk": state_risk.cpu().tolist(),
        "selected_conditional_gain": utility.cpu().tolist(),
        "selected_harm": harm.cpu().tolist(),
        "executed": executed.cpu().tolist(),
        "candidate_regret": (oracle - utility).cpu().tolist(),
        "selected_candidate": selected.cpu().tolist(),
        "physical_plugin_selected_candidate": plugin_selected.cpu().tolist(),
        "physical_plugin_conditional_gain": plugin_utility.cpu().tolist(),
        "physical_plugin_harm": plugin_harm.cpu().tolist(),
        "physical_plugin_executed": plugin_executed.cpu().tolist(),
        "physical_plugin_candidate_regret": (oracle - plugin_utility).cpu().tolist(),
    }
    for key in (
        "rho8_effective_unique_views",
        "rho8_unique_compensation_maps",
        "rho8_unit_map_mask",
    ):
        if key in batch:
            samples[key] = batch[key].cpu().tolist()
    return summary, samples
