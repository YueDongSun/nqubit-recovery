"""Fixed-point, non-mutating diagnostics for effective observation pilots."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch

from .contracts import canonical_json_bytes
from .effective_observation_loss import Effective_observation_loss_config


SHARED_ENCODER_PREFIXES = ("visible_encoder.", "attention.", "norms.", "command_encoder.")
QUANTILES = (0.01, 0.5, 0.99)
SNAPSHOT_SCHEMA = "TrackBEffectivePilotDiagnosticSnapshotV1"


@dataclass(frozen=True)
class Pilot_diagnostic_config:
    """Diagnostic plan does not change the optimizer budget or loss weights."""

    enabled: bool = False
    sample_updates: tuple[int, ...] = (1, 128, 512, 2000)
    schema: str = "TrackBEffectivePilotDiagnosticConfigV1"

    def validate(self) -> None:
        if type(self.enabled) is not bool or self.schema != "TrackBEffectivePilotDiagnosticConfigV1":
            raise ValueError("invalid pilot diagnostic configuration")
        if not self.sample_updates or tuple(sorted(set(self.sample_updates))) != self.sample_updates \
                or any(type(value) is not int or not 1 <= value <= 2000 for value in self.sample_updates):
            raise ValueError("diagnostic updates must be unique sorted integers in [1,2000]")

    def requested_updates(self, optimizer_steps: int) -> tuple[int, ...]:
        self.validate()
        if type(optimizer_steps) is not int or not 1 <= optimizer_steps <= 2000:
            raise ValueError("diagnostic optimizer budget must be an integer in [1,2000]")
        return tuple(sorted({value for value in self.sample_updates if value <= optimizer_steps} | {optimizer_steps}))


def _quantiles(value: torch.Tensor) -> list[float]:
    detached = value.detach().to(torch.float64).reshape(-1)
    if not detached.numel() or not bool(torch.isfinite(detached).all()):
        raise ValueError("diagnostic quantity must be nonempty and finite")
    return torch.quantile(detached, detached.new_tensor(QUANTILES)).cpu().tolist()


def capture_pilot_diagnostics(
    model: torch.nn.Module, prediction: Mapping[str, Any], losses: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any], *, loss_config: Effective_observation_loss_config,
    identity_sha: str, runtime_config_sha: str, update_step: int,
) -> dict[str, Any]:
    """Inspect the existing pre-update graph without backward accumulation.

No forward is repeated, no random samples are drawn, and ``autograd.grad``
does not change parameter ``.grad``. Retain the graph for the real optimizer
backward. Norms and angles refer to *weighted* loss contributions over the
same named shared encoder parameters, never to a reweighted training loss.
"""
    named = [(name, parameter) for name, parameter in model.named_parameters()
             if name.startswith(SHARED_ENCODER_PREFIXES) and parameter.requires_grad]
    if not named:
        raise ValueError("pilot diagnostics require the registered shared encoder")
    weights = {
        "four_cell_nll": 1.0, "global_multi_nll": loss_config.global_multi_weight,
        "effective_response_prior": loss_config.effective_response_prior_weight,
        "prepared_state_prior": loss_config.prepared_state_prior_weight,
    }
    vectors = {}
    for name, weight in weights.items():
        value = losses[name] * weight
        gradients = torch.autograd.grad(value, [parameter for _, parameter in named],
                                        retain_graph=True, allow_unused=True)
        vector = torch.cat([
            (torch.zeros_like(parameter) if gradient is None else gradient).detach().double().reshape(-1)
            for (_, parameter), gradient in zip(named, gradients, strict=True)
        ])
        if not bool(torch.isfinite(vector).all()):
            raise FloatingPointError(f"nonfinite diagnostic gradient for {name}")
        vectors[name] = vector
    norms = {name: float(vector.norm().cpu()) for name, vector in vectors.items()}
    cosines = {}
    for left, right in itertools.combinations(vectors, 2):
        denominator = norms[left] * norms[right]
        cosines[f"{left}__{right}"] = (
            None if denominator == 0 else max(-1.0, min(1.0, float((vectors[left].dot(vectors[right]) / denominator).cpu())))
        )
    query = batch["query_mask"] & batch["operation_mask"]
    target = batch["four_cell_counts"].detach().double()
    state = prediction["latent_state"].detach()
    from .training import trackb_model_state_sha
    probability_names = ("signal_probability", "coin_given_signal_probability", "idler_given_no_signal_probability")
    return {
        "schema": SNAPSHOT_SCHEMA,
        "identity_sha": identity_sha,
        "runtime_config_sha": runtime_config_sha,
        "evaluated_model_state_sha": trackb_model_state_sha(model),
        "evaluated_model_step": update_step - 1,
        "committed_update_step": update_step,
        "evaluation_phase": "PRE_UPDATE_GRAPH_RECORDED_AFTER_SUCCESSFUL_COMMIT",
        "sample_ids": list(batch["sample_ids"]),
        "batch_plan_sha": batch["batch_plan_sha"], "query_mask_sha": batch["query_mask_sha"],
        "epoch": batch["epoch"], "batch_index": batch["batch_index"],
        "shared_parameter_names": [name for name, _ in named],
        "weighted_loss_components": {name: float((losses[name].detach() * weight).cpu()) for name, weight in weights.items()},
        "weighted_shared_gradient_norms": norms,
        "weighted_shared_gradient_cosines": cosines,
        "quantile_levels": list(QUANTILES),
        "query_probability_quantiles": {name: _quantiles(prediction[name][query]) for name in probability_names},
        "concentration_quantiles": _quantiles(prediction["four_cell_concentration"]),
        "effective_offset_quantiles": _quantiles(prediction["effective_offsets"]),
        "effective_contrast_quantiles": _quantiles(prediction["effective_contrasts"]),
        "effective_state_purity_quantiles": _quantiles((state @ state).diagonal(dim1=-2, dim2=-1).sum(-1).real),
        "effective_state_min_eigenvalue": float(torch.linalg.eigvalsh(state).min().cpu()),
        "query_mean_absolute_error": float((target - prediction["four_cell_mean"].detach())[query].abs().mean().cpu()),
        "read_trials_semantics": prediction["read_trials_semantics"],
        "global_multi_available_rows": int(batch["global_multi_mask"].sum().cpu()),
        "scientific_gate_issued": False,
    }


class Pilot_diagnostic_recorder:
    """One run-owned directory of immutable, explicitly pre-update records."""

    def __init__(self, root: Path, *, identity_sha: str, runtime_config_sha: str, config: Pilot_diagnostic_config,
                 optimizer_steps: int, restored_step: int) -> None:
        self.root = root
        self.identity_sha = identity_sha
        self.runtime_config_sha = runtime_config_sha
        self.config = config
        self.updates = config.requested_updates(optimizer_steps)
        self.records: list[dict[str, Any]] = []
        if restored_step == 0:
            root.mkdir(parents=True, exist_ok=False)
        elif not root.is_dir():
            raise ValueError("diagnostic resume requires the original trajectory directory")
        for update in self.updates:
            path = root / f"update_{update:06d}.json"
            if update <= restored_step:
                if not path.is_file():
                    raise ValueError("diagnostic resume cannot fabricate missing historical samples")
                record = json.loads(path.read_bytes())
                if record.get("schema") != SNAPSHOT_SCHEMA or record.get("identity_sha") != identity_sha \
                        or record.get("runtime_config_sha") != runtime_config_sha \
                        or record.get("committed_update_step") != update \
                        or record.get("evaluated_model_step") != update - 1 \
                        or record.get("diagnostic_config") != json.loads(canonical_json_bytes(asdict(config))):
                    raise ValueError("diagnostic history identity or plan differs")
                self.records.append({"update_step": update, "path": str(path)})
            elif path.exists():
                raise ValueError("diagnostic history is ahead of the restored optimizer cursor")

    def record(self, update_step: int, losses: Mapping[str, Any]) -> None:
        snapshot = losses.get("diagnostic_snapshot")
        if update_step not in self.updates:
            if snapshot is not None:
                raise ValueError("unscheduled diagnostic snapshot")
            return
        if not isinstance(snapshot, dict) or snapshot.get("schema") != SNAPSHOT_SCHEMA \
                or snapshot.get("identity_sha") != self.identity_sha \
                or snapshot.get("runtime_config_sha") != self.runtime_config_sha \
                or snapshot.get("committed_update_step") != update_step \
                or snapshot.get("evaluated_model_step") != update_step - 1:
            raise ValueError("scheduled diagnostic snapshot missing or mismatched")
        payload = canonical_json_bytes({**snapshot, "diagnostic_config": asdict(self.config)})
        destination = self.root / f"update_{update_step:06d}.json"
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.root, prefix="diagnostic_", suffix=".tmp", delete=False) as stream:
                temporary_path = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            # Atomic create-if-absent: a historical snapshot is never replaced.
            os.link(temporary_path, destination)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        self.records.append({"update_step": update_step, "path": str(destination)})
