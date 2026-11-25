"""Typed engineering-only scorer for Track B candidate logical actions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterator, Mapping

import math

import numpy as np
import torch

from ml_recon.data_v2_4.hashing import sha256_array
from ml_recon.data_v2_4.sample_keys import make_sample_key
from ml_recon.data_v2_4.synthetic.feedback_renderers import (
    TrackBRecedingHorizonForwardContextV1,
    evaluate_trackb_candidate_transition,
    replay_trackb_ten_round_forward_contexts,
)

from .contracts import canonical_sha256
from .inverse_transformer import TrackBInverseTransformerV1


MATERIAL_NEGATIVE_GAIN_TOLERANCE = 1e-10


def is_materially_negative_trackb_gain(gain: float) -> bool:
    """Classify harm while ignoring double-precision zero-action residue."""
    if not math.isfinite(float(gain)):
        raise ValueError("Track B candidate gain must be finite")
    return float(gain) < -MATERIAL_NEGATIVE_GAIN_TOLERANCE


@dataclass(frozen=True)
class TrackBCandidateActionScoreV1:
    """One exact-source synthetic forward score for a logical action candidate."""

    sample_id: str
    round_index: int
    candidate_kind: str
    pre_state_sha: str
    candidate_action_sha: str
    before_distance: float
    after_distance: float
    gain: float
    research_only: bool = True
    selection_evidence: bool = False
    promotion_ready: bool = False
    hardware_action_authorized: bool = False
    schema: str = "TrackBCandidateActionScoreV1"

    def validate(self) -> None:
        if (type(self.sample_id) is not str or not self.sample_id
                or type(self.round_index) is not int or not 0 <= self.round_index < 10
                or self.candidate_kind not in {"zero", "teacher", "model"}):
            raise ValueError("Track B candidate score identity is invalid")
        for value in (self.pre_state_sha, self.candidate_action_sha):
            if type(value) is not str or len(value) != 64:
                raise ValueError("Track B candidate score hashes are invalid")
        for value in (self.before_distance, self.after_distance, self.gain):
            if not math.isfinite(float(value)):
                raise ValueError("Track B candidate score is non-finite")
        if (not self.research_only or self.selection_evidence or self.promotion_ready
                or self.hardware_action_authorized):
            raise ValueError("candidate scoring cannot authorize selection, promotion, or hardware")
        if self.schema != "TrackBCandidateActionScoreV1":
            raise ValueError("unsupported Track B candidate score schema")

    def sha256(self) -> str:
        self.validate()
        return canonical_sha256(asdict(self))


def score_trackb_candidate_action(
    context: TrackBRecedingHorizonForwardContextV1,
    candidate_action: np.ndarray,
    *,
    candidate_kind: str,
) -> TrackBCandidateActionScoreV1:
    """Score zero, teacher, or model action against exactly one synthetic pre-state."""
    context.validate()
    value = evaluate_trackb_candidate_transition(context, candidate_action)
    result = TrackBCandidateActionScoreV1(
        sample_id=context.sample_id,
        round_index=context.round_index,
        candidate_kind=candidate_kind,
        pre_state_sha=context.pre_state_sha,
        candidate_action_sha=sha256_array(np.asarray(candidate_action, dtype=np.float64)),
        before_distance=float(value["before_distance"]),
        after_distance=float(value["after_distance"]),
        gain=float(value["gain"]),
    )
    result.validate()
    return result


def _model_inputs_for_round(
    row: Mapping[str, Any],
    *,
    round_index: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], np.ndarray]:
    history = row["history_tokens"].unsqueeze(0).to(device=device, dtype=torch.float32)
    history_mask = row["history_mask"].unsqueeze(0).to(device=device, dtype=torch.bool)
    target = row["target_mode_tokens"].unsqueeze(0).to(device=device, dtype=torch.float32)
    mode_mask = row["mode_mask"].unsqueeze(0).to(device=device, dtype=torch.bool)
    if (history.ndim != 4 or history.shape[1] != 10 or history.shape[-1] != 7
            or history_mask.shape != history.shape[:2] or not 0 <= round_index < history.shape[1]):
        raise ValueError("candidate evaluator requires a registered ten-round V2.4 history row")
    if not bool(history_mask[:, :round_index + 1].all()):
        raise ValueError("candidate evaluator round requires every preceding and current history record")
    target_action = history[0, round_index, :, 2:4].detach().cpu().numpy().astype(np.float64)
    return {
        "history_tokens": history[:, :round_index],
        "history_mask": history_mask[:, :round_index],
        "noisy_measurement_tokens": history[:, round_index, :, :2],
        "target_mode_tokens": target,
        "mode_mask": mode_mask,
    }, target_action


def _iter_trackb_model_action_contexts(
    model: TrackBInverseTransformerV1,
    rows: Mapping[int, Mapping[str, Any]],
    *,
    global_seed: int,
    generator_sha: str,
    cell_id: str,
    split: str,
    noise_family: str,
    eta: float,
    kappa: float,
    shot_mode: str,
    shot_budget: int,
    device: str,
) -> Iterator[tuple[TrackBRecedingHorizonForwardContextV1, np.ndarray]]:
    """Yield one frozen inverse-model action for each bound V2.4 pre-state."""
    if not isinstance(model, TrackBInverseTransformerV1) or not rows:
        raise ValueError("candidate evaluator requires a model and at least one V2.4 row")
    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()
    with torch.no_grad():
        for source_index in sorted(rows):
            row = rows[source_index]
            sample_id = str(row["sample_id"])
            parsed_index = int(sample_id.rsplit(":", 1)[-1])
            if parsed_index != int(source_index):
                raise ValueError("candidate evaluator row index does not match the V2.4 sample identity")
            key = make_sample_key(
                global_seed=global_seed,
                split=split,
                dataset_family="feedback_history",
                N=model.config.mode_count,
                index=parsed_index,
                generator_sha=generator_sha,
                cell_id=cell_id,
            )
            contexts = replay_trackb_ten_round_forward_contexts(
                key,
                shot_mode=shot_mode,
                shot_budget=shot_budget,
                noise_family=noise_family,
                eta=eta,
                kappa=kappa,
            )
            for round_index, context in enumerate(contexts):
                model_inputs, target_action = _model_inputs_for_round(
                    row, round_index=round_index, device=torch_device)
                if not np.allclose(context.teacher_action, target_action, atol=1e-6, rtol=0.0):
                    raise ValueError("candidate evaluator replay differs from the stored V2.4 teacher action")
                output = model(model_inputs)
                model_action = torch.stack((
                    output["amplitude_delta"][0], output["phase_delta"][0],
                ), dim=-1).detach().cpu().numpy()
                yield context, model_action


def evaluate_trackb_model_candidate_actions(
    model: TrackBInverseTransformerV1,
    rows: Mapping[int, Mapping[str, Any]],
    *,
    global_seed: int,
    generator_sha: str,
    cell_id: str,
    split: str,
    noise_family: str,
    eta: float,
    kappa: float,
    shot_mode: str,
    shot_budget: int,
    device: str = "cpu",
) -> dict[str, object]:
    """Score frozen model actions on the exact generated candidate pre-states.

    ``rows`` maps source row indices to V2.4 feedback rows.  It remains an
    engineering evaluator: the forward context is never passed back to model
    training and no external hardware state is involved.
    """
    by_round = [
        {"model_gain": [], "zero_gain": [], "teacher_gain": [], "model_negative": 0}
        for _ in range(10)
    ]
    sample_ids = []
    for context, model_action in _iter_trackb_model_action_contexts(
        model,
        rows,
        global_seed=global_seed,
        generator_sha=generator_sha,
        cell_id=cell_id,
        split=split,
        noise_family=noise_family,
        eta=eta,
        kappa=kappa,
        shot_mode=shot_mode,
        shot_budget=shot_budget,
        device=device,
    ):
        if context.round_index == 0:
            sample_ids.append(context.sample_id)
        model_score = score_trackb_candidate_action(
            context, model_action, candidate_kind="model")
        zero_score = score_trackb_candidate_action(
            context, np.zeros_like(model_action), candidate_kind="zero")
        teacher_score = score_trackb_candidate_action(
            context, context.teacher_action, candidate_kind="teacher")
        values = by_round[context.round_index]
        values["model_gain"].append(model_score.gain)
        values["zero_gain"].append(zero_score.gain)
        values["teacher_gain"].append(teacher_score.gain)
        values["model_negative"] += int(is_materially_negative_trackb_gain(model_score.gain))
    summaries = []
    for round_index, values in enumerate(by_round):
        model_gain = np.asarray(values["model_gain"], dtype=np.float64)
        zero_gain = np.asarray(values["zero_gain"], dtype=np.float64)
        teacher_gain = np.asarray(values["teacher_gain"], dtype=np.float64)
        summaries.append({
            "round_index": round_index + 1,
            "n": int(model_gain.size),
            "model_gain_mean": float(model_gain.mean()),
            "zero_gain_mean": float(zero_gain.mean()),
            "teacher_gain_mean": float(teacher_gain.mean()),
            "model_minus_zero_mean": float((model_gain - zero_gain).mean()),
            "model_negative_gain_rate": float(values["model_negative"] / model_gain.size),
        })
    return {
        "schema": "TrackBModelCandidateActionEvaluationV1",
        "sample_ids_sha": canonical_sha256(sample_ids),
        "sample_count": len(sample_ids),
        "rounds": summaries,
        "evaluation_kind": "SYNTHETIC_CANDIDATE_ACTION_FORWARD_ENGINEERING_ONLY",
        "research_only": True,
        "selection_evidence": False,
        "promotion_ready": False,
        "hardware_action_authorized": False,
    }


__all__ = [
    "TrackBCandidateActionScoreV1",
    "evaluate_trackb_model_candidate_actions",
    "is_materially_negative_trackb_gain",
    "score_trackb_candidate_action",
]
