"""Opt-in training ideas with explicit observation/teacher boundaries.

This module belongs to the six-class V10 operational taxonomy.  Its MoE is
not a relabeling of the ten-class nqrec R6 identifier.  No target_* field is
read by the forward path.  Teachers enter only the loss functions below.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

TRAINING_IDEAS_PROFILES = (
    "baseline",
    "noise_moe",
    "rho8",
    "esm",
    "csm",
    "laplacian",
    "fd",
    "all",
    "all_no_noise_moe",
    "all_no_rho8",
    "all_no_esm",
    "all_no_laplacian",
    "all_no_fd",
)
NOISE_CLASS_NAMES = (
    "phase",
    "amplitude",
    "mixed_coherent",
    "decoherence_leakage",
    "count_background",
    "mechanism_shift",
)
RHO8_VIEW_NAMES = tuple(
    f"rho_{estimator}_{compensation}"
    for estimator in ("cs_minus_ir", "cs")
    for compensation in ("retrieval_raw", "retrieval_sqrt", "idler_raw", "idler_sqrt")
)


@dataclass(frozen=True)
class TrainingIdeasConfig:
    profile: str = "baseline"
    noise_moe: bool = False
    rho8: bool = False
    esm: bool = False
    csm: bool = False
    laplacian: bool = False
    fd: bool = False
    noise_residual_scale: float = 0.1
    noise_severity_weight: float = 0.05
    noise_balance_weight: float = 0.01
    rho8_weight: float = 1.0
    rho8_max_fraction: float = 0.05
    esm_weight: float = 0.1
    csm_weight: float = 0.1
    laplacian_weight: float = 0.01
    rsm_weight: float = 0.05
    fd_weight: float = 0.1
    fd_harm_weight: float = 0.1
    geometry_rank: int = 8

    def __post_init__(self) -> None:
        if self.profile not in TRAINING_IDEAS_PROFILES:
            raise ValueError(f"unknown training ideas profile: {self.profile}")
        for name in ("noise_moe", "rho8", "esm", "csm", "laplacian", "fd"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be Boolean")
        for name in (
            "noise_residual_scale",
            "noise_severity_weight",
            "noise_balance_weight",
            "rho8_weight",
            "rho8_max_fraction",
            "esm_weight",
            "csm_weight",
            "laplacian_weight",
            "rsm_weight",
            "fd_weight",
            "fd_harm_weight",
        ):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.rho8_max_fraction > 0.05:
            raise ValueError("the combined rho8 contribution cannot exceed five percent")
        if self.rho8_weight > 1.0:
            raise ValueError("rho8_weight must not enlarge the capped weak-teacher contribution")
        if type(self.geometry_rank) is not int or self.geometry_rank < 1:
            raise ValueError("geometry_rank must be a positive integer")

    @classmethod
    def from_profile(cls, profile: str = "baseline", **overrides) -> "TrainingIdeasConfig":
        if profile not in TRAINING_IDEAS_PROFILES:
            raise ValueError(f"unknown training ideas profile: {profile}")
        names = ("noise_moe", "rho8", "esm", "csm", "laplacian", "fd")
        all_family = profile == "all" or profile.startswith("all_no_")
        flags = {name: all_family or profile == name for name in names}
        if profile.startswith("all_no_"):
            flags[profile.removeprefix("all_no_")] = False
        if profile == "fd":
            flags["csm"] = True
        return replace(cls(profile=profile, **flags), **overrides)

    for_profile = from_profile

    @property
    def profile_name(self) -> str:
        return self.profile

    @property
    def enabled(self) -> bool:
        return any((self.noise_moe, self.rho8, self.esm, self.csm, self.laplacian, self.fd))

    def to_dict(self) -> dict:
        return asdict(self)


def _mode_panel(fused: torch.Tensor, mode_mask: torch.Tensor) -> torch.Tensor:
    if fused.ndim != 3 or mode_mask.shape != fused.shape[:2] or mode_mask.dtype != torch.bool:
        raise ValueError("fused/mode_mask must have shapes [B,N,w]/[B,N]")
    if not bool(mode_mask.any(-1).all()):
        raise ValueError("each row needs an active mode")
    if not bool(torch.isfinite(fused[mode_mask]).all()):
        raise ValueError("active mode features must be finite")
    return torch.where(mode_mask[..., None], fused, torch.zeros_like(fused))


def _pool(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (value * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)


class TrainingIdeasModule(nn.Module):
    """Trainable optional trunk residual, target-free geometry and decision heads.

    Instantiate after the existing baseline heads to retain their seeded
    initialization.  Baseline creates no parameters and consumes no RNG.
    CSM coordinate packing is [a0, phi0, a1, phi1, ...].
    """

    def __init__(self, width: int, config: TrainingIdeasConfig):
        super().__init__()
        if type(width) is not int or width < 1 or not isinstance(config, TrainingIdeasConfig):
            raise ValueError("width and TrainingIdeasConfig are required")
        self.width, self.config = width, config
        if config.noise_moe:
            self.noise_router = nn.Linear(width, len(NOISE_CLASS_NAMES))
            self.noise_experts = nn.ModuleList(
                nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
                for _ in NOISE_CLASS_NAMES
            )
            self.noise_strength_heads = nn.ModuleList(
                nn.Linear(width, 1) for _ in NOISE_CLASS_NAMES
            )
        if config.csm or config.fd:
            self.geometry_factor = nn.Linear(width, 2 * config.geometry_rank)
            self.geometry_diagonal = nn.Linear(width, 2)
        if config.fd:
            self.decision_mode_encoder = nn.Sequential(
                nn.Linear(width + 9, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU()
            )
            self.rsm_head = nn.Linear(width, 1)
            self.candidate_mode_encoder = nn.Sequential(
                nn.Linear(width + 4, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU()
            )
            self.candidate_head = nn.Sequential(
                nn.Linear(width + 5, width), nn.SiLU(), nn.Linear(width, 2)
            )

    def forward(
        self, fused: torch.Tensor, mode_mask: torch.Tensor, batch: Mapping | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not self.config.noise_moe:
            return fused, {}
        x = _mode_panel(fused, mode_mask)
        pooled = _pool(x, mode_mask)
        logits = self.noise_router(pooled)
        gates = torch.softmax(logits, -1)
        experts = torch.stack([expert(x) for expert in self.noise_experts], dim=2)
        residual = (experts * gates[:, None, :, None]).sum(2)
        residual = torch.tanh(residual) * self.config.noise_residual_scale
        enhanced = fused + torch.where(mode_mask[..., None], residual, torch.zeros_like(residual))
        strengths = torch.sigmoid(
            torch.cat([head(pooled) for head in self.noise_strength_heads], dim=-1)
        )
        balance = (gates.mean(0) - 1.0 / len(NOISE_CLASS_NAMES)).square().mean()
        return enhanced, {
            "operational_noise_logits": logits,
            "operational_noise_posterior": gates,
            "idea_noise_gates": gates,
            "idea_noise_strengths": strengths,
            "idea_noise_balance": balance,
            "idea_noise_residual": residual,
        }

    def _geometry(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        weight = mask.to(x.dtype)
        factor = self.geometry_factor(x).reshape(b, n, 2, self.config.geometry_rank)
        factor = factor * weight[:, :, None, None]
        mean = factor.sum(1, keepdim=True) / weight.sum(1)[:, None, None, None]
        factor = (factor - mean) * weight[:, :, None, None]
        factor = factor.reshape(b, 2 * n, self.config.geometry_rank)
        diagonal = F.softplus(self.geometry_diagonal(x)).reshape(b, 2 * n)
        p = (
            torch.diag_embed(weight)
            - weight[:, :, None] * weight[:, None, :] / weight.sum(-1)[:, None, None]
        )
        identity2 = torch.eye(2, dtype=x.dtype, device=x.device)
        q = torch.einsum("bij,kl->bikjl", p, identity2).reshape(b, 2 * n, 2 * n)
        gram = factor @ factor.transpose(-1, -2) + q @ torch.diag_embed(diagonal) @ q
        return 0.5 * (gram + gram.transpose(-1, -2))

    def decision_heads(
        self,
        fused: torch.Tensor,
        state_outputs: Mapping[str, torch.Tensor],
        batch: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if not (self.config.csm or self.config.fd):
            return {}
        if "mode_mask" not in batch:
            raise ValueError("geometry needs the public mode_mask")
        mask = batch["mode_mask"]
        x = _mode_panel(fused, mask)
        gram = self._geometry(x, mask)
        result = {"idea_csm_gram": gram}
        if not self.config.fd:
            return result
        required = {
            "public_control_target",
            "public_candidate_actions",
            "public_candidate_valid_mask",
            "public_noop_candidate_index",
        }
        missing = required - set(batch)
        if missing:
            raise ValueError(f"FD requires public decision inputs: {sorted(missing)}")
        target = batch["public_control_target"]
        actions = batch["public_candidate_actions"]
        valid = batch["public_candidate_valid_mask"]
        b, n, _ = x.shape
        if (
            target.shape != (b, n, 2)
            or actions.ndim != 4
            or actions.shape[0] != b
            or actions.shape[2:] != (n, 2)
        ):
            raise ValueError("public target/actions must be [B,N,2]/[B,K,N,2]")
        k = actions.shape[1]
        if (
            k < 1
            or valid.shape != (b, k)
            or valid.dtype != torch.bool
            or not bool(valid.any(-1).all())
        ):
            raise ValueError("candidate validity must be nonempty Boolean [B,K]")
        if not bool(torch.isfinite(target).all()) or not bool(torch.isfinite(actions[valid]).all()):
            raise ValueError("public targets and valid actions must be finite")
        if bool((target[..., 0] < 0).any()):
            raise ValueError("target amplitude must be nonnegative")
        noop = torch.as_tensor(batch["public_noop_candidate_index"], device=x.device)
        if noop.ndim == 0:
            noop = noop.expand(b)
        if (
            noop.shape != (b,)
            or noop.dtype not in (torch.int32, torch.int64)
            or bool(((noop < 0) | (noop >= k)).any())
        ):
            raise ValueError("public_noop_candidate_index must be integer [B] or scalar")
        rows = torch.arange(b, device=x.device)
        if not bool(valid[rows, noop].all()) or not bool((actions[rows, noop].abs() <= 1e-8).all()):
            raise ValueError("the designated no-op must be valid and zero")
        physics_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        target = target.to(physics_dtype)
        amplitude = target[..., 0] * mask
        norm = torch.linalg.vector_norm(amplitude, dim=-1, keepdim=True)
        if bool((norm <= 0).any()):
            raise ValueError("public target needs nonzero active support")
        amplitude = amplitude / norm
        # A zero-amplitude mode has no physical phase.  Fix the public target
        # gauge at its largest occupied mode, not merely the first active cell.
        reference_mode = amplitude.argmax(-1)
        phase = target[..., 1] - target[rows, reference_mode, 1][:, None]
        target_support = amplitude > 0
        target_complex = torch.complex(amplitude * torch.cos(phase), amplitude * torch.sin(phase))
        required_state = {"R1_re", "R1_im", "p1"}
        if not required_state <= set(state_outputs):
            raise ValueError("FD requires current predicted R1 and p1")
        state_prefix = "conditional_R1" if "conditional_R1_re" in state_outputs else "R1"
        rho = torch.complex(
            state_outputs[state_prefix + "_re"].to(physics_dtype),
            state_outputs[state_prefix + "_im"].to(physics_dtype),
        )
        if rho.shape != (b, n, n) or state_outputs["p1"].shape != (b,):
            raise ValueError("current state shape is inconsistent with modes")
        trace = rho.diagonal(dim1=-2, dim2=-1).real.sum(-1)
        if not bool(torch.isfinite(trace).all()) or bool((trace <= 0).any()):
            raise ValueError("FD needs a finite positive current single-excitation trace")
        # Match the primary conditional-state objective exactly.  Using a
        # separately floored p1 would change the state when a sector is rare.
        rho = rho / trace[:, None, None]
        population = rho.diagonal(dim1=-2, dim2=-1).real
        coherence = rho.abs().sum(-1) / mask.sum(-1, keepdim=True)
        alignment = target_complex.conj() * torch.einsum("bij,bj->bi", rho, target_complex)
        geom_diag = gram.diagonal(dim1=-2, dim2=-1).reshape(b, n, 2)
        numeric = torch.stack(
            (
                population,
                coherence,
                amplitude,
                torch.sin(phase) * target_support,
                torch.cos(phase) * target_support,
                alignment.real,
                alignment.imag,
                geom_diag[..., 0],
                geom_diag[..., 1],
            ),
            -1,
        ).to(x.dtype)
        decision_modes = self.decision_mode_encoder(torch.cat((x, numeric), -1))
        context = _pool(decision_modes, mask)
        fraction = torch.sigmoid(self.rsm_head(context)).squeeze(-1)
        actions = torch.where(
            valid[:, :, None, None], actions.to(x.dtype), torch.zeros_like(actions, dtype=x.dtype)
        )
        actions = actions * mask[:, None, :, None]
        actions = actions - actions.sum(2, keepdim=True) / mask.sum(-1)[:, None, None, None]
        actions = actions * mask[:, None, :, None]
        candidate_features = torch.cat(
            (decision_modes[:, None].expand(-1, k, -1, -1), actions, actions.square()), -1
        )
        candidate_modes = self.candidate_mode_encoder(candidate_features)
        candidate_context = (candidate_modes * mask[:, None, :, None]).sum(2) / mask.sum(-1)[
            :, None, None
        ]
        flat_action = actions.reshape(b, k, 2 * n)
        quadratic = torch.einsum("bki,bij,bkj->bk", flat_action, gram, flat_action)
        trace = gram.diagonal(dim1=-2, dim2=-1).sum(-1)
        overlap = alignment.real.sum(-1)
        global_features = torch.stack(
            (
                torch.log1p(quadratic.clamp_min(0)),
                flat_action.square().sum(-1),
                fraction[:, None].expand(-1, k),
                overlap[:, None].expand(-1, k),
                torch.log1p(trace.clamp_min(0))[:, None].expand(-1, k),
            ),
            -1,
        )
        score = self.candidate_head(torch.cat((candidate_context, global_features), -1))
        utility = score[..., 0] - score[rows, noop, 0][:, None]
        harm = score[..., 1]
        noop_mask = F.one_hot(noop, k).bool()
        harm = torch.where(noop_mask, torch.full_like(harm, -20.0), harm)
        result.update(
            {
                "idea_rsm_fraction": fraction,
                "idea_fd_logits": utility,
                "idea_fd_harm_logits": harm,
                "idea_fd_valid_mask": valid,
                "idea_fd_observability": quadratic,
            }
        )
        return result


def differentiable_rho8_loss(
    predicted_rho: torch.Tensor,
    views: torch.Tensor,
    view_weights: torch.Tensor,
    *,
    view_mask: torch.Tensor | None = None,
    sample_mask: torch.Tensor | None = None,
    primary_count_loss: torch.Tensor,
    max_fraction: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Detached weak teachers; detached cap rescaling retains student gradients."""
    if (
        predicted_rho.ndim != 3
        or not predicted_rho.is_complex()
        or predicted_rho.shape[-1] != predicted_rho.shape[-2]
    ):
        raise ValueError("predicted_rho must be complex [B,N,N]")
    b, n, _ = predicted_rho.shape
    if views.shape != (b, 8, n, n) or not views.is_complex() or view_weights.shape != (b, 8):
        raise ValueError("eight complex views and weights must have [B,8,N,N]/[B,8]")
    if not math.isfinite(max_fraction) or not 0 <= max_fraction <= 0.05:
        raise ValueError("rho8 cap must be between zero and five percent")
    vm = (
        torch.ones((b, 8), dtype=torch.bool, device=predicted_rho.device)
        if view_mask is None
        else view_mask
    )
    sm = (
        torch.ones(b, dtype=torch.bool, device=predicted_rho.device)
        if sample_mask is None
        else sample_mask
    )
    if vm.shape != (b, 8) or sm.shape != (b,) or vm.dtype != torch.bool or sm.dtype != torch.bool:
        raise ValueError("rho8 masks must be Boolean")
    active = vm & sm[:, None]
    weight = view_weights.detach().to(predicted_rho.real.dtype)
    if not bool(torch.isfinite(weight[active]).all()) or bool((weight[active] < 0).any()):
        raise ValueError("active view weights must be finite and nonnegative")
    teacher = views.detach().to(predicted_rho.dtype)
    if not bool(torch.isfinite(teacher[active]).all()) or not bool(
        torch.isfinite(predicted_rho[sm]).all()
    ):
        raise ValueError("active predicted/teacher states must be finite")
    teacher = torch.where(active[..., None, None], teacher, torch.zeros_like(teacher))
    weight = torch.where(active, weight, torch.zeros_like(weight))
    weight_sum = weight.sum(-1)
    valid_sample = sm & (weight_sum > 0)
    weight = weight / weight_sum.clamp_min(torch.finfo(weight.dtype).tiny)[:, None]
    safe_prediction = torch.where(
        valid_sample[:, None, None], predicted_rho, torch.zeros_like(predicted_rho)
    )
    difference = safe_prediction[:, None] - teacher
    hs = difference.abs().square().sum((-1, -2))
    diagonal = difference.diagonal(dim1=-2, dim2=-1).real.abs().sum(-1)
    offdiag = ~torch.eye(n, dtype=torch.bool, device=predicted_rho.device)
    coherence = (difference.real.abs() + difference.imag.abs())[:, :, offdiag].sum(-1)
    per_row = (weight * (hs + diagonal + coherence)).sum(-1)
    raw = (per_row * valid_sample).sum() / valid_sample.sum().clamp_min(1)
    primary = torch.as_tensor(primary_count_loss, device=raw.device, dtype=raw.dtype)
    if not bool(torch.isfinite(primary).all()) or bool((primary < 0).any()):
        raise ValueError("primary count loss must be finite and nonnegative")
    if primary.ndim == 0:
        primary_mean = primary
    elif primary.shape == (b,):
        primary_mean = (primary * valid_sample).sum() / valid_sample.sum().clamp_min(1)
    else:
        raise ValueError("primary_count_loss must be scalar or one value per sample")
    cap = max_fraction * primary_mean.detach()
    scale = torch.minimum(
        torch.ones_like(raw), cap / raw.detach().clamp_min(torch.finfo(raw.dtype).tiny)
    ).detach()
    return {
        "rho8_raw": raw,
        "rho8_applied": raw * scale,
        "rho8_scale": scale,
        "rho8_valid_samples": valid_sample.sum().to(raw.dtype),
    }


def compute_training_ideas_losses(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    config: TrainingIdeasConfig,
    primary_count_loss: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Only auxiliary losses; geometry/FD teacher losses live in their owner module."""
    anchor = outputs["R1_re"]
    zero = anchor.sum() * 0
    result = {
        "ideas_rho8": zero,
        "ideas_noise_severity": zero,
        "ideas_noise_balance": zero,
        "ideas_rho8_valid_samples": zero.detach(),
        "ideas_noise_severity_valid_samples": zero.detach(),
    }
    if config.rho8:
        required = {
            "rho8_views_re",
            "rho8_views_im",
            "rho8_weights",
            "rho8_view_mask",
            "rho8_sample_mask",
            "rho8_visible_only",
        }
        if not required <= set(batch):
            raise ValueError(
                f"rho8 training requires visible-only teacher fields: {sorted(required-set(batch))}"
            )
        provenance = batch["rho8_visible_only"]
        if (
            provenance.dtype != torch.bool
            or provenance.shape != anchor.shape[:1]
            or not bool(provenance.all())
        ):
            raise ValueError("rho8 teachers must be rebuilt from visible operations only")
        state_prefix = "conditional_R1" if "conditional_R1_re" in outputs else "R1"
        prediction = torch.complex(outputs[state_prefix + "_re"].float(), outputs[state_prefix + "_im"].float())
        trace = prediction.diagonal(dim1=-2, dim2=-1).real.sum(-1)
        if not bool(torch.isfinite(trace).all()) or bool((trace <= 0).any()):
            raise ValueError("rho8 needs a finite positive predicted single-excitation trace")
        prediction = prediction / trace[:, None, None]
        views = torch.complex(batch["rho8_views_re"].float(), batch["rho8_views_im"].float())
        terms = differentiable_rho8_loss(
            prediction,
            views,
            batch["rho8_weights"],
            view_mask=batch["rho8_view_mask"],
            sample_mask=batch["rho8_sample_mask"],
            primary_count_loss=primary_count_loss,
            max_fraction=config.rho8_max_fraction,
        )
        result["ideas_rho8"] = terms["rho8_applied"]
        result["ideas_rho8_valid_samples"] = terms["rho8_valid_samples"]
        result["ideas_rho8_raw"] = terms["rho8_raw"]
        result["ideas_rho8_scale"] = terms["rho8_scale"]
    if config.noise_moe:
        if "idea_noise_balance" not in outputs:
            raise ValueError("noise_moe profile needs trainable MoE forward outputs")
        result["ideas_noise_balance"] = outputs["idea_noise_balance"]
        if "target_noise_severity" in batch:
            needed = {
                "target_noise_severity_mask",
                "target_operational_noise",
                "target_operational_noise_mask",
            }
            if not needed <= set(batch):
                raise ValueError("severity supervision requires explicit severity and class masks")
            target = batch["target_noise_severity"].detach()
            labels = batch["target_operational_noise"].detach().long()
            mask = (
                batch["target_noise_severity_mask"].bool()
                & batch["target_operational_noise_mask"].bool()
            )
            if (
                target.shape != anchor.shape[:1]
                or labels.shape != target.shape
                or mask.shape != target.shape
            ):
                raise ValueError("noise targets require one value per sample")
            if bool(((labels[mask] < 0) | (labels[mask] >= len(NOISE_CLASS_NAMES))).any()):
                raise ValueError("noise label lies outside the six-class operational taxonomy")
            if not bool(torch.isfinite(target[mask]).all()) or bool(
                ((target[mask] < 0) | (target[mask] > 1)).any()
            ):
                raise ValueError("severity is normalized to [0,1]")
            selected = (
                outputs["idea_noise_strengths"].gather(-1, labels.clamp(0, 5)[:, None]).squeeze(-1)
            )
            safe_target = torch.where(mask, target, torch.zeros_like(target))
            result["ideas_noise_severity"] = (
                (selected - safe_target).square() * mask
            ).sum() / mask.sum().clamp_min(1)
            result["ideas_noise_severity_valid_samples"] = mask.sum().to(anchor.dtype)
    result["ideas_total"] = (
        config.rho8_weight * result["ideas_rho8"]
        + config.noise_severity_weight * result["ideas_noise_severity"]
        + config.noise_balance_weight * result["ideas_noise_balance"]
    )
    return result


@torch.no_grad()
def build_visible_rho8_views(
    counts: torch.Tensor,
    analyzers: torch.Tensor,
    visible_mask: torch.Tensor,
    shots: torch.Tensor,
    *,
    retrieval_map: torch.Tensor | None = None,
    idler_map: torch.Tensor | None = None,
    ridge: float = 1e-3,
) -> dict[str, torch.Tensor]:
    """Two ratio estimators x four public compensation hypotheses.

    Counts use 00,10,01,11. Hidden entries are removed before any feature or
    reconstruction. The dual ridge solver uses visible projectors only;
    teacher states are detached, PSD and trace one. Weights are uniform over
    valid views, not target-fidelity selection or a claimed cross-fit gate.
    Linear reconstructions may be non-PSD. Their discarded negative mass and
    projection norms are explicitly returned, never silently reinterpreted as
    a physical input state. Unit compensation hypotheses can duplicate views;
    the effective numerical view count is reported separately from eight slots.
    """
    if counts.ndim != 3 or counts.shape[-1] != 4:
        raise ValueError("counts must be [B,S,4]")
    b, s, _ = counts.shape
    if analyzers.ndim != 3 or analyzers.shape[:2] != (b, s) or not analyzers.is_complex():
        raise ValueError("analyzers must be complex [B,S,N]")
    n = analyzers.shape[-1]
    if (
        n < 2
        or visible_mask.shape != (b, s)
        or visible_mask.dtype != torch.bool
        or shots.shape != (b, s)
    ):
        raise ValueError("visible mask/shots must match the operation panel")
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    device = counts.device
    if analyzers.device != device or visible_mask.device != device or shots.device != device:
        raise ValueError("teacher inputs must share one device")
    observed = torch.where(
        visible_mask[..., None],
        counts.to(torch.float64),
        torch.zeros_like(counts, dtype=torch.float64),
    )
    trials = torch.where(
        visible_mask, shots.to(torch.float64), torch.zeros_like(shots, dtype=torch.float64)
    )
    modes = torch.where(
        visible_mask[..., None],
        analyzers.to(torch.complex128),
        torch.zeros_like(analyzers, dtype=torch.complex128),
    )
    if (
        not bool(torch.isfinite(observed).all())
        or bool((observed < 0).any())
        or not bool(torch.isfinite(trials).all())
    ):
        raise ValueError("visible observations must be finite and nonnegative")
    if not bool(torch.isfinite(modes).all()) or bool((trials < 0).any()):
        raise ValueError("visible analyzers/trials must be finite and nonnegative")
    if not torch.allclose(
        observed.sum(-1)[visible_mask], trials[visible_mask], atol=1e-6, rtol=1e-6
    ):
        raise ValueError("four-cell counts must sum to their native trials")
    norm = torch.linalg.vector_norm(modes, dim=-1)
    active = visible_mask & (trials > 0) & (norm > 0)
    modes = modes / norm.clamp_min(1e-30)[..., None]
    signal = observed[..., 1] + observed[..., 3]
    idler = observed[..., 2] + observed[..., 3]
    coin = observed[..., 3]
    active = active & (signal > 0)
    cs = coin / signal.clamp_min(1)
    signed = cs - idler / trials.clamp_min(1)
    negative_fraction = ((signed < 0) & active).sum(-1) / active.sum(-1).clamp_min(1)

    def public_map(value: torch.Tensor | None) -> torch.Tensor:
        value = (
            torch.ones((b, n), dtype=torch.float64, device=device)
            if value is None
            else value.detach().to(device=device, dtype=torch.float64)
        )
        if value.ndim == 1 and value.shape == (n,):
            value = value[None].expand(b, -1)
        if (
            value.shape != (b, n)
            or not bool(torch.isfinite(value).all())
            or bool((value < 0).any())
            or bool((value.amax(-1) <= 0).any())
        ):
            raise ValueError(
                "public compensation maps must be nonnegative [B,N] with positive support"
            )
        return value / value.amax(-1, keepdim=True)

    retrieval, idler_cal = public_map(retrieval_map), public_map(idler_map)
    maps = (retrieval, retrieval.sqrt(), idler_cal, idler_cal.sqrt())
    states = (
        torch.eye(n, dtype=torch.complex128, device=device)[None, None].expand(b, 8, -1, -1).clone()
        / n
    )
    view_valid = torch.zeros((b, 8), dtype=torch.bool, device=device)
    discarded_negative_mass = torch.zeros((b, 8), dtype=torch.float64, device=device)
    psd_correction = torch.zeros_like(discarded_negative_mass)
    relative_psd_correction = torch.zeros_like(discarded_negative_mass)
    raw_trace = torch.zeros_like(discarded_negative_mass)
    for row in range(b):
        for estimator_index, estimator in enumerate((signed, cs)):
            for map_index, compensation in enumerate(maps):
                view_index = 4 * estimator_index + map_index
                response = (modes[row].abs().square() * compensation[row]).sum(-1)
                selected = active[row] & (response > 1e-12)
                if not bool(selected.any()):
                    continue
                u = modes[row, selected]
                y = estimator[row, selected].clamp_min(0) / response[selected]
                if not bool((y > 0).any()):
                    continue
                confidence = (signal[row, selected] / signal[row, selected].max()).sqrt()
                kernel = (u.conj() @ u.transpose(-1, -2)).abs().square()
                weighted_kernel = confidence[:, None] * kernel * confidence[None, :]
                alpha = torch.linalg.solve(
                    weighted_kernel + ridge * torch.eye(len(u), dtype=torch.float64, device=device),
                    confidence * y,
                )
                raw = torch.einsum("o,oi,oj->ij", alpha * confidence, u, u.conj())
                raw = 0.5 * (raw + raw.mH)
                eigenvalues, eigenvectors = torch.linalg.eigh(raw)
                discarded_negative_mass[row, view_index] = (-eigenvalues.clamp_max(0)).sum()
                psd_correction[row, view_index] = torch.linalg.vector_norm(eigenvalues.clamp_max(0))
                relative_psd_correction[row, view_index] = psd_correction[
                    row, view_index
                ] / torch.linalg.matrix_norm(raw).clamp_min(1e-30)
                raw_trace[row, view_index] = eigenvalues.sum()
                eigenvalues = eigenvalues.clamp_min(0)
                trace = eigenvalues.sum()
                if not bool(torch.isfinite(trace)) or float(trace) <= 1e-15:
                    continue
                states[row, view_index] = (
                    eigenvectors * (eigenvalues / trace)[None, :]
                ) @ eigenvectors.mH
                view_valid[row, view_index] = True
    weights = view_valid.to(torch.float64) / view_valid.sum(-1, keepdim=True).clamp_min(1)
    unique_views = torch.zeros(b, dtype=torch.int64, device=device)
    unique_maps = torch.zeros(b, dtype=torch.int64, device=device)
    for row in range(b):
        representatives = []
        for index in torch.nonzero(view_valid[row], as_tuple=False).flatten().tolist():
            if not any(
                torch.allclose(states[row, index], states[row, previous], atol=1e-6, rtol=1e-5)
                for previous in representatives
            ):
                representatives.append(index)
        unique_views[row] = len(representatives)
        map_representatives = []
        for index, compensation in enumerate(maps):
            if not any(
                torch.allclose(compensation[row], maps[previous][row], atol=1e-12, rtol=1e-10)
                for previous in map_representatives
            ):
                map_representatives.append(index)
        unique_maps[row] = len(map_representatives)
    # Codes: 0 = absent map, declared ideal/unit hypothesis; 1 = explicitly
    # supplied uniform map; 2 = supplied heterogeneous public hypothesis.
    unit_maps = torch.stack(((retrieval == 1).all(-1), (idler_cal == 1).all(-1)), -1)
    assumption_code = torch.where(
        unit_maps,
        torch.ones_like(unit_maps, dtype=torch.int64),
        torch.full_like(unit_maps, 2, dtype=torch.int64),
    )
    if retrieval_map is None:
        assumption_code[:, 0] = 0
    if idler_map is None:
        assumption_code[:, 1] = 0
    return {
        "rho8_views_re": states.real.float(),
        "rho8_views_im": states.imag.float(),
        "rho8_weights": weights.float(),
        "rho8_view_mask": view_valid,
        "rho8_sample_mask": view_valid.any(-1),
        "rho8_visible_only": torch.ones(b, dtype=torch.bool, device=device),
        "rho8_negative_fraction": negative_fraction.float(),
        "rho8_visible_operation_count": active.sum(-1),
        "rho8_discarded_negative_mass": discarded_negative_mass.float(),
        "rho8_psd_correction_frobenius": psd_correction.float(),
        "rho8_psd_relative_correction": relative_psd_correction.float(),
        "rho8_raw_reconstruction_trace": raw_trace.float(),
        "rho8_psd_projected_mask": discarded_negative_mass > 1e-12,
        "rho8_effective_unique_views": unique_views,
        "rho8_unique_compensation_maps": unique_maps,
        "rho8_unit_map_mask": unit_maps,
        "rho8_map_assumption_code": assumption_code,
        "rho8_uniform_view_weights": torch.ones(b, dtype=torch.bool, device=device),
    }


__all__ = [
    "TrainingIdeasConfig",
    "TrainingIdeasModule",
    "TRAINING_IDEAS_PROFILES",
    "NOISE_CLASS_NAMES",
    "RHO8_VIEW_NAMES",
    "differentiable_rho8_loss",
    "compute_training_ideas_losses",
    "build_visible_rho8_views",
]
