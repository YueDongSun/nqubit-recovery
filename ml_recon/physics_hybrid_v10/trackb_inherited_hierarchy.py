"""Frozen Track A models with an ordered, trainable Track B hierarchy.

The classifier and CSM feedback student are the actual Track A implementations.
This module adds a representation residual, rather than a second action head.
Native observations must already have a qualified classifier-input adapter;
this module never treats sticky detector events as Pauli outcome counts.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import nn

from ml_recon.physics_hybrid_v10.classifier_n_full import V10ClassifierNFull
from ml_recon.physics_hybrid_v10.feedback_n_student_csm import V10FeedbackNStudentCSM
from ml_recon.physics_hybrid_v10.training_ideas_data import PUBLIC_PROJECTOR_KEYS

SEMANTIC_BRIDGE = "v24_frozen_classifier_cli_coherence_confidence_v1"
HIERARCHY_SCHEMA = "TrackBInheritedHierarchyV1"
CLASSIFIER_REQUIRED_FIELDS = frozenset(
    {
        "counts",
        "basis_descriptors",
        "shots",
        "setting_uncertainty",
        "setting_mask",
        "mode_features",
        "mode_mask",
        "physical_adjacency",
        "correlation_adjacency",
        "incidence",
        "certificate_features",
        "certificate_mask",
    }
)
CLASSIFIER_OPTIONAL_FIELDS = frozenset(
    {
        "measurement_probability",
        "global_outcome_histogram",
        "full_outcome_mask",
        "subset_qubits",
        "subset_order",
        "subset_axes",
        "subset_marginal_counts",
        "subset_marginal_mask",
        "subset_shots",
        "pair_stat_qubits",
        "pair_stat_axes",
        "pair_stat_expectation",
        "pair_stat_variance",
        "pair_stat_shots",
        "pair_stat_coverage_mask",
    }
)
FEEDBACK_INPUT_FIELDS = frozenset(
    {
        "noisy_measurement_tokens",
        "target_mode_tokens",
        "physical_mode_tokens",
        "gate_features",
        "analytic_action",
        "mode_mask",
        "history_tokens",
        "history_mask",
        "csm_node_context",
        "csm_edge_context",
        "csm_global_context",
    }
)


@dataclass(frozen=True)
class TrackBInheritedHierarchyConfig:
    """Only the additional hierarchy capacity is configured here."""

    width: int = 64
    heads: int = 4
    temporal_blocks: int = 2
    spatial_blocks: int = 1
    max_decisions: int = 10
    residual_scale: float = 0.1
    semantic_bridge: str = SEMANTIC_BRIDGE
    schema: str = HIERARCHY_SCHEMA

    def validate(self) -> None:
        for name in ("width", "heads", "temporal_blocks", "spatial_blocks", "max_decisions"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"hierarchy {name} must be a positive integer")
        if self.width % self.heads or self.max_decisions != 10:
            raise ValueError("hierarchy requires divisible attention width and ten decisions")
        if not math.isfinite(self.residual_scale) or not 0 < self.residual_scale <= 1:
            raise ValueError("hierarchy residual_scale must lie in (0,1]")
        if self.semantic_bridge != SEMANTIC_BRIDGE or self.schema != HIERARCHY_SCHEMA:
            raise ValueError("unknown inherited hierarchy or semantic bridge")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)


def classifier_feedback_tokens(output: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Use the existing V24 frozen-classifier handoff, on the current device.

    These equations match ``V10FeedbackSemanticInferenceAdapter.__call__``:
    population/reference-phase tokens, predicted covariance and reference
    coherence/confidence ESM tokens. The first mode is the declared phase
    reference. These are inferred features, not new certificates or labels.
    """
    re = output["W_coherence_re"][:, :, 0]
    im = output["W_coherence_im"][:, :, 0]
    relative_phase = torch.atan2(im, re) / torch.pi
    covariance = torch.stack(
        (
            output["W_amplitude_variance"],
            output["W_phase_variance"] / (torch.pi**2),
        ),
        dim=-1,
    )
    # The input variances are nonnegative; this is the existing confidence map.
    confidence = torch.exp(-covariance.mean(-1)).clamp(0.0, 1.0)
    return {
        "denoised_measurement_tokens": torch.stack((output["W_populations"], relative_phase), -1),
        "covariance": covariance,
        "esm_mode_tokens": torch.stack((torch.sqrt(re.square() + im.square()), confidence), -1),
    }


class OrderedTrackBHierarchy(nn.Module):
    """Attend to completed transitions in order, then combine mode/CSM context."""

    def __init__(self, config: TrackBInheritedHierarchyConfig, output_width: int) -> None:
        super().__init__()
        config.validate()
        self.config = config
        # Six two-feature mode tensors, nine CSM node and four CSM global features.
        self.current_input = nn.Linear(25, config.width)
        self.history_input = nn.Linear(7, config.width)
        self.round_embedding = nn.Embedding(config.max_decisions, config.width)
        self.temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                config.width,
                config.heads,
                4 * config.width,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            config.temporal_blocks,
            enable_nested_tensor=False,
        )
        self.csm_message = nn.Linear(config.width, config.width, bias=False)
        self.spatial = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                config.width,
                config.heads,
                4 * config.width,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            config.spatial_blocks,
            enable_nested_tensor=False,
        )
        self.output = nn.Linear(config.width, output_width)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def encode(
        self,
        inputs: Mapping[str, torch.Tensor],
        semantics: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return an order-sensitive latent without a future observation path."""
        history = inputs["history_tokens"]
        prefix_mask = inputs["history_mask"]
        batch, rounds, modes, _ = history.shape
        current = torch.cat(
            (
                inputs["noisy_measurement_tokens"],
                inputs["target_mode_tokens"],
                inputs["physical_mode_tokens"],
                semantics["denoised_measurement_tokens"],
                semantics["esm_mode_tokens"],
                semantics["covariance"],
                inputs["csm_node_context"],
                inputs["csm_global_context"][:, None, :].expand(-1, modes, -1),
            ),
            dim=-1,
        )
        # Mask before projection as well as attention; padded records carry no signal.
        safe_history = torch.where(prefix_mask[:, :, None, None], history, 0.0)
        past = self.history_input(safe_history)
        if rounds:
            positions = torch.arange(rounds, device=history.device)
            past = past + self.round_embedding(positions)[None, :, None, :]
        query = self.current_input(current)
        query = query + self.round_embedding(prefix_mask.sum(1))[:, None, :]
        sequence = torch.cat((past, query[:, None, :, :]), dim=1)
        sequence = sequence.permute(0, 2, 1, 3).reshape(batch * modes, rounds + 1, -1)
        valid = torch.cat(
            (
                prefix_mask,
                torch.ones(
                    (batch, 1),
                    dtype=torch.bool,
                    device=history.device,
                ),
            ),
            dim=1,
        )
        valid = valid[:, None, :].expand(-1, modes, -1).reshape(batch * modes, rounds + 1)
        causal = torch.triu(
            torch.ones(
                (rounds + 1, rounds + 1),
                dtype=torch.bool,
                device=history.device,
            ),
            diagonal=1,
        )
        temporal = self.temporal(sequence, mask=causal, src_key_padding_mask=~valid)
        mode_context = temporal[:, -1].reshape(batch, modes, -1)
        coupling = inputs["csm_edge_context"]
        degree = coupling.sum(-1, keepdim=True)
        # A zero-degree row receives a zero message; one protects that division.
        message = coupling @ mode_context / torch.where(degree > 0, degree, 1.0)
        return self.spatial(mode_context + self.csm_message(message))

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
        semantics: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        latent = self.encode(inputs, semantics)
        return self.config.residual_scale * torch.tanh(self.output(latent))


class TrackBInheritedHierarchy(nn.Module):
    """Compose actual frozen C_N/CSM-F_N modules with one trainable hierarchy.

    Call ``load_core_state_dicts`` with complete recorded checkpoints before
    inference. Merely constructing randomly initialized modules does not count
    as inherited checkpoint evidence. Synthetic tests may explicitly reload
    their synthetic core states, without claiming trained scientific results.
    """

    def __init__(
        self,
        classifier: V10ClassifierNFull,
        feedback: V10FeedbackNStudentCSM,
        config: TrackBInheritedHierarchyConfig | None = None,
    ) -> None:
        super().__init__()
        if (
            type(classifier) is not V10ClassifierNFull
            or type(feedback) is not V10FeedbackNStudentCSM
        ):
            raise TypeError("inheritance requires the actual Track A classifier and CSM student")
        if feedback.mode_input.in_features != 6 or feedback.measurement_input.in_features != 6:
            raise ValueError("canonical V24 feedback bridge requires six-feature core input")
        router = feedback.relational_router.control_block
        if (
            router.node_projection is None
            or router.node_projection.in_features != 9
            or router.global_projection is None
            or router.global_projection.in_features != 4
        ):
            raise ValueError(
                "inherited feedback must use the canonical nine/four-feature CSM inputs"
            )
        self.config = config or TrackBInheritedHierarchyConfig()
        self.config.validate()
        self.classifier = classifier
        self.feedback = feedback
        self.hierarchy = OrderedTrackBHierarchy(self.config, feedback.mode_input.out_features)
        self.core_states_loaded = False
        self._freeze_core()

    def _freeze_core(self) -> None:
        for core in (self.classifier, self.feedback):
            core.requires_grad_(False)
            core.eval()

    def train(self, mode: bool = True) -> "TrackBInheritedHierarchy":
        super().train(mode)
        self._freeze_core()
        return self

    def load_core_state_dicts(
        self,
        classifier_state: Mapping[str, torch.Tensor],
        feedback_state: Mapping[str, torch.Tensor],
    ) -> dict[str, object]:
        """Preflight and strictly load every core parameter; partial reuse raises."""
        for name, module, incoming in (
            ("classifier", self.classifier, classifier_state),
            ("feedback", self.feedback, feedback_state),
        ):
            expected = module.state_dict()
            if set(expected) != set(incoming):
                raise ValueError(f"{name} checkpoint must contain the complete exact core state")
            for key, reference in expected.items():
                value = incoming[key]
                if (
                    not isinstance(value, torch.Tensor)
                    or value.shape != reference.shape
                    or value.dtype != reference.dtype
                    or not bool(torch.isfinite(value).all())
                ):
                    raise ValueError(f"invalid {name} checkpoint tensor: {key}")
        self.classifier.load_state_dict(classifier_state, strict=True)
        self.feedback.load_state_dict(feedback_state, strict=True)
        self.core_states_loaded = True
        self._freeze_core()
        return self.parameter_report()

    def parameter_report(self) -> dict[str, object]:
        classifier = sum(p.numel() for p in self.classifier.parameters())
        feedback = sum(p.numel() for p in self.feedback.parameters())
        hierarchy = sum(p.numel() for p in self.hierarchy.parameters())
        total = classifier + feedback + hierarchy
        return {
            "classifier_parameters": classifier,
            "feedback_parameters": feedback,
            "inherited_parameters": classifier + feedback,
            "hierarchy_parameters": hierarchy,
            "total_parameters": total,
            "inherited_parameter_fraction": (classifier + feedback) / total,
            "core_states_loaded": self.core_states_loaded,
            "trainable_components": ["hierarchy"],
            "semantic_bridge": SEMANTIC_BRIDGE,
            "hardware_action_authorized": False,
        }

    def _validate_inputs(
        self,
        classifier_batch: Mapping[str, torch.Tensor],
        inputs: Mapping[str, torch.Tensor],
        N: int,
    ) -> None:
        if not self.core_states_loaded:
            raise RuntimeError("complete recorded classifier/feedback core states must be loaded")
        if type(N) is not int or not 3 <= N <= 256:
            raise ValueError("inherited hierarchy supports exact N in [3,256]")
        if set(inputs) != FEEDBACK_INPUT_FIELDS:
            raise ValueError(
                "feedback fields must match exactly; inferred ESM and labels stay separate"
            )
        required = CLASSIFIER_REQUIRED_FIELDS
        allowed = required | CLASSIFIER_OPTIONAL_FIELDS
        if self.classifier.readout_domain == "single_excitation_projector":
            allowed = set(PUBLIC_PROJECTOR_KEYS)
            if self.classifier.observation_law == "effective_single_event_v1":
                allowed.discard("readout_parameters")
                allowed.add("effective_calibration")
            else:
                allowed.discard("effective_calibration")
            required = allowed
        if not required <= set(classifier_batch) or set(classifier_batch) - allowed:
            raise ValueError("classifier inputs contain missing, unknown or privileged fields")
        for core in (self.classifier, self.feedback):
            if any(module.training for module in core.modules()) or any(
                parameter.requires_grad for parameter in core.parameters()
            ):
                raise RuntimeError("the inherited core must stay frozen and in evaluation mode")
        for name, value in {**classifier_batch, **inputs}.items():
            if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
                raise ValueError(f"input {name} must be a finite tensor")
        mask = inputs["mode_mask"]
        if (
            mask.dtype != torch.bool
            or mask.ndim != 2
            or mask.shape[1] != N
            or not bool(mask.all())
            or not torch.equal(mask, classifier_batch["mode_mask"])
        ):
            raise ValueError("both cores require one identical unpadded exact-N mode panel")
        batch = mask.shape[0]
        for name in (
            "noisy_measurement_tokens",
            "target_mode_tokens",
            "physical_mode_tokens",
            "analytic_action",
        ):
            if inputs[name].shape != (batch, N, 2):
                raise ValueError(f"{name} must have shape [batch,N,2]")
        expected_csm = {
            "csm_node_context": (batch, N, 9),
            "csm_edge_context": (batch, N, N),
            "csm_global_context": (batch, 4),
        }
        for name, shape in expected_csm.items():
            if inputs[name].shape != shape:
                raise ValueError(f"{name} has an incompatible same-forward CSM shape")
        coupling = inputs["csm_edge_context"]
        if bool(((coupling < 0) | (coupling > 1)).any()) or not torch.allclose(
            coupling,
            coupling.transpose(-1, -2),
            rtol=0,
            atol=1e-6,
        ):
            raise ValueError("CSM coupling must be symmetric and in [0,1]")
        history, prefix = inputs["history_tokens"], inputs["history_mask"]
        if (
            history.ndim != 4
            or history.shape[0] != batch
            or history.shape[2:] != (N, 7)
            or prefix.dtype != torch.bool
            or prefix.shape != history.shape[:2]
        ):
            raise ValueError("history must have shape [batch,completed_rounds,N,7]")
        if history.shape[1] >= self.config.max_decisions or bool(
            (prefix[:, 1:] & ~prefix[:, :-1]).any()
        ):
            raise ValueError("decision ten may read only the first nine completed transitions")
        if bool((history[..., 6][prefix[:, :, None].expand(-1, -1, N)] != 0).any()):
            raise ValueError(
                "unobserved history gain must be zero; latent rewards are training labels"
            )
        if inputs["gate_features"].shape != (batch, self.feedback.moe.gate[0].in_features):
            raise ValueError("feedback gate features differ from the inherited core width")

    def base_feedback_inputs(
        self,
        classifier_batch: Mapping[str, torch.Tensor],
        feedback_inputs: Mapping[str, torch.Tensor],
        *,
        N: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Construct the unchanged frozen-core baseline and return its live semantics."""
        self._validate_inputs(classifier_batch, feedback_inputs, N)
        with torch.no_grad():
            classifier_output = self.classifier(
                dict(classifier_batch), N=N, fine_identifiable=False
            )
            semantics = classifier_feedback_tokens(classifier_output)
            if self.classifier.readout_domain == "single_excitation_projector":
                # This generic envelope carries no out-of-sample uncertainty calibration.
                # RecoveryController supplies its own public, same-law Fisher dispersion.
                semantics["covariance"] = torch.zeros_like(semantics["covariance"])
                semantics["esm_mode_tokens"] = semantics["esm_mode_tokens"].clone()
                semantics["esm_mode_tokens"][..., 1] = 0
        core_inputs = dict(feedback_inputs)
        core_inputs.update(semantics)
        core_inputs["structure_confidence"] = semantics["esm_mode_tokens"][..., 1].mean(-1)
        return core_inputs, classifier_output, semantics

    def forward(
        self,
        classifier_batch: Mapping[str, torch.Tensor],
        feedback_inputs: Mapping[str, torch.Tensor],
        *,
        N: int,
    ) -> dict[str, object]:
        core_inputs, classifier_output, semantics = self.base_feedback_inputs(
            classifier_batch,
            feedback_inputs,
            N=N,
        )
        residual = self.hierarchy(feedback_inputs, semantics)
        # Frozen weights still propagate derivatives to the trainable residual.
        output = dict(self.feedback(core_inputs, hierarchy_residual=residual))
        output.update(
            {
                "classifier_output": classifier_output,
                "semantic_tokens": semantics,
                "hierarchy_residual": residual,
                "hardware_action_authorized": torch.zeros(
                    feedback_inputs["mode_mask"].shape[0],
                    dtype=torch.bool,
                    device=residual.device,
                ),
            }
        )
        return output


__all__ = [
    "SEMANTIC_BRIDGE",
    "HIERARCHY_SCHEMA",
    "CLASSIFIER_REQUIRED_FIELDS",
    "CLASSIFIER_OPTIONAL_FIELDS",
    "FEEDBACK_INPUT_FIELDS",
    "TrackBInheritedHierarchyConfig",
    "OrderedTrackBHierarchy",
    "TrackBInheritedHierarchy",
    "classifier_feedback_tokens",
]
