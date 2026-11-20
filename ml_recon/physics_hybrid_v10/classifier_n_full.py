"""C_N_FULL_V10 variable-N measurement/ESM/W/denoising model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from .certificate_encoder import V10CertificateEncoder
from .count_tokenizer import V10CountTokenizer
from .denoising_head import V10DenoisingHead
from .general_structure_head import V10GeneralStructureHead
from .full_outcome_tokenizer import V10FullOutcomeTokenizer
from .graph_encoder import V10SeparatedGraphEncoders
from .hypergraph_encoder import V10HypergraphEncoder
from .measurement_encoder import V10MeasurementEncoder
from .small_n_fine_head import V10SmallNFineHead
from .subset_marginal_encoder import V10SubsetMarginalEncoder
from .w_structure_head import V10WStructureHead


@dataclass(frozen=True)
class ClassifierProfile:
    width: int
    set_blocks: int
    hypergraph_blocks: int
    heads: int


CLASSIFIER_PROFILES = {
    # Smoke-only engineering profile. It is excluded from capacity selection.
    "C-TEST": ClassifierProfile(32, 1, 1, 4),
    "C-S": ClassifierProfile(192, 6, 3, 8),
    "C-M": ClassifierProfile(256, 8, 4, 8),
    "C-L": ClassifierProfile(320, 10, 5, 8),
    "C-XL": ClassifierProfile(512, 12, 6, 16),
}


class V10ClassifierNFull(nn.Module):
    """No fixed operation embedding; all setting identity comes from descriptors."""

    model_id = "C_N_FULL_V10"

    def __init__(self, profile: str = "C-S", descriptor_dim: int = 15,
                 mode_feature_dim: int = 12, certificate_dim: int = 10,
                 *, training_ideas=None,
                 readout_domain: str = "registered_pauli", observation_law: str | None = None,
                 factor_rank: int = 4, conditional_state_form: str = "sector_trace_v1"):
        super().__init__()
        if profile not in CLASSIFIER_PROFILES:
            raise ValueError(f"unknown profile {profile}")
        cfg = CLASSIFIER_PROFILES[profile]
        self.profile_name = profile
        if conditional_state_form not in ("sector_trace_v1", "factor_gram_v2"):
            raise ValueError("Unknown conditional state representation")
        if conditional_state_form == "factor_gram_v2" and observation_law != "effective_single_event_v1":
            raise ValueError("Factor-normalized conditioning is registered for the effective single-event law")
        self.conditional_state_form = conditional_state_form
        self.profile = cfg
        self.tokenizer = V10CountTokenizer(4, descriptor_dim, cfg.width)
        self.full_outcome_tokenizer = V10FullOutcomeTokenizer(cfg.width)
        self.subset_marginal_encoder = V10SubsetMarginalEncoder(cfg.width)
        self.measurement_encoder = V10MeasurementEncoder(
            cfg.width, cfg.set_blocks, cfg.heads)
        self.mode_input = nn.Linear(mode_feature_dim, cfg.width)
        self.graph_encoders = V10SeparatedGraphEncoders(cfg.width, cfg.width, 3)
        self.hypergraph_encoder = V10HypergraphEncoder(cfg.width, cfg.hypergraph_blocks)
        self.certificate_encoder = V10CertificateEncoder(certificate_dim, cfg.width)
        self.cross_attention = nn.MultiheadAttention(cfg.width, cfg.heads, batch_first=True)
        self.semantic_refinement = nn.Sequential(
            nn.Linear(cfg.width, cfg.width * 4), nn.GELU(),
            nn.Linear(cfg.width * 4, cfg.width), nn.LayerNorm(cfg.width))
        self.small_n_head = V10SmallNFineHead(cfg.width)
        self.structure_head = V10GeneralStructureHead(cfg.width)
        self.w_head = V10WStructureHead(cfg.width, factor_rank=factor_rank)
        self.denoising_head = V10DenoisingHead(cfg.width)
        self.ood_head = nn.Linear(cfg.width, 2)
        self.hyperedge_head = nn.Linear(cfg.width, 1)
        self.certificate_status_head = nn.Linear(cfg.width, 9)
        self.operational_noise_head = nn.Linear(cfg.width, 6)
        if readout_domain not in ("registered_pauli", "single_excitation_projector"):
            raise ValueError("unknown readout domain")
        self.readout_domain = readout_domain
        from .training_ideas_data import PROJECTOR_READOUT_SCOPE
        self.observation_law = observation_law or (PROJECTOR_READOUT_SCOPE if readout_domain == "single_excitation_projector" else "registered_pauli")
        if self.observation_law not in ((PROJECTOR_READOUT_SCOPE, "effective_single_event_v1") if readout_domain == "single_excitation_projector" else ("registered_pauli",)):
            raise ValueError("observation law and readout domain disagree")
        from .training_ideas import TrainingIdeasConfig, TrainingIdeasModule
        self.training_ideas_config = (
            TrainingIdeasConfig.from_profile("baseline") if training_ideas is None
            else TrainingIdeasConfig(**training_ideas) if isinstance(training_ideas, dict) else training_ideas
        )
        self.projector_conditioning = None
        if readout_domain == "single_excitation_projector":
            from .projector_conditioning import ProjectorConditioning
            self.projector_conditioning = ProjectorConditioning(cfg.width, effective=self.observation_law == "effective_single_event_v1")
        self.training_ideas = TrainingIdeasModule(cfg.width, self.training_ideas_config)
        self.hyperedge_measurement_attention = None
        if self.training_ideas_config.esm:
            self.hyperedge_measurement_attention = nn.MultiheadAttention(
                cfg.width, cfg.heads, batch_first=True)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def shared_parameters_for_audit(self):
        """Parameters shared across all task heads for gradient audit.

        Returns only the trunk/encoder parameters that are upstream of
        every task-specific head.  Head-exclusive parameters are returned
        by ``audit_parameter_groups()`` for diagnostic norm tracking.
        """
        shared_modules = [
            self.tokenizer, self.full_outcome_tokenizer,
            self.subset_marginal_encoder, self.measurement_encoder,
            self.mode_input, self.graph_encoders, self.hypergraph_encoder,
            self.certificate_encoder, self.cross_attention,
            self.semantic_refinement,
        ]
        params = []
        for module in shared_modules:
            params.extend(module.parameters())
        if self.projector_conditioning is not None:
            params.extend(self.projector_conditioning.parameters())
        return params

    def audit_parameter_groups(self):
        """Return parameter groups for multi-task gradient audit.

        Keys: ``shared``, ``family_head``, ``structure_heads``,
        ``w_head``, ``denoise_head``, ``noise_head``.
        Shared cosine is the primary diagnostic; head norms are secondary.
        """
        return {
            "shared": list(self.shared_parameters_for_audit()),
            "family_head": list(self.small_n_head.parameters()),
            "structure_heads": list(
                list(self.structure_head.parameters())
                + list(self.hyperedge_head.parameters())
                + list(self.certificate_status_head.parameters())),
            "w_head": list(self.w_head.parameters()),
            "denoise_head": list(self.denoising_head.parameters()),
            "noise_head": list(
                list(self.ood_head.parameters())
                + list(self.operational_noise_head.parameters())),
        }

    def forward(self, batch: Dict[str, torch.Tensor], *, N: int,
                fine_identifiable: bool = False) -> Dict[str, torch.Tensor | bool]:
        if not 3 <= N <= 256:
            raise ValueError("C_N_FULL_V10 supports 3 <= N <= 256")
        valid_settings = batch["setting_mask"].bool()
        if valid_settings.ndim != 2 or torch.any(valid_settings.sum(-1) == 0):
            raise ValueError("each sample requires a visible measurement")
        if self.readout_domain == "single_excitation_projector":
            count_input = batch["counts"]
            if torch.any(count_input[~valid_settings] != 0):
                raise ValueError("query counts must be removed before the model forward")
            if not torch.allclose(
                count_input.sum(-1)[valid_settings].to(torch.float64),
                batch["shots"][valid_settings].to(torch.float64),
            ):
                raise ValueError("visible cells do not conserve known opportunities")
        else:
            count_input = batch.get("measurement_probability", batch["counts"])
        count_tokens = self.tokenizer(
            count_input, batch["basis_descriptors"], batch["shots"],
            batch["setting_uncertainty"], valid_settings)
        full_histogram_available = (
            "global_outcome_histogram" in batch and "full_outcome_mask" in batch)
        if self.readout_domain == "registered_pauli" and N in (3, 4) and full_histogram_available:
            full_tokens = self.full_outcome_tokenizer(
                batch["global_outcome_histogram"], batch["full_outcome_mask"],
                batch["shots"], valid_settings, N)
            if self.training_ideas_config.esm:
                count_tokens = torch.nn.functional.layer_norm(
                    count_tokens + full_tokens * torch.tanh(count_tokens),
                    (count_tokens.shape[-1],))
            else:
                count_tokens = count_tokens + full_tokens
        if self.readout_domain == "single_excitation_projector" and fine_identifiable:
            raise ValueError("projector readout does not supply full Pauli family evidence")
        if N in (3, 4) and fine_identifiable and not full_histogram_available:
            raise ValueError("fine-family supervision requires full global outcome histograms")
        measurement = self.measurement_encoder(count_tokens, valid_settings)
        mode_mask = batch["mode_mask"].bool()
        mode = self.mode_input(batch["mode_features"])
        if self.projector_conditioning is not None:
            mode = mode + self.projector_conditioning(
                mode, measurement["measurement_tokens"], batch, valid_settings)
        graphs = self.graph_encoders(mode, batch["physical_adjacency"],
                                     batch["correlation_adjacency"])
        hyper = self.hypergraph_encoder(graphs["physical_tokens"], batch["incidence"])
        cert = self.certificate_encoder(batch["certificate_features"],
                                        batch["certificate_mask"].bool())
        context = torch.cat((measurement["measurement_tokens"], cert), dim=1)
        context_mask = torch.cat((valid_settings, batch["certificate_mask"].bool()), dim=1)
        fused, _ = self.cross_attention(hyper["node_tokens"], context, context,
                                        key_padding_mask=~context_mask)
        fused = fused + graphs["correlation_tokens"]
        if N > 4 and self.readout_domain == "registered_pauli":
            fused = fused + self.subset_marginal_encoder(batch, N)
        fused = fused + self.semantic_refinement(fused)
        fused, idea_outputs = self.training_ideas(fused, mode_mask, batch)
        pooled = (fused * mode_mask.unsqueeze(-1)).sum(1) / mode_mask.sum(1, keepdim=True).clamp_min(1)
        result: Dict[str, torch.Tensor | bool] = {}
        result.update(self.structure_head(fused, mode_mask))
        command = None
        if self.observation_law == "effective_single_event_v1":
            command = torch.complex(batch["prepared_re"], batch["prepared_im"])
        result.update(self.w_head(fused, mode_mask, prepared=command,
                                  conditional_state_form=self.conditional_state_form))
        result.update(self.denoising_head(measurement["measurement_tokens"]))
        ood = self.ood_head(pooled)
        hyperedge_tokens = hyper["hyperedge_tokens"]
        if self.hyperedge_measurement_attention is not None:
            edge_context = torch.cat((context, fused), dim=1)
            edge_mask = torch.cat((context_mask, mode_mask), dim=1)
            attended, _ = self.hyperedge_measurement_attention(
                hyperedge_tokens, edge_context, edge_context,
                key_padding_mask=~edge_mask)
            hyperedge_tokens = hyperedge_tokens + attended
        hyperedge_probability = torch.sigmoid(
            self.hyperedge_head(hyperedge_tokens)).squeeze(-1)
        certificate_status_logits = self.certificate_status_head(pooled)
        operational_noise_logits = self.operational_noise_head(pooled)
        result.update({"OOD_probability": torch.sigmoid(ood[:, 0]),
                       "model_uncertainty": torch.nn.functional.softplus(ood[:, 1]),
                       "hyperedge_entanglement_probs": hyperedge_probability,
                       "certificate_status_logits": certificate_status_logits,
                       "certificate_status_probs": torch.softmax(certificate_status_logits, -1),
                       "operational_noise_logits": operational_noise_logits,
                       "operational_noise_posterior": torch.softmax(
                           operational_noise_logits, dim=-1),
                       "partition_antichain_scores": result["partition_link_probs"],
                       "partition_representation": "pair_coclustering_with_deterministic_projection",
                       "semantic_latent": pooled})
        result.update(idea_outputs)
        result.update(self.training_ideas.decision_heads(fused, result, batch))
        if N in (3, 4) and self.readout_domain == "registered_pauli":
            result.update(self.small_n_head(pooled, N, fine_identifiable))
        return result
