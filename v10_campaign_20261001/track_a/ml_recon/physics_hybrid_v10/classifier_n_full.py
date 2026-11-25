"""C_N_FULL_V10 variable-N measurement/ESM/W/denoising model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as functional

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
from .published_output_registry import output_training_manifest, output_training_metadata


@dataclass(frozen=True)
class ClassifierProfile:
    width: int
    set_blocks: int
    hypergraph_blocks: int
    heads: int
    hyperedge_measurement_fusion: bool = False


CLASSIFIER_PROFILES = {
    # Smoke-only engineering profile. It is excluded from capacity selection.
    "C-TEST": ClassifierProfile(32, 1, 1, 4),
    "C-S": ClassifierProfile(192, 6, 3, 8),
    "C-M": ClassifierProfile(256, 8, 4, 8),
    "C-L": ClassifierProfile(320, 10, 5, 8),
    "C-XL": ClassifierProfile(512, 12, 6, 16),
    # Opt-in measurement-fused hyperedge variants (external review 2026-10-02:
    # the stock hyperedge head reads only sample-invariant topology/incidence
    # tokens and cannot respond to the measured state). New trial identities;
    # flag-off profiles are byte-identical to their parents.
    "C-TEST-HE": ClassifierProfile(32, 1, 1, 4, hyperedge_measurement_fusion=True),
    "C-S-HE": ClassifierProfile(192, 6, 3, 8, hyperedge_measurement_fusion=True),
}


class V10ClassifierNFull(nn.Module):
    """No fixed operation embedding; all setting identity comes from descriptors."""

    model_id = "C_N_FULL_V10"

    def __init__(self, profile: str = "C-S", descriptor_dim: int = 15,
                 mode_feature_dim: int = 12, certificate_dim: int = 10):
        super().__init__()
        if profile not in CLASSIFIER_PROFILES:
            raise ValueError(f"unknown profile {profile}")
        cfg = CLASSIFIER_PROFILES[profile]
        self.profile_name = profile
        self.profile = cfg
        self.constructor_dimensions = (descriptor_dim, mode_feature_dim, certificate_dim)
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
        self.w_head = V10WStructureHead(cfg.width)
        self.denoising_head = V10DenoisingHead(cfg.width)
        self.ood_head = nn.Linear(cfg.width, 2)
        self.hyperedge_head = nn.Linear(cfg.width, 1)
        self.certificate_status_head = nn.Linear(cfg.width, 9)
        self.operational_noise_head = nn.Linear(cfg.width, 6)
        self.hyperedge_measurement_fusion = cfg.hyperedge_measurement_fusion
        if self.hyperedge_measurement_fusion:
            # Cross-attends the hyperedge subset tokens to the measurement
            # context so the hyperedge head can respond to the measured
            # state (the stock path reads only sample-invariant topology
            # and incidence tokens).
            self.hyperedge_context_attention = nn.MultiheadAttention(
                cfg.width, cfg.heads, batch_first=True)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def checkpoint_build_spec(self) -> dict:
        """Expose exact constructor inputs without inferring capacity from size."""
        descriptor_dim, mode_feature_dim, certificate_dim = self.constructor_dimensions
        spec = {"architecture_class": type(self).__name__, "profile": self.profile_name,
                "descriptor_dim": descriptor_dim, "mode_feature_dim": mode_feature_dim,
                "certificate_dim": certificate_dim}
        if self.hyperedge_measurement_fusion:
            # Same parameter shapes across wirings, so checkpoints stay
            # load-compatible; this field distinguishes which forward the
            # checkpoint was trained under (1 = raw measurement context).
            spec["hyperedge_fusion_wiring_version"] = 2
        return spec

    def output_training_manifest(self) -> dict:
        return output_training_manifest()

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
        return params

    def audit_parameter_groups(self):
        """Return parameter groups for multi-task gradient audit.

        Keys: ``shared``, ``family_head``, ``structure_heads``,
        ``w_head``, ``denoise_head``, ``noise_head``.
        Shared cosine is the primary diagnostic; head norms are secondary.
        """
        groups = {
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
        if self.hyperedge_measurement_fusion:
            # Feeds only the hyperedge head, so it belongs with the structure
            # heads rather than the shared trunk.
            groups["structure_heads"] = groups["structure_heads"] + list(
                self.hyperedge_context_attention.parameters())
        return groups

    def forward(self, batch: Dict[str, torch.Tensor], *, N: int,
                fine_identifiable: bool = False) -> Dict[str, Any]:
        if not 3 <= N <= 256:
            raise ValueError("C_N_FULL_V10 supports 3 <= N <= 256")
        valid_settings = batch["setting_mask"].bool()
        count_input = batch.get("measurement_probability", batch["counts"])
        count_tokens = self.tokenizer(
            count_input, batch["basis_descriptors"], batch["shots"],
            batch["setting_uncertainty"], valid_settings)
        full_histogram_available = (
            "global_outcome_histogram" in batch and "full_outcome_mask" in batch)
        if N in (3, 4) and full_histogram_available:
            full_outcome_tokens = self.full_outcome_tokenizer(
                batch["global_outcome_histogram"], batch["full_outcome_mask"],
                batch["shots"], valid_settings, N)
            count_tokens = functional.layer_norm(
                count_tokens + full_outcome_tokens * torch.tanh(count_tokens),
                (count_tokens.shape[-1],),
            )
        if N in (3, 4) and fine_identifiable and not full_histogram_available:
            raise ValueError("fine-family supervision requires full global outcome histograms")
        measurement = self.measurement_encoder(count_tokens, valid_settings)
        mode_mask = batch["mode_mask"].bool()
        mode = self.mode_input(batch["mode_features"])
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
        if N > 4 or "pair_stat_expectation" in batch:
            fused = fused + self.subset_marginal_encoder(batch, N)
        fused = fused + self.semantic_refinement(fused)
        pooled = (fused * mode_mask.unsqueeze(-1)).sum(1) / mode_mask.sum(1, keepdim=True).clamp_min(1)
        result: Dict[str, torch.Tensor | bool] = {}
        result.update(self.structure_head(fused, mode_mask))
        result.update(self.w_head(fused, mode_mask))
        result.update(self.denoising_head(measurement["measurement_tokens"]))
        ood = self.ood_head(pooled)
        if self.hyperedge_measurement_fusion:
            # Wiring v2 (external review round 2): the all-pair observations
            # (pair statistics, correlation adjacency) reach only the fused
            # trunk, so the trunk tokens join the attention context;
            # certificate columns masked in every batch row are dead keys and
            # are pruned; the quadratic attention-weight tensor is never
            # needed and its materialization is disabled.
            alive = context_mask.any(dim=0)
            if bool(alive.all()):
                measured_context, measured_mask = context, context_mask
            else:
                measured_context = context[:, alive]
                measured_mask = context_mask[:, alive]
            hyperedge_context = torch.cat((measured_context, fused), dim=1)
            hyperedge_context_mask = torch.cat((measured_mask, mode_mask), dim=1)
            hyperedge_tokens, _ = self.hyperedge_context_attention(
                hyper["hyperedge_tokens"], hyperedge_context, hyperedge_context,
                key_padding_mask=~hyperedge_context_mask, need_weights=False)
            hyperedge_tokens = hyper["hyperedge_tokens"] + hyperedge_tokens
        else:
            hyperedge_tokens = hyper["hyperedge_tokens"]
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
        if N in (3, 4):
            result.update(self.small_n_head(pooled, N, fine_identifiable))
        result.update(output_training_metadata(result))
        return result
