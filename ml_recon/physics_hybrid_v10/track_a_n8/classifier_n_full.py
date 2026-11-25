"""C_N_FULL_V10 variable-N measurement/ESM/W/denoising model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as functional

from ml_recon.physics_hybrid_v10.certificate_encoder import V10CertificateEncoder
from ml_recon.physics_hybrid_v10.count_tokenizer import V10CountTokenizer
from ml_recon.physics_hybrid_v10.denoising_head import V10DenoisingHead
from ml_recon.physics_hybrid_v10.general_structure_head import V10GeneralStructureHead
from ml_recon.physics_hybrid_v10.full_outcome_tokenizer import V10FullOutcomeTokenizer
from ml_recon.physics_hybrid_v10.graph_encoder import V10SeparatedGraphEncoders
from ml_recon.physics_hybrid_v10.hypergraph_encoder import V10HypergraphEncoder
from ml_recon.physics_hybrid_v10.measurement_encoder import V10MeasurementEncoder
from ml_recon.physics_hybrid_v10.small_n_fine_head import V10SmallNFineHead
from ml_recon.physics_hybrid_v10.subset_marginal_encoder import V10SubsetMarginalEncoder
from ml_recon.physics_hybrid_v10.track_a_n8.w_structure_head import V10WStructureHead
from ml_recon.physics_hybrid_v10.published_output_registry import (
    output_training_manifest,
    output_training_metadata,
)
from ml_recon.physics_hybrid_v10.control_coordinate_layout import (
    INTERLEAVED_CONTROL_COORDINATE_SCHEMA,
    INTERLEAVED_CONTROL_LAYOUT_SHA,
)


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
    coordinate_layout_schema = INTERLEAVED_CONTROL_COORDINATE_SCHEMA
    coordinate_layout_sha = INTERLEAVED_CONTROL_LAYOUT_SHA

    def __init__(
        self,
        profile: str = "C-S",
        descriptor_dim: int = 15,
        mode_feature_dim: int = 12,
        certificate_dim: int = 10,
    ):
        super().__init__()
        if profile not in CLASSIFIER_PROFILES:
            raise ValueError(f"unknown profile {profile}")
        cfg = CLASSIFIER_PROFILES[profile]
        self.profile_name = profile
        self.profile = cfg
        self.constructor_dimensions = (
            descriptor_dim,
            mode_feature_dim,
            certificate_dim,
        )
        self.tokenizer = V10CountTokenizer(4, descriptor_dim, cfg.width)
        self.full_outcome_tokenizer = V10FullOutcomeTokenizer(cfg.width)
        self.subset_marginal_encoder = V10SubsetMarginalEncoder(cfg.width)
        self.measurement_encoder = V10MeasurementEncoder(
            cfg.width, cfg.set_blocks, cfg.heads
        )
        self.mode_input = nn.Linear(mode_feature_dim, cfg.width)
        self.graph_encoders = V10SeparatedGraphEncoders(cfg.width, cfg.width, 3)
        self.hypergraph_encoder = V10HypergraphEncoder(cfg.width, cfg.hypergraph_blocks)
        self.certificate_encoder = V10CertificateEncoder(certificate_dim, cfg.width)
        self.cross_attention = nn.MultiheadAttention(
            cfg.width, cfg.heads, batch_first=True
        )
        self.semantic_refinement = nn.Sequential(
            nn.Linear(cfg.width, cfg.width * 4),
            nn.GELU(),
            nn.Linear(cfg.width * 4, cfg.width),
            nn.LayerNorm(cfg.width),
        )
        self.small_n_head = V10SmallNFineHead(cfg.width)
        self.structure_head = V10GeneralStructureHead(cfg.width)
        self.w_head = V10WStructureHead(cfg.width)
        self.denoising_head = V10DenoisingHead(cfg.width)
        self.ood_head = nn.Linear(cfg.width, 2)
        self.hyperedge_head = nn.Linear(cfg.width, 1)
        self.certificate_status_head = nn.Linear(cfg.width, 9)
        self.operational_noise_head = nn.Linear(cfg.width, 6)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def checkpoint_build_spec(self) -> dict:
        """Expose exact constructor inputs without inferring capacity from size."""
        descriptor_dim, mode_feature_dim, certificate_dim = self.constructor_dimensions
        return {
            "architecture_class": type(self).__name__,
            "profile": self.profile_name,
            "descriptor_dim": descriptor_dim,
            "mode_feature_dim": mode_feature_dim,
            "certificate_dim": certificate_dim,
            "coordinate_layout_schema": self.coordinate_layout_schema,
            "coordinate_layout_sha": self.coordinate_layout_sha,
        }

    def validate_coordinate_layout(self) -> None:
        """Bind every registered loss entry to the covariance coordinate ABI."""
        build_spec = self.checkpoint_build_spec()
        if (
            self.model_id != "C_N_FULL_V10"
            or build_spec.get("coordinate_layout_schema")
            != INTERLEAVED_CONTROL_COORDINATE_SCHEMA
            or build_spec.get("coordinate_layout_sha") != INTERLEAVED_CONTROL_LAYOUT_SHA
        ):
            raise ValueError("classifier build identity is not bound to the interleaved ABI")

    def output_training_manifest(self) -> dict:
        return output_training_manifest()

    def shared_parameters_for_audit(self):
        """Parameters shared across all task heads for gradient audit.

        Returns only the trunk/encoder parameters that are upstream of
        every task-specific head.  Head-exclusive parameters are returned
        by ``audit_parameter_groups()`` for diagnostic norm tracking.
        """
        shared_modules = [
            self.tokenizer,
            self.full_outcome_tokenizer,
            self.subset_marginal_encoder,
            self.measurement_encoder,
            self.mode_input,
            self.graph_encoders,
            self.hypergraph_encoder,
            self.certificate_encoder,
            self.cross_attention,
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
        return {
            "shared": list(self.shared_parameters_for_audit()),
            "family_head": list(self.small_n_head.parameters()),
            "structure_heads": list(
                list(self.structure_head.parameters())
                + list(self.hyperedge_head.parameters())
                + list(self.certificate_status_head.parameters())
            ),
            "w_head": list(self.w_head.parameters()),
            "denoise_head": list(self.denoising_head.parameters()),
            "noise_head": list(
                list(self.ood_head.parameters())
                + list(self.operational_noise_head.parameters())
            ),
        }

    def forward(
        self, batch: Dict[str, torch.Tensor], *, N: int, fine_identifiable: bool = False
    ) -> Dict[str, Any]:
        if not 3 <= N <= 256:
            raise ValueError("C_N_FULL_V10 supports 3 <= N <= 256")
        valid_settings = batch["setting_mask"].bool()
        physical_aod_tokens = batch.get("physical_aod_tokens")
        if physical_aod_tokens is None:
            count_input = batch.get("measurement_probability", batch["counts"])
            count_tokens = self.tokenizer(
                count_input,
                batch["basis_descriptors"],
                batch["shots"],
                batch["setting_uncertainty"],
                valid_settings,
            )
        else:
            if not isinstance(
                physical_aod_tokens, torch.Tensor
            ) or physical_aod_tokens.shape != valid_settings.shape + (
                self.profile.width,
            ):
                raise ValueError(
                    "physical_aod_tokens must align with [batch,setting,width]"
                )
            # Opt-in physical branch: full rank-one AOD projectors are encoded
            # upstream; legacy basis descriptors are neither required nor read.
            count_tokens = physical_aod_tokens
        full_histogram_available = physical_aod_tokens is None and (
            "global_outcome_histogram" in batch and "full_outcome_mask" in batch
        )
        if N in (3, 4) and full_histogram_available:
            full_outcome_tokens = self.full_outcome_tokenizer(
                batch["global_outcome_histogram"],
                batch["full_outcome_mask"],
                batch["shots"],
                valid_settings,
                N,
            )
            count_tokens = functional.layer_norm(
                count_tokens + full_outcome_tokens * torch.tanh(count_tokens),
                (count_tokens.shape[-1],),
            )
        if N in (3, 4) and fine_identifiable and not full_histogram_available:
            raise ValueError(
                "fine-family supervision requires full global outcome histograms"
            )
        measurement = self.measurement_encoder(count_tokens, valid_settings)
        mode_mask = batch["mode_mask"].bool()
        mode = self.mode_input(batch["mode_features"])
        graphs = self.graph_encoders(
            mode, batch["physical_adjacency"], batch["correlation_adjacency"]
        )
        hyper = self.hypergraph_encoder(graphs["physical_tokens"], batch["incidence"])
        cert = self.certificate_encoder(
            batch["certificate_features"], batch["certificate_mask"].bool()
        )
        context = torch.cat((measurement["measurement_tokens"], cert), dim=1)
        context_mask = torch.cat(
            (valid_settings, batch["certificate_mask"].bool()), dim=1
        )
        fused, _ = self.cross_attention(
            hyper["node_tokens"], context, context, key_padding_mask=~context_mask
        )
        fused = fused + graphs["correlation_tokens"]
        if N > 4 or "pair_stat_expectation" in batch:
            fused = fused + self.subset_marginal_encoder(batch, N)
        fused = fused + self.semantic_refinement(fused)
        pooled = (fused * mode_mask.unsqueeze(-1)).sum(1) / mode_mask.sum(
            1, keepdim=True
        ).clamp_min(1)
        result: Dict[str, torch.Tensor | bool] = {}
        result.update(self.structure_head(fused, mode_mask))
        result.update(self.w_head(fused, mode_mask, batch.get("sector_probability_override"),
                                 factor_prior=batch.get("physical_factor_prior")))
        result.update(self.denoising_head(measurement["measurement_tokens"]))
        ood = self.ood_head(pooled)
        hyperedge_probability = torch.sigmoid(
            self.hyperedge_head(hyper["hyperedge_tokens"])
        ).squeeze(-1)
        certificate_status_logits = self.certificate_status_head(pooled)
        operational_noise_logits = self.operational_noise_head(pooled)
        result.update(
            {
                "OOD_probability": torch.sigmoid(ood[:, 0]),
                "model_uncertainty": torch.nn.functional.softplus(ood[:, 1]),
                "hyperedge_entanglement_probs": hyperedge_probability,
                "certificate_status_logits": certificate_status_logits,
                "certificate_status_probs": torch.softmax(
                    certificate_status_logits, -1
                ),
                "operational_noise_logits": operational_noise_logits,
                "operational_noise_posterior": torch.softmax(
                    operational_noise_logits, dim=-1
                ),
                "partition_antichain_scores": result["partition_link_probs"],
                "partition_representation": "pair_coclustering_with_deterministic_projection",
                "semantic_latent": pooled,
            }
        )
        if N in (3, 4):
            result.update(self.small_n_head(pooled, N, fine_identifiable))
        result.update(output_training_metadata(result))
        return result
