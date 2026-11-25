"""Torch models for deep physics-informed classification and feedback."""

from __future__ import annotations

from typing import Dict

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover - imported only when torch exists.
    torch = None
    nn = None
    F = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

from .configs import SystemPhysicsConfig
from .synthetic_physics import MAX_RHO_SIZE, RHO_CHANNELS
from .taxonomy import (
    COARSE_ERROR_LABELS,
    ENTANGLEMENT_TIER_LABELS,
    EXCITATION_MODE_LABELS,
    FINE_ERROR_LABELS,
    PHYSICS_AUX_TARGET_NAMES,
    STATE_FAMILY_LABELS,
)


def require_torch() -> None:
    if torch is None:
        raise RuntimeError(f"PyTorch is required for deep models: {_TORCH_IMPORT_ERROR}")


if torch is not None:

    class SqueezeExcite(nn.Module):
        def __init__(self, channels: int, ratio: int = 8) -> None:
            super().__init__()
            hidden = max(8, channels // ratio)
            self.net = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, hidden, 1),
                nn.SiLU(),
                nn.Conv2d(hidden, channels, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * self.net(x)


    class ResidualBlock(nn.Module):
        def __init__(self, channels: int, dropout: float = 0.0) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.SiLU(),
                nn.Dropout2d(dropout),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                SqueezeExcite(channels),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.silu(x + self.net(x))


    class SpatialAttention(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.attn = nn.Sequential(
                nn.Conv2d(channels + 2, channels // 2, 1),
                nn.SiLU(),
                nn.Conv2d(channels // 2, 1, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            avg = torch.mean(x, dim=1, keepdim=True)
            mx = torch.max(x, dim=1, keepdim=True).values
            return x * self.attn(torch.cat([x, avg, mx], dim=1))


    class ImageEncoder(nn.Module):
        def __init__(self, in_channels: int, base_channels: int, hidden_dim: int, dropout: float) -> None:
            super().__init__()
            c = int(base_channels)
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, c, 1),
                nn.BatchNorm2d(c),
                nn.SiLU(),
                nn.Conv2d(c, c, 3, padding=1),
                nn.BatchNorm2d(c),
                nn.SiLU(),
            )
            self.blocks = nn.Sequential(
                ResidualBlock(c, dropout),
                ResidualBlock(c, dropout),
                nn.Conv2d(c, c * 2, 3, stride=2, padding=1),
                nn.BatchNorm2d(c * 2),
                nn.SiLU(),
                ResidualBlock(c * 2, dropout),
                SpatialAttention(c * 2),
                nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1),
                nn.BatchNorm2d(c * 4),
                nn.SiLU(),
                ResidualBlock(c * 4, dropout),
            )
            self.proj = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(c * 4, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(self.blocks(self.stem(x)))


    class PhysicsFusion(nn.Module):
        def __init__(self, physics_dim: int, hidden_dim: int, dropout: float) -> None:
            super().__init__()
            self.physics = nn.Sequential(
                nn.Linear(physics_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )
            self.system_embed = nn.Embedding(2, hidden_dim)
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )

        def forward(self, image_latent: torch.Tensor, physics: torch.Tensor, system_id: torch.Tensor) -> torch.Tensor:
            sys_latent = self.system_embed(system_id.long().view(-1))
            phys_latent = self.physics(physics.float())
            return self.fusion(torch.cat([image_latent, phys_latent, sys_latent], dim=-1))


    class DeepPhysicsClassifier(nn.Module):
        def __init__(
            self,
            physics_dim: int = 64,
            hidden_dim: int = 192,
            base_channels: int = 32,
            dropout: float = 0.08,
        ) -> None:
            super().__init__()
            self.encoder = ImageEncoder(RHO_CHANNELS, base_channels, hidden_dim, dropout)
            self.fusion = PhysicsFusion(physics_dim, hidden_dim, dropout)
            self.shared = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            self.family_head = nn.Linear(hidden_dim, len(STATE_FAMILY_LABELS))
            self.entanglement_head = nn.Linear(hidden_dim, len(ENTANGLEMENT_TIER_LABELS))
            self.excitation_head = nn.Linear(hidden_dim, len(EXCITATION_MODE_LABELS))
            self.coarse_head = nn.Linear(hidden_dim, len(COARSE_ERROR_LABELS))
            self.primary_head = nn.Linear(hidden_dim, len(FINE_ERROR_LABELS))
            self.fine_head = nn.Linear(hidden_dim, len(FINE_ERROR_LABELS))
            self.reg_head = nn.Linear(hidden_dim, 3)
            self.physics_head = nn.Linear(hidden_dim, len(PHYSICS_AUX_TARGET_NAMES))
            self.uncertainty_head = nn.Sequential(nn.Linear(hidden_dim, 3), nn.Softplus())

        def forward(self, image: torch.Tensor, physics: torch.Tensor, system_id: torch.Tensor) -> Dict[str, torch.Tensor]:
            latent = self.shared(self.fusion(self.encoder(image.float()), physics.float(), system_id))
            regs = torch.sigmoid(self.reg_head(latent))
            return {
                "family_logits": self.family_head(latent),
                "entanglement_logits": self.entanglement_head(latent),
                "excitation_logits": self.excitation_head(latent),
                "coarse_logits": self.coarse_head(latent),
                "primary_logits": self.primary_head(latent),
                "fine_logits": self.fine_head(latent),
                "severity": regs[:, 0:1],
                "fidelity": regs[:, 1:2],
                "purity": regs[:, 2:3],
                "physics_aux": self.physics_head(latent),
                "uncertainty": self.uncertainty_head(latent),
                "latent": latent,
            }


    class PhaseMapEncoder(nn.Module):
        def __init__(self, base_channels: int, hidden_dim: int, dropout: float) -> None:
            super().__init__()
            c = max(12, base_channels // 2)
            self.net = nn.Sequential(
                nn.Conv2d(1, c, 3, padding=1),
                nn.BatchNorm2d(c),
                nn.SiLU(),
                ResidualBlock(c, dropout),
                nn.Conv2d(c, c * 2, 3, stride=2, padding=1),
                nn.BatchNorm2d(c * 2),
                nn.SiLU(),
                ResidualBlock(c * 2, dropout),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(c * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )

        def forward(self, phase_map: torch.Tensor) -> torch.Tensor:
            return self.net(phase_map.float())


    class PhaseResidualUNet(nn.Module):
        def __init__(self, in_channels: int, base_channels: int, cond_dim: int, phase_clamp: float) -> None:
            super().__init__()
            c = int(base_channels)
            self.phase_clamp = float(phase_clamp)
            self.cond = nn.Linear(cond_dim, c)
            self.down1 = nn.Sequential(nn.Conv2d(in_channels + c, c, 3, padding=1), nn.SiLU(), ResidualBlock(c, 0.0))
            self.down2 = nn.Sequential(nn.Conv2d(c, c * 2, 3, stride=2, padding=1), nn.SiLU(), ResidualBlock(c * 2, 0.0))
            self.mid = ResidualBlock(c * 2, 0.0)
            self.up = nn.Sequential(nn.Upsample(size=(MAX_RHO_SIZE, MAX_RHO_SIZE), mode="bilinear", align_corners=False), nn.Conv2d(c * 2, c, 3, padding=1), nn.SiLU())
            self.out = nn.Conv2d(c * 2, 1, 3, padding=1)

        def forward(self, phase_input: torch.Tensor, image: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
            cond = self.cond(latent).view(latent.shape[0], -1, 1, 1).expand(-1, -1, MAX_RHO_SIZE, MAX_RHO_SIZE)
            x = torch.cat([phase_input.float(), image[:, 4:6].float(), cond], dim=1)
            d1 = self.down1(x)
            d2 = self.mid(self.down2(d1))
            up = self.up(d2)
            raw = self.out(torch.cat([up, d1], dim=1))
            return torch.tanh(raw).squeeze(1) * self.phase_clamp


    class DeepFeedbackPolicy(nn.Module):
        def __init__(
            self,
            physics_dim: int = 64,
            hidden_dim: int = 256,
            base_channels: int = 32,
            dropout: float = 0.08,
            scalar_delta_dim: int = 8,
            phase_clamp: float = 0.20,
        ) -> None:
            super().__init__()
            self.phase_clamp = float(phase_clamp)
            self.scalar_delta_dim = int(scalar_delta_dim)
            self.image_encoder = ImageEncoder(RHO_CHANNELS, base_channels, hidden_dim, dropout)
            self.phase_encoder = PhaseMapEncoder(base_channels, hidden_dim, dropout)
            self.fusion = PhysicsFusion(physics_dim, hidden_dim, dropout)
            self.feedback_fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )
            self.phase_unet = PhaseResidualUNet(3, max(12, base_channels // 2), hidden_dim, phase_clamp)
            self.scalar_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, scalar_delta_dim), nn.Tanh())
            self.entanglement_head = nn.Linear(hidden_dim, len(ENTANGLEMENT_TIER_LABELS))
            self.coarse_head = nn.Linear(hidden_dim, len(COARSE_ERROR_LABELS))
            self.primary_head = nn.Linear(hidden_dim, len(FINE_ERROR_LABELS))
            self.fine_head = nn.Linear(hidden_dim, len(FINE_ERROR_LABELS))
            self.physics_head = nn.Linear(hidden_dim, len(PHYSICS_AUX_TARGET_NAMES))
            self.correctable_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
            self.gain_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
            self.irreversible_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

        def forward(
            self,
            image: torch.Tensor,
            physics: torch.Tensor,
            phase_map_input: torch.Tensor,
            system_id: torch.Tensor,
        ) -> Dict[str, torch.Tensor]:
            image_latent = self.image_encoder(image.float())
            fused = self.fusion(image_latent, physics.float(), system_id)
            phase_latent = self.phase_encoder(phase_map_input.float())
            latent = self.feedback_fusion(torch.cat([fused, phase_latent], dim=-1))
            phase_map = self.phase_unet(phase_map_input.float(), image.float(), latent)
            correctable = self.correctable_head(latent)
            return {
                "phase_map_delta": phase_map,
                "scalar_delta": self.scalar_head(latent),
                "entanglement_logits": self.entanglement_head(latent),
                "coarse_logits": self.coarse_head(latent),
                "primary_logits": self.primary_head(latent),
                "fine_logits": self.fine_head(latent),
                "physics_aux": self.physics_head(latent),
                "correctable": correctable,
                "expected_gain": self.gain_head(latent),
                "irreversible": self.irreversible_head(latent),
                "latent": latent,
            }


def build_classifier(config: SystemPhysicsConfig, hidden_dim: int, base_channels: int, dropout: float):
    require_torch()
    return DeepPhysicsClassifier(
        physics_dim=config.physics_feature_dim,
        hidden_dim=hidden_dim,
        base_channels=base_channels,
        dropout=dropout,
    )


def build_feedback(config: SystemPhysicsConfig, hidden_dim: int, base_channels: int, dropout: float):
    require_torch()
    return DeepFeedbackPolicy(
        physics_dim=config.physics_feature_dim,
        hidden_dim=hidden_dim,
        base_channels=base_channels,
        dropout=dropout,
        scalar_delta_dim=config.scalar_delta_dim,
        phase_clamp=config.phase_map_clamp_rad,
    )


def count_parameters(model) -> int:
    require_torch()
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
