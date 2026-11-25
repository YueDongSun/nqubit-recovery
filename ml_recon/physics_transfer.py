"""Production wiring for PhysicsTransferGNN feedback head (A+C scheme).

Validated in `ml_recon/sandbox/physics_transfer/` at -37% over baseline.
This module exposes minimal, production-ready glue:

  - `PhysicsTransferGNN`  — the model class (mirrors sandbox.physics_transfer.models.FullPipeline minus B)
  - `train_physics_transfer_model` — two-stage training driver
  - `predict_physics_transfer_delta` — single-sample inference returning a raw delta vector

The classifier knowledge is consumed via `context["classifier_diagnostics"]`,
which is populated by `runner.py:_reconstruct` (F.3 hookup, 2026-06-18).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import GATv2Conv, global_mean_pool

    TORCH_GEOMETRIC_AVAILABLE = True
except Exception:  # noqa: BLE001
    TORCH_GEOMETRIC_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


# ---------- qubit-graph construction (mirrors sandbox.advanced_heads.graph_utils) ----------


_QUBIT_INDICES = {
    3: [4, 2, 1],
    4: [8, 4, 2, 1],
}


def _qubit_indices(num_qubits: int) -> List[int]:
    return list(_QUBIT_INDICES[num_qubits])


def rho_pair_to_graph(
    rho_target: np.ndarray,
    rho_measured: np.ndarray,
    num_qubits: int,
    device: str = "cpu",
):
    """Build one torch_geometric Data object from a (rho_target, rho_measured) pair."""
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise RuntimeError("torch_geometric not available")
    idx = _qubit_indices(num_qubits)

    node_list = []
    for i in idx:
        ri, rt = rho_measured[i, i], rho_target[i, i]
        node_list.append([
            np.real(ri), np.imag(ri), np.abs(ri), np.angle(ri),
            np.real(rt), np.imag(rt), np.abs(rt), np.angle(rt),
            np.real(rt - ri), np.imag(rt - ri),
        ])
    node_feat = torch.tensor(node_list, dtype=torch.float32, device=device)

    edge_src, edge_dst, edge_list = [], [], []
    for i_idx, i in enumerate(idx):
        for j_idx, j in enumerate(idx):
            if i_idx == j_idx:
                continue
            ri, rt = rho_measured[i, j], rho_target[i, j]
            edge_src.append(i_idx)
            edge_dst.append(j_idx)
            edge_list.append([
                np.abs(ri), np.angle(ri),
                np.abs(rt), np.angle(rt),
                np.abs(rt) - np.abs(ri),
                np.angle(rt) - np.angle(ri),
            ])
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long, device=device)
    edge_attr = torch.tensor(edge_list, dtype=torch.float32, device=device)
    return Data(x=node_feat, edge_index=edge_index, edge_attr=edge_attr)


def batch_rho_pairs_to_graph(
    rho_targets: np.ndarray,
    rho_measureds: np.ndarray,
    num_qubits: int,
    device: str = "cpu",
):
    """Batch a list of (rho_target, rho_measured) pairs into a single graph batch."""
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise RuntimeError("torch_geometric not available")
    from torch_geometric.loader import DataLoader  # noqa: PLC0415

    graphs = [
        rho_pair_to_graph(rho_targets[i], rho_measureds[i], num_qubits, device)
        for i in range(len(rho_targets))
    ]
    loader = DataLoader(graphs, batch_size=len(graphs))
    return next(iter(loader)).to(device)


# ---------- model class ----------


if TORCH_GEOMETRIC_AVAILABLE:

    class _QubitGAT(nn.Module):
        """2-layer GATv2 over qubit graph."""

        def __init__(self, in_dim: int = 10, hidden: int = 32, out_dim: int = 32, heads: int = 2, edge_dim: int = 6):
            super().__init__()
            self.conv1 = GATv2Conv(in_dim, hidden, heads=heads, edge_dim=edge_dim, concat=True)
            self.conv2 = GATv2Conv(hidden * heads, out_dim, heads=1, edge_dim=edge_dim, concat=False)

        def forward(self, data: "Batch") -> "torch.Tensor":
            x = F.silu(self.conv1(data.x, data.edge_index, data.edge_attr))
            x = F.silu(self.conv2(x, data.edge_index, data.edge_attr))
            return global_mean_pool(x, data.batch)


    class StructuredPhaseMapDecoder(nn.Module):
        """Decode a full W_CH1 phase map with physics-informed components.

        The map is represented as global + row + column + low-frequency DCT +
        a shrunken per-cell residual.  The final mean removal fixes the otherwise
        unobservable global phase gauge while retaining a global coefficient in
        the learned decomposition for checkpoint diagnostics.
        """

        def __init__(self, latent_dim: int, grid_size: int, dct_rank: int = 4, residual_scale: float = 0.25):
            super().__init__()
            self.grid_size = int(grid_size)
            self.dct_rank = min(int(dct_rank), self.grid_size)
            self.residual_scale = float(residual_scale)
            self.global_head = nn.Linear(latent_dim, 1)
            self.row_head = nn.Linear(latent_dim, self.grid_size)
            self.column_head = nn.Linear(latent_dim, self.grid_size)
            self.dct_head = nn.Linear(latent_dim, self.dct_rank * self.dct_rank)
            self.cell_head = nn.Linear(latent_dim, self.grid_size * self.grid_size)

            coord = np.arange(self.grid_size, dtype=np.float32) + 0.5
            basis = np.stack([
                np.cos(np.pi * k * coord / self.grid_size)
                for k in range(self.dct_rank)
            ]).astype(np.float32)
            basis /= np.maximum(np.linalg.norm(basis, axis=1, keepdims=True), 1e-8)
            self.register_buffer("dct_basis", torch.from_numpy(basis))

        def forward(self, latent: "torch.Tensor") -> "torch.Tensor":
            batch = latent.shape[0]
            global_term = self.global_head(latent)[:, :, None]
            row = self.row_head(latent)[:, :, None]
            column = self.column_head(latent)[:, None, :]
            coeff = self.dct_head(latent).reshape(batch, self.dct_rank, self.dct_rank)
            low_frequency = torch.einsum("bkl,ki,lj->bij", coeff, self.dct_basis, self.dct_basis)
            residual = self.cell_head(latent).reshape(batch, self.grid_size, self.grid_size)
            phase = global_term + row + column + low_frequency + self.residual_scale * residual
            phase = phase - phase.mean(dim=(1, 2), keepdim=True)
            return torch.tanh(phase).reshape(batch, -1)

        def regularization(self, phase_flat: "torch.Tensor") -> Dict[str, "torch.Tensor"]:
            phase = phase_flat.reshape(-1, self.grid_size, self.grid_size)
            tv = (phase[:, 1:, :] - phase[:, :-1, :]).abs().mean()
            tv = tv + (phase[:, :, 1:] - phase[:, :, :-1]).abs().mean()
            return {
                "phase_tv": tv,
                "phase_sparse": phase.abs().mean(),
            }


    class PhysicsTransferGNN(nn.Module):
        """A+C feedback head: GNN encoder + classifier physics_aux features + dual head.

        Stage 1 (synthetic pretrain): both `delta_head` and `physics_head` active;
            loss = SmoothL1(delta) + alpha * MSE(physics_aux_pred, gt).
        Stage 2 (real finetune): only `delta_head` active; encoder frozen by caller.
        """

        def __init__(
            self,
            output_dim: int,
            classifier_feat_dim: int = 26,
            hidden: int = 64,
            dropout: float = 0.1,
            phase_grid_size: Optional[int] = None,
            dct_rank: int = 4,
        ):
            super().__init__()
            self.classifier_feat_dim = int(classifier_feat_dim)
            self.output_dim = int(output_dim)
            self.phase_grid_size = None if phase_grid_size is None else int(phase_grid_size)
            self.gnn = _QubitGAT()
            self.clf_proj = nn.Sequential(
                nn.Linear(classifier_feat_dim, 32),
                nn.SiLU(),
            )
            fused_dim = 32 + 32
            if self.phase_grid_size is None:
                # Preserve the historical module/key layout so pre-v3 checkpoints
                # remain loadable without a migration rewrite.
                self.delta_trunk = nn.Identity()
                self.delta_head = nn.Sequential(
                    nn.Linear(fused_dim, hidden), nn.SiLU(), nn.Dropout(dropout),
                    nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout),
                    nn.Linear(hidden, output_dim),
                )
            else:
                expected = self.phase_grid_size * self.phase_grid_size
                if self.output_dim != expected:
                    raise ValueError(f"output_dim={self.output_dim} must equal phase_grid_size^2={expected}")
                self.delta_trunk = nn.Sequential(
                    nn.Linear(fused_dim, hidden), nn.SiLU(), nn.Dropout(dropout),
                    nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout),
                )
                self.delta_head = StructuredPhaseMapDecoder(hidden, self.phase_grid_size, dct_rank=dct_rank)
            # Auxiliary head used only during stage-1 pretraining; can be discarded after.
            self.physics_head = nn.Linear(32, classifier_feat_dim)

        def forward(
            self,
            graph_data: "Batch",
            classifier_physics_aux: "torch.Tensor",
            stage: str = "finetune",
        ) -> Dict[str, "torch.Tensor"]:
            gnn_latent = self.gnn(graph_data)
            clf_latent = self.clf_proj(classifier_physics_aux)
            fused = torch.cat([gnn_latent, clf_latent], dim=1)
            out = {"delta": self.delta_head(self.delta_trunk(fused))}
            if stage == "pretrain":
                out["physics_aux_pred"] = self.physics_head(gnn_latent)
            return out

        def freeze_encoder(self) -> None:
            for p in self.gnn.parameters():
                p.requires_grad_(False)

        def unfreeze_encoder(self) -> None:
            for p in self.gnn.parameters():
                p.requires_grad_(True)


else:  # torch_geometric absent — provide a stub so imports don't break

    class PhysicsTransferGNN:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("torch_geometric is required for PhysicsTransferGNN")


# ---------- training driver ----------


@dataclass
class PhysicsTransferTrainConfig:
    """Subset of training knobs for production use."""

    pretrain_epochs: int = 15
    finetune_epochs: int = 20
    batch_size: int = 32
    head_lr: float = 1e-3
    encoder_lr: float = 1e-4
    weight_decay: float = 1e-4
    physics_aux_weight: float = 0.2
    classifier_feat_dim: int = 26
    output_dim: int = 8
    phase_grid_size: Optional[int] = None
    dct_rank: int = 4
    phase_tv_weight: float = 1e-3
    phase_sparse_weight: float = 1e-4


def _resolve_device(device: Optional[str]) -> str:
    if device is None:
        return "cuda" if (TORCH_GEOMETRIC_AVAILABLE and torch.cuda.is_available()) else "cpu"
    return device


def _validate_sample_vectors(samples: Sequence[Any], cfg: PhysicsTransferTrainConfig) -> None:
    for index, sample in enumerate(samples):
        delta = np.asarray(sample.scalar_delta).reshape(-1)
        aux = np.asarray(sample.classifier_physics_aux).reshape(-1)
        if delta.size != cfg.output_dim:
            raise ValueError(f"sample {index} delta width {delta.size} != expected {cfg.output_dim}")
        if aux.size != cfg.classifier_feat_dim:
            raise ValueError(f"sample {index} classifier feature width {aux.size} != expected {cfg.classifier_feat_dim}")


def _graph_cache(samples: Sequence[Any], num_qubits: int) -> List[Any]:
    return [
        rho_pair_to_graph(sample.target_rho, sample.measured_rho, num_qubits, device="cpu")
        for sample in samples
    ]


def _cached_batch(graphs: Sequence[Any], indices: np.ndarray, device: str) -> Any:
    return Batch.from_data_list([graphs[int(i)] for i in indices]).to(device)


def _optimizer(model: "PhysicsTransferGNN", cfg: PhysicsTransferTrainConfig, *, finetune: bool = False):
    encoder = [p for p in model.gnn.parameters() if p.requires_grad]
    encoder_ids = {id(p) for p in encoder}
    head = [p for p in model.parameters() if p.requires_grad and id(p) not in encoder_ids]
    groups = []
    if encoder:
        groups.append({"params": encoder, "lr": cfg.encoder_lr})
    if head:
        groups.append({"params": head, "lr": cfg.head_lr * (0.3 if finetune else 1.0)})
    return None if not groups else torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)


def train_physics_transfer_model(
    *,
    pretrain_samples: List[Any],
    finetune_samples: List[Any],
    num_qubits: int,
    config: Optional[PhysicsTransferTrainConfig] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Two-stage training driver.

    Each "sample" must expose:
      .target_rho           (D,D) complex
      .measured_rho         (D,D) complex
      .scalar_delta         (output_dim,)
      .classifier_physics_aux (classifier_feat_dim,)
      .physics_aux_gt       (classifier_feat_dim,)  — optional, used only for stage 1

    Returns a checkpoint dict with state_dict + metadata, suitable for storing in
    `TrainedFeedbackModel.physics_transfer_checkpoint`.
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise RuntimeError("torch_geometric is required for PhysicsTransferGNN training")
    cfg = config or PhysicsTransferTrainConfig()
    dev = _resolve_device(device)

    model = PhysicsTransferGNN(
        output_dim=cfg.output_dim,
        classifier_feat_dim=cfg.classifier_feat_dim,
        phase_grid_size=cfg.phase_grid_size,
        dct_rank=cfg.dct_rank,
    ).to(dev)

    delta_loss_fn = nn.SmoothL1Loss()
    phys_loss_fn = nn.MSELoss()
    rng = np.random.default_rng(0)

    # ---- Stage 1: pretrain on synthetic with auxiliary physics_aux loss ----
    if pretrain_samples and cfg.pretrain_epochs > 0:
        _validate_sample_vectors(pretrain_samples, cfg)
        optim = _optimizer(model, cfg)
        assert optim is not None
        n = len(pretrain_samples)
        print(f"[physics_transfer] building {n} graphs ...", flush=True)
        _t0 = time.time()
        pt_graphs = _graph_cache(pretrain_samples, num_qubits)
        print(f"[physics_transfer] graph cache built in {time.time()-_t0:.1f}s", flush=True)
        pt_clf_aux = torch.from_numpy(np.stack([s.classifier_physics_aux for s in pretrain_samples])).float().to(dev)
        pt_delta = torch.from_numpy(np.stack([s.scalar_delta for s in pretrain_samples])).float().to(dev)
        pt_phys_gt = torch.from_numpy(np.stack([
            getattr(s, "physics_aux_gt", np.zeros(cfg.classifier_feat_dim, dtype=np.float32))
            for s in pretrain_samples
        ])).float().to(dev)
        print(f"[physics_transfer] pretrain samples={n} batches/epoch={math.ceil(n/cfg.batch_size)} epochs={cfg.pretrain_epochs} dev={dev}", flush=True)

        for _epoch in range(cfg.pretrain_epochs):
            model.train()
            perm = rng.permutation(n)
            _ep_loss = 0.0
            _ep_n = 0
            for start in range(0, n, cfg.batch_size):
                idx = perm[start : start + cfg.batch_size]
                gdata = _cached_batch(pt_graphs, idx, dev)
                idx_t = torch.as_tensor(idx, dtype=torch.long, device=dev)
                optim.zero_grad(set_to_none=True)
                out = model(gdata, pt_clf_aux.index_select(0, idx_t), stage="pretrain")
                loss = delta_loss_fn(out["delta"], pt_delta.index_select(0, idx_t))
                if "physics_aux_pred" in out:
                    loss = loss + cfg.physics_aux_weight * phys_loss_fn(
                        out["physics_aux_pred"], pt_phys_gt.index_select(0, idx_t)
                    )
                if isinstance(model.delta_head, StructuredPhaseMapDecoder):
                    regularization = model.delta_head.regularization(out["delta"])
                    loss = loss + cfg.phase_tv_weight * regularization["phase_tv"]
                    loss = loss + cfg.phase_sparse_weight * regularization["phase_sparse"]
                loss.backward()
                optim.step()
                _ep_loss += float(loss.detach().item()) * len(idx)
                _ep_n += len(idx)
            print(f"[physics_transfer] pretrain epoch {_epoch+1}/{cfg.pretrain_epochs} loss={_ep_loss/max(_ep_n,1):.6f}", flush=True)

        model.freeze_encoder()  # stage-2: encoder frozen

    # ---- Stage 2: finetune on real data with delta-only loss ----
    if finetune_samples and cfg.finetune_epochs > 0:
        _validate_sample_vectors(finetune_samples, cfg)
        n = len(finetune_samples)
        ft_graphs = _graph_cache(finetune_samples, num_qubits)
        ft_clf_aux = torch.from_numpy(np.stack([s.classifier_physics_aux for s in finetune_samples])).float().to(dev)
        ft_delta = torch.from_numpy(np.stack([s.scalar_delta for s in finetune_samples])).float().to(dev)

        optim = _optimizer(model, cfg, finetune=bool(cfg.pretrain_epochs > 0))

        for _epoch in range(cfg.finetune_epochs):
            if optim is None:
                break
            model.train()
            perm = rng.permutation(n)
            for start in range(0, n, cfg.batch_size):
                idx = perm[start : start + cfg.batch_size]
                gdata = _cached_batch(ft_graphs, idx, dev)
                idx_t = torch.as_tensor(idx, dtype=torch.long, device=dev)
                optim.zero_grad(set_to_none=True)
                out = model(gdata, ft_clf_aux.index_select(0, idx_t), stage="finetune")
                loss = delta_loss_fn(out["delta"], ft_delta.index_select(0, idx_t))
                loss.backward()
                optim.step()

    # Build checkpoint payload
    state = {k: v.detach().cpu().tolist() for k, v in model.state_dict().items()}
    return {
        "training_mode": "physics_transfer_gnn_a_plus_c",
        "state_dict": state,
        "config": {
            "pretrain_epochs": cfg.pretrain_epochs,
            "finetune_epochs": cfg.finetune_epochs,
            "batch_size": cfg.batch_size,
            "head_lr": cfg.head_lr,
            "encoder_lr": cfg.encoder_lr,
            "weight_decay": cfg.weight_decay,
            "physics_aux_weight": cfg.physics_aux_weight,
            "classifier_feat_dim": cfg.classifier_feat_dim,
            "output_dim": cfg.output_dim,
            "phase_grid_size": cfg.phase_grid_size,
            "dct_rank": cfg.dct_rank,
            "phase_tv_weight": cfg.phase_tv_weight,
            "phase_sparse_weight": cfg.phase_sparse_weight,
        },
        "num_qubits": int(num_qubits),
        "n_pretrain": len(pretrain_samples) if pretrain_samples else 0,
        "n_finetune": len(finetune_samples) if finetune_samples else 0,
        "output_mode": "wch1_phase_map_v1" if cfg.phase_grid_size is not None else "legacy_vector_v1",
        "calibration_status": "unvalidated",
        "confidence": None,
    }


def restore_physics_transfer_model(checkpoint: Dict[str, Any], device: Optional[str] = None) -> "PhysicsTransferGNN":
    """Rebuild the model from a checkpoint payload (the kind returned by train_physics_transfer_model)."""
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise RuntimeError("torch_geometric is required for PhysicsTransferGNN")
    cfg = checkpoint.get("config", {})
    model = PhysicsTransferGNN(
        output_dim=int(cfg.get("output_dim", 8)),
        classifier_feat_dim=int(cfg.get("classifier_feat_dim", 26)),
        phase_grid_size=cfg.get("phase_grid_size"),
        dct_rank=int(cfg.get("dct_rank", 4)),
    )
    state = {k: torch.tensor(v) for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state)
    dev = _resolve_device(device)
    model.to(dev)
    return model


def predict_physics_transfer_delta(
    model: "PhysicsTransferGNN",
    target_rho: np.ndarray,
    measured_rho: np.ndarray,
    classifier_physics_aux: np.ndarray,
    num_qubits: int,
    device: Optional[str] = None,
    train_mode_for_dropout: bool = False,
) -> np.ndarray:
    """Run a single forward pass. Returns the raw delta vector as numpy.

    Set `train_mode_for_dropout=True` to keep dropout active (for MC Dropout
    ensemble inference at the runner layer).
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise RuntimeError("torch_geometric is required for PhysicsTransferGNN inference")
    dev = _resolve_device(device)
    if train_mode_for_dropout:
        model.train()
    else:
        model.eval()
    gdata = batch_rho_pairs_to_graph(target_rho[None, ...], measured_rho[None, ...], num_qubits, dev)
    clf_aux = torch.from_numpy(np.asarray(classifier_physics_aux, dtype=np.float32)[None, :]).to(dev)
    with torch.no_grad():
        out = model(gdata, clf_aux, stage="finetune")
    return out["delta"][0].cpu().numpy()
