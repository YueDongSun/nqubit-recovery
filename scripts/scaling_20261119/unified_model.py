"""One shared-weight conditional-density model for the frozen visible STAR layout.

The retained producer's ``phase_graph.pair_measurement_schedule`` orders edges
(0, 1), (0, 2), ... and phases 0 then pi/2 on each leaf. Its
``joint_mc_stream.joint_measurement_schedule`` inserts the query chord (1, 2)
after the first edge. The producer selects ``visible and native_read_trials > 0``
before ``build_features`` flattens the visible cells in C order. Thus each leaf
has eight consecutive log-count features: phase 0 cells (00, 01, 10, 11), then
phase pi/2 cells in the same order. The last feature is log(READ exposure).
This contract uses positive-exposure STAR observations, reference mode 0 and
the source schedule's N=3..256 domain. Callers own observation provenance and
the visible filtering; tensor width alone cannot establish that ordering.

Only observed counts, exposure and size enter the model. Counts retain their
log1p representation and are divided by log1p(exposure), evaluated from the
stored log exposure with logaddexp. The two global coordinates are log(N)/log256
and log(exposure/(8192*N))/log30. These fixed transforms use no clipping.
One parameter set covers all supported N. The result is the conditional N-by-N
state; this module supplies no source-sector, count, training or hardware path.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional


ARCHITECTURE_VERSION = "unified_star_density_v1"
LAYOUT_VERSION = "visible_star_leaf_pairs_v1"
NATIVE_CELL_ORDER = ("00", "01", "10", "11")
REFERENCE_MODE = 0
MINIMUM_N = 3
MAXIMUM_N = 256
RANK_COLUMN_CHOICES = (4, 16)
LEAF_COUNT_FEATURES = 8
GLOBAL_FEATURES = 2
EXPOSURE_PER_MODE_AT_1X = 8192
LOG_SIZE_SCALE = math.log(256)
LOG_EXPOSURE_SCALE = math.log(30)
LAYOUT_SOURCES = (
    "ml_recon/physics_hybrid_v10/fullsetpro/phase_graph.py::pair_measurement_schedule",
    "ml_recon/physics_hybrid_v10/fullsetpro/joint_mc_stream.py::joint_measurement_schedule",
    "ml_recon/physics_hybrid_v10/fullsetpro/contracts.py::native_cell_order",
    "ml_recon/reports/fullsetpro_event_scaling_20261022/n16_law_datascale_20261113/law_common.py::build_features",
)


class Unified_star_density_net(nn.Module):
    """Shared leaf/global inference with a fixed-column Gram plus diagonal state.

    ``forward`` accepts float64 [B, 8*(N-1)+1] and returns complex128 [B,N,N].
    ``rank_columns`` remains 4 or 16 at every N, including overcomplete N<R.
    Architecture metadata is separate from the tensor-only ``state_dict``.
    ``device='meta'`` followed by ``load_state_dict(..., assign=True)`` restores
    the same constructor at every size, without size-dependent parameters.
    """

    def __init__(self, hidden=128, rank_columns=4, device=None):
        super().__init__()
        if type(hidden) is not int or hidden < 1:
            raise ValueError("hidden must be a positive integer")
        if type(rank_columns) is not int or rank_columns not in RANK_COLUMN_CHOICES:
            raise ValueError("rank_columns must be 4 or 16 and stays fixed across N")
        self.hidden, self.rank_columns = hidden, rank_columns
        output_features = 2 * rank_columns + 1
        self.leaf_encoder = nn.Sequential(
            nn.Linear(LEAF_COUNT_FEATURES + GLOBAL_FEATURES, hidden, device=device, dtype=torch.float64),
            nn.SiLU(), nn.Linear(hidden, hidden, device=device, dtype=torch.float64), nn.SiLU())
        self.leaf_decoder = nn.Sequential(
            nn.Linear(2 * hidden, hidden, device=device, dtype=torch.float64), nn.SiLU(),
            nn.Linear(hidden, output_features, device=device, dtype=torch.float64))
        self.reference_decoder = nn.Sequential(
            nn.Linear(hidden + GLOBAL_FEATURES, hidden, device=device, dtype=torch.float64), nn.SiLU(),
            nn.Linear(hidden, output_features, device=device, dtype=torch.float64))

    def architecture_metadata(self) -> dict:
        return {"architecture": ARCHITECTURE_VERSION, "hidden": self.hidden, "rank_columns": self.rank_columns,
            "parameter_count": sum(value.numel() for value in self.parameters()),
            "parameter_dtype": str(next(self.parameters()).dtype).removeprefix("torch."),
            "input_dtype": "float64", "output_dtype": "complex128", "size_specific_parameters": False,
            "gram_rank_bound": "min(rank_columns,N)", "pooling": "mean_over_non_reference_modes",
            "layout": {"schema": LAYOUT_VERSION, "reference_mode": REFERENCE_MODE,
                "n_domain": [MINIMUM_N, MAXIMUM_N], "leaf_order": "ascending modes 1..N-1",
                "phases_per_leaf": [0.0, math.pi / 2], "native_cell_order": list(NATIVE_CELL_ORDER),
                "visibility_filter": "visible and native_read_trials > 0", "flatten_order": "C",
                "input_width": "8*(N-1)+1", "last_feature": "log(exposure)",
                "sources": list(LAYOUT_SOURCES)},
            "scaling": {"count": "log1p(count)/logaddexp(0,log(exposure))",
                "size": "log(N)/log(256)", "exposure": "(log(exposure)-log(8192*N))/log(30)",
                "clipping": False, "trace_floor": False}}

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if (not isinstance(features, torch.Tensor) or features.dtype != torch.float64
                or features.ndim != 2 or features.shape[0] < 1 or features.shape[1] <= 1
                or (features.shape[1] - 1) % LEAF_COUNT_FEATURES):
            raise ValueError("features must be nonempty float64 [B,8*(N-1)+1]")
        n_mode = (features.shape[1] - 1) // LEAF_COUNT_FEATURES + 1
        if not MINIMUM_N <= n_mode <= MAXIMUM_N:
            raise ValueError("Feature geometry lies outside the frozen STAR N=3..256 domain")
        parameter = next(self.parameters())
        if parameter.dtype != torch.float64 or parameter.device != features.device:
            raise ValueError("Model parameters and features require float64 on the same materialized device")
        if features.device.type == "meta":
            raise ValueError("Restore meta parameters before numeric inference")
        if not bool(torch.isfinite(features).all()) or bool((features[:, :-1] < 0).any()):
            raise ValueError("features require finite log exposure and nonnegative finite log1p counts")
        log_exposure = features[:, -1]
        log1p_exposure = torch.logaddexp(torch.zeros_like(log_exposure), log_exposure)
        if not bool(torch.isfinite(log1p_exposure).all()) or bool((log1p_exposure <= 0).any()):
            raise ValueError("log1p(exposure) must have a finite positive representation")
        metadata = torch.stack((torch.full_like(log_exposure, math.log(n_mode) / LOG_SIZE_SCALE),
            (log_exposure - math.log(EXPOSURE_PER_MODE_AT_1X * n_mode)) / LOG_EXPOSURE_SCALE), dim=-1)
        leaf_counts = features[:, :-1].reshape(features.shape[0], n_mode - 1, LEAF_COUNT_FEATURES)
        leaf_inputs = torch.cat((leaf_counts / log1p_exposure[:, None, None],
                                 metadata[:, None, :].expand(-1, n_mode - 1, -1)), dim=-1)
        if not bool(torch.isfinite(leaf_inputs).all()):
            raise ValueError("Scaled observations must remain finite")
        leaf_hidden = self.leaf_encoder(leaf_inputs)
        pooled = leaf_hidden.mean(dim=1)
        leaf_values = self.leaf_decoder(torch.cat((leaf_hidden,
            pooled[:, None, :].expand(-1, n_mode - 1, -1)), dim=-1))
        reference_values = self.reference_decoder(torch.cat((pooled, metadata), dim=-1))
        node_values = torch.cat((reference_values[:, None, :], leaf_values), dim=1)
        if not bool(torch.isfinite(node_values).all()):
            raise ArithmeticError("Unified density decoder produced nonfinite values")
        factor = torch.complex(node_values[..., :self.rank_columns],
                               node_values[..., self.rank_columns:2 * self.rank_columns])
        diagonal = functional.softplus(node_values[..., -1])
        density = factor @ factor.conj().transpose(-2, -1) + torch.diag_embed(diagonal).to(torch.complex128)
        normalization = density.diagonal(dim1=-2, dim2=-1).real.sum(-1)
        if not bool(torch.isfinite(normalization).all()) or bool((normalization <= 0).any()):
            raise ArithmeticError("Unified density normalization must be finite and positive")
        result = density / normalization[:, None, None]
        if not bool(torch.isfinite(result).all()):
            raise ArithmeticError("Normalized unified density contains nonfinite values")
        return result
