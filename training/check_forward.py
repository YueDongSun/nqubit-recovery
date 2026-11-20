"""Generate a fixture and evaluate an initialized classifier on CPU."""

import torch

from ml_recon.physics_hybrid_v10.long_projector_training import (
    V10ClassifierNFull, build_bank, constructor,
)
from ml_recon.physics_hybrid_v10.training_ideas_data import public_projector_inputs
from ml_recon.physics_hybrid_v10.training_ideas_objective import normalized_state


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    torch.manual_seed(7301)
    model = V10ClassifierNFull(**constructor("joint", "factor_gram_v2", "C-TEST")).eval()
    for n in (3, 4, 8):
        batch = build_bank(n, 2, "fixture", 0, True, split_namespace="public_forward_check")
        rho = normalized_state(model(public_projector_inputs(batch), N=n))
        if rho.shape != (2, n, n) or not torch.isfinite(rho).all():
            raise ValueError(f"Invalid conditional density at N={n}")
        trace = rho.diagonal(dim1=-2, dim2=-1).sum(-1)
        torch.testing.assert_close(trace, torch.ones_like(trace))
        print(f"N={n}  rows={len(rho)}  mean trace={trace.real.mean():.6f}")


if __name__ == "__main__":
    main()
