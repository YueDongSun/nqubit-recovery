"""Check a two-state identifiability counterexample under the registered four-cell law."""

from pathlib import Path
import json
import numpy as np
import torch


from ml_recon.physics_hybrid_v10.effective_projector_law import effective_probabilities
from ml_recon.physics_hybrid_v10.training_ideas_data import projector_schedule


@torch.inference_mode()
def analyze():
    calibration = [0.025, 0.40, 0.002, 0.50, 0.002]
    rows = []
    for n in (3, 4, 8, 16):
        eye = torch.eye(n, dtype=torch.complex128)
        w = eye.sum(0) / n**0.5
        center = 0.5 * (eye / n + torch.outer(w, w.conj()))
        direction = (torch.outer(eye[1], eye[2]) + torch.outer(eye[2], eye[1])) / (
            2 * n
        )
        rho = torch.stack([center + direction, center - direction])
        torch.testing.assert_close(
            rho.diagonal(dim1=1, dim2=2).sum(-1), torch.ones(2, dtype=rho.dtype)
        )
        assert torch.linalg.eigvalsh(rho).min() >= -1e-12
        star = torch.from_numpy(projector_schedule(n)[0])
        leaf = (eye[1] + eye[2]) / 2**0.5
        analyzers = torch.cat([star, leaf[None]]).expand(2, -1, -1)
        probs = effective_probabilities(
            rho, analyzers, torch.tensor([calibration] * 2, dtype=torch.float64)
        )
        torch.testing.assert_close(probs.sum(-1), torch.ones_like(probs[..., 0]))
        assert torch.isfinite(probs).all() and (probs > 0).all()
        max_star = float((probs[0, :-1] - probs[1, :-1]).abs().max())
        assert max_star < 1e-12
        fidelity = torch.einsum("i,bij,j->b", w.conj(), rho, w).real
        gap = float(fidelity[0] - fidelity[1])
        assert abs(gap - 2 / n**2) < 1e-12
        a, b = probs[:, -1].numpy()
        # The other 3,072 READs contribute zero KL on these two hypotheses.
        kl = float(1024 * np.sum(a * np.log(a / b)))
        assert kl > 0
        rows.append(
            dict(
                n=n,
                star_settings=len(star),
                f_plus=float(fidelity[0]),
                f_minus=float(fidelity[1]),
                conditional_overlap_gap=gap,
                max_star_four_cell_difference=max_star,
                star_only_equal_prior_optimal_accuracy=0.5,
                star_only_kl_nats=0.0,
                leaf_setting_max_four_cell_difference=float(np.max(np.abs(a - b))),
                total_4096_read_kl_nats_with_1024_leaf_reads=kl,
            )
        )
    return dict(
        source_commit="12baba4c97d52acd4449a1a225407c5dbecfd0b9",
        law="effective_single_event_v1",
        calibration=calibration,
        k2_source_weight=0,
        construction="rho+/- = (|W><W| + I/N)/2 +/- (|1><2|+|2><1|)/(2N); mode 0 is the reference",
        allocation="4096 READs in either design; move 1024 from STAR to (|1>+|2>)/sqrt(2) in alternative",
        evidence="Exact conditional K1 counterexample on general mixed states, outside the training-family guarantee. No training, network or experiment.",
        rows=rows,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    out = parser.parse_args().output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(analyze(), indent=2, allow_nan=False) + "\n")
