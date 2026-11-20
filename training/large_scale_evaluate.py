"""Score frozen capacity and step contrasts on the same synthetic states and paid counts."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ml_recon.physics_hybrid_v10.effective_inference_metrics import rank_floor
from ml_recon.physics_hybrid_v10.effective_projector_law import effective_probabilities
from ml_recon.physics_hybrid_v10.projector_budget_response import measurement_panels
from ml_recon.physics_hybrid_v10.training_ideas_data import public_projector_inputs
from ml_recon.physics_hybrid_v10.training_ideas_objective import normalized_state


PANEL_NAMESPACE = 202611201831731
ROWS = 384
BUDGETS = (1024, 4096, 16384)
SIZES = (3, 4, 8)


@lru_cache(maxsize=3)
def panel(n):
    truth, family, panels = measurement_panels(n, ROWS, PANEL_NAMESPACE)
    first = panels[BUDGETS[0]][0]
    amp = torch.complex(first["prepared_re"], first["prepared_im"])
    command = amp.unsqueeze(2) * amp.conj().unsqueeze(1)
    analyzer = torch.complex(first["analyzer_re"], first["analyzer_im"])
    pi = effective_probabilities(command, analyzer, first["effective_calibration"]).double().numpy()
    pi /= pi.sum(axis=-1, keepdims=True)
    counter, identities = {}, {}
    for budget, (obs, query) in panels.items():
        rng = np.random.default_rng([PANEL_NAMESPACE, n, budget, 73])
        counts = np.zeros(pi.shape, dtype=np.int64)
        for col in range(pi.shape[1]):
            shots = obs["shots"][:, col].cpu().numpy()
            vis = obs["setting_mask"][:, col].cpu().numpy()
            for paid in np.unique(shots[vis]):
                selected = vis & (shots == paid)
                counts[selected, col] = rng.multinomial(int(paid), pi[selected, col])
        counter[budget] = torch.from_numpy(counts)
        digest = hashlib.sha256()
        for t in (truth, obs["counts"], obs["shots"], obs["setting_mask"], query, counter[budget]):
            digest.update(t.contiguous().numpy().tobytes())
        identities[str(budget)] = digest.hexdigest()
    return truth, family, panels, command, counter, identities


def predict(model, obs, n, device):
    outputs = []
    for i in range(0, len(obs["counts"]), 32):
        values = {k: v[i:i+32].to(device) for k, v in obs.items()}
        outputs.append(normalized_state(model(values, N=n)).cpu())
    return torch.cat(outputs)


@torch.inference_mode()
def score_model_at_step(model, *, step, identity, dev_banks, seen, device, output_dir):
    """Called at prespecified steps only; all model-visible tensors use the public observation schema."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    training = model.training
    model.eval()
    rows, references, state_files, development = [], {}, {}, {}
    try:
        for n in SIZES:
            truth, family, panels, command, counter, ids = panel(n)
            references[str(n)] = ids
            rank = int(identity.get("factor_rank", identity.get("constructor", {}).get("factor_rank",4)))
            if dev_banks is not None:
                dev = dev_banks[n]
                dev_truth = torch.complex(dev["target_R1_re"], dev["target_R1_im"]).cpu()
                dev_est = predict(model, public_projector_inputs(dev), n, device)
                development[str(n)] = dict(rows=len(dev_truth), risk=float((dev_est-dev_truth).abs().square().sum((-2,-1)).mean()),
                                           distinct_train_states=int(seen[n].sum()) if seen is not None else None)
            state = dict(family=family, state_id=np.arange(ROWS), rank_floor=rank_floor(truth, min(n,rank)).numpy(),
                         public_command_risk=(command-truth).abs().square().sum((-2,-1)).numpy())
            for budget in BUDGETS:
                obs, query = panels[budget]
                estimate = predict(model, obs, n, device)
                replaced = predict(model, dict(obs,counts=counter[budget]), n, device)
                delta = (estimate - truth).abs().square()
                risk = delta.sum((-2,-1))
                diag = delta.diagonal(dim1=-2, dim2=-1).sum(-1)
                analyzer = torch.complex(obs["analyzer_re"],obs["analyzer_im"])
                probability = effective_probabilities(estimate,analyzer,obs["effective_calibration"])
                qreads = query.sum((-2,-1))
                qnll = -(query * probability.log()).sum((-2,-1))/qreads
                replace_risk = (replaced-truth).abs().square().sum((-2,-1))
                values = dict(risk=risk, diagonal_risk=diag, offdiagonal_risk=risk-diag, query_nats_per_read=qnll,
                              command_count_risk=replace_risk, query_reads=qreads)
                if not all(torch.isfinite(v).all() for v in values.values()):
                    raise FloatingPointError("nonfinite exploratory scale score")
                state.update({f"b{budget}_{k}":v.numpy() for k,v in values.items()})
                rows.append(dict(n=n, budget=budget, states=ROWS, risk=float(risk.mean()),
                                 diagonal=float(diag.mean()), offdiagonal=float((risk-diag).mean()),
                                 query_nats_per_read=float(qnll.mean()),
                                 sampled_minus_command_count_risk=float((risk-replace_risk).mean())))
            name=f"n{n}_states.npz"
            tmp=output_dir/f"n{n}.partial.npz"
            np.savez_compressed(tmp, **state)
            tmp.replace(output_dir/name)
            state_files[str(n)] = name
    finally:
        model.train(training)
    report = dict(schema="fixed_parameter_step_scale_scores_v1",step=int(step),identity=identity,
                  namespace=PANEL_NAMESPACE,rows_per_n=ROWS,observations=references,rows=rows,state_files=state_files,
                  development=development)
    tmp=output_dir/"scale_scores.partial.json"
    tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    tmp.replace(output_dir/"scale_scores.json")
    return dict(step=int(step), state_files=state_files, metrics_file="scale_scores.json", rows_per_n=ROWS)
