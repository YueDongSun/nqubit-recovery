"""Paired new-budget and missing-support evaluation of three archived amplitude estimators."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch


from scripts.high_n_amplitude import amp_data, amp_fit
from scripts.high_n_amplitude.amp_control import centered_action, conditional_fidelity
from scripts.high_n_amplitude.long_model import AmplitudeSetModel
from scripts.scaling_20261119.recovery_geometry import reachable_bounds, score_uncertainty


SEEDS = (61, 62, 63)
SIZES = (32, 64, 128, 256)
GROUPS = {"nominal": 101, "near_zero": 102, "zero": 103}
BUDGETS = ("low", "high", "fixed")
COMMIT = "12baba4c97d52acd4449a1a225407c5dbecfd0b9"
TOL = 1e-12
FIXED_READS = 134217701


def write_json(path, record):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    temp.replace(path)


def load_models(formal):
    protocol = json.loads((formal / "evaluation/protocol.json").read_text())
    models, records = {}, {}
    for seed in SEEDS:
        folder = formal / "recovered" / f"first_order_s{seed}"
        path = folder / "selected.pt"
        raw = path.read_bytes()
        git_blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + bytes([0]) + raw).hexdigest()
        selected = torch.load(path, map_location="cpu", weights_only=True)
        result = json.loads((folder / "result.json").read_text())
        recipe = selected["recipe"]
        best = min(result["validation_history"], key=lambda row: row["objective"])
        if (recipe != result["recipe"] or result["state"] != "completed" or recipe["arm"] != "first_order"
                or recipe["seed"] != seed or recipe["bank_id"] != protocol["training_bank_id"]
                or selected["step"] != best["step"] or best["step"] != result["selected_step"]):
            raise ValueError(f"Checkpoint does not match the declared validation selection: {seed}")
        model = AmplitudeSetModel("first_order", width=recipe["width"]).double().eval()
        model.load_state_dict(selected["weights"], strict=True)
        if not all(torch.isfinite(t).all() for t in model.parameters()):
            raise ValueError(f"Nonfinite archived weights: {seed}")
        models[seed] = model
        records[str(seed)] = dict(step=best["step"], sha256=hashlib.sha256(raw).hexdigest(), git_blob=git_blob)
    return models, records


def allocate(n, size, budget, source):
    sim,device,design=amp_data._design(str(source.resolve()),n)
    if budget=="fixed":
        trials, remainder=divmod(FIXED_READS, size)
        if remainder > size-n:
            raise ValueError("Remainder cannot be allocated to distinct unused trine settings")
        return trials, remainder
    dose,rounds=(30,1) if budget=="low" else (300,16)
    return rounds*sim.trials_per_setting(device,design,dose), 0


def generate_states(n, group, rows, source, latent_seed):
    code=GROUPS[group]
    with patch.dict(amp_data.ROLE_IDS,{"new_evaluation":code}), patch.dict(amp_data.FAMILY_NAMESPACE,{"structured":latent_seed}):
        data=amp_data.generate(n,rows,"new_evaluation",nqrec_root=source,shots=30,rounds=1,family="structured")
    truth=data.pop("population")
    if group!="nominal":
        weakest=np.random.default_rng([latent_seed,code,n,97]).integers(n,size=rows)
        ids=np.arange(rows)
        target=(1e-6/n) if group=="near_zero" else 0.
        truth[ids,weakest]=0.
        truth *= ((1.-target)/truth.sum(1))[:,None]
        truth[ids,weakest]=target
    return truth,data


def active_rank(scores, count):
    chosen=np.argsort(-scores,axis=-1,kind="stable")[...,:count]
    active=np.zeros(scores.shape,dtype=bool)
    np.put_along_axis(active,chosen,True,axis=-1)
    return active


def score_policy(gain, score, mode, ix):
    """Use the complete independent-state resample across fixed weights; rerank inside each draw."""
    total=len(gain)
    if mode=="noop":
        active=np.zeros(total,dtype=bool)
    else:
        active=active_rank(score,total//2) if mode=="half" else np.ones(total,dtype=bool)
    sampled=gain[ix]
    if mode=="half":
        selected=active_rank(score[ix],total//2)
    else:
        selected=np.zeros_like(ix,dtype=bool) if mode=="noop" else np.ones_like(ix,dtype=bool)
    actual=gain*active
    boot=np.stack([(sampled*selected).mean(1), ((sampled< -TOL)&selected).mean(1)],-1)
    return actual, active, boot


def summarize(gains, actives, boot):
    g=np.asarray(gains); mask=np.asarray(actives); active=int(mask.sum()); harm=int((g< -TOL).sum())
    ci=np.quantile(boot,[.025,.975],axis=0)
    return dict(mean_gain=float(g.mean()),gain_ci95=ci[:,0].tolist(),harm_rate=harm/g.size,
                harm_ci95=ci[:,1].tolist(),harmed=harm,executed=active,total_decisions=g.size,
                independent_states=g.shape[-1],coverage=active/g.size,
                gain_given_execution=float(g.sum()/active) if active else None,
                harm_given_execution=harm/active if active else None,median_gain=float(np.median(g)))


def cell(truth, base, *, n,group,budget,models,source,count_seed,draws):
    rows=len(truth); code=GROUPS[group]; size=4*n-3; j=BUDGETS.index(budget)
    trials,rem=allocate(n,size,budget,source)
    meta={**base["metadata"], "trials":trials, "population_read_opportunities_per_row":n*trials,
          "full_chain_read_opportunities_per_row":size*trials+rem}
    probs=amp_data.population_probabilities(truth,base["efficiency"],meta)
    if not np.isfinite(probs).all() or (probs<0).any() or not np.allclose(probs.sum(-1),1.,rtol=0.,atol=2e-14):
        raise ValueError("Invalid archived population observation probabilities")
    counts=np.random.default_rng([count_seed,code,n,j]).multinomial(trials,probs)
    if not (counts.sum(-1)==trials).all():
        raise ValueError("Population exposure mismatch")
    public={k:base[k] for k in ("efficiency","log_weight","cell_index")}
    public.update(metadata=meta,counts=counts)
    zero=np.zeros_like(truth)
    before=conditional_fidelity(truth,zero)
    lower,upper=reachable_bounds(truth)
    support_cap=(truth>0).sum(-1)/n
    if np.any(upper>support_cap+1e-10) or np.any(lower<before-1e-10):
        raise ArithmeticError("Invalid attainable overlap bracket")
    minimal=truth.min(1); absent=(truth==0).sum(1); prior_support=int((absent>0).sum())
    samples=dict(before=before,minimum_population=minimal,missing_modes=absent,oracle_feasible=lower,
                 oracle_relaxed_upper=upper,support_ceiling=support_cap,row_id=np.arange(rows))
    ix=np.random.default_rng([count_seed,code,n,803]).integers(rows,size=(draws,rows))
    outputs={};network_inputs=torch.from_numpy(amp_fit.encode_observations(public))
    def gains(estimate):
        action=centered_action(estimate,.25)
        if np.any(np.abs(action).max(1)>.25+1e-12) or np.any(abs(action.sum(1))>1e-8):
            raise ValueError("Action exceeds the physical box or centered gauge")
        g=conditional_fidelity(truth,action)-before
        if np.any(g>upper-before+1e-10):
            raise ArithmeticError("Action exceeded the relaxed upper bound")
        return action,g
    baselines=amp_fit.likelihood_estimates(public)
    for name in ("noop","template_grid_mle","local_grid_shrinkage"):
        gain=np.zeros(rows) if name=="noop" else gains(baselines[name])[1]
        actual,active,boot=score_policy(gain,None,"noop" if name=="noop" else "all",ix)
        outputs[name]=summarize([actual],[active],boot)
        samples[name+"_gain"]=actual;samples[name+"_active"]=active
    grouped={name:[] for name in ("always","scalar_half","fisher_half")}
    for seed,model in models.items():
        with torch.inference_mode():
            estimated=np.concatenate([model(b).exp().numpy() for b in network_inputs.split(32)])
        action,gain=gains(estimated)
        pred=conditional_fidelity(estimated,action)-conditional_fidelity(estimated,zero)
        _,profile,rank=score_uncertainty(estimated,public["efficiency"],meta,action)
        if not np.isfinite(profile).all():
            raise ArithmeticError("Non-estimable Fisher direction in the policy score")
        fisher=pred-1.645*profile
        samples[f"s{seed}_scalar"]=pred;samples[f"s{seed}_fisher"]=fisher;samples[f"s{seed}_nuisance_rank"]=rank
        for name,score,mode in (("always",pred,"all"),("scalar_half",pred,"half"),("fisher_half",fisher,"half")):
            actual,active,boot=score_policy(gain,score,mode,ix)
            outputs[f"s{seed}_{name}"]=summarize([actual],[active],boot)
            samples[f"s{seed}_{name}_gain"]=actual;samples[f"s{seed}_{name}_active"]=active
            grouped[name].append((actual,active,boot))
    for name,parts in grouped.items():
        g,active,boots=zip(*parts)
        outputs["mean3_"+name]=summarize(g,active,np.mean(boots,axis=0))
    info=dict(n=n,support=group,budget=budget,rows=rows,population_trials_per_setting=trials,
              population_reads=n*trials,paid_reads=size*trials+rem,extra_trine_reads=rem,
              p_min_mean=float(minimal.mean()),p_min_min=float(minimal.min()),zero_support_rows=prior_support,
              mean_before=float(before.mean()),mean_feasible_gain=float((lower-before).mean()),
              mean_relaxed_upper_gain=float((upper-before).mean()),mean_support_ceiling=float(support_cap.mean()))
    return dict(config=info,policies=outputs),samples


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output",type=Path,required=True)
    cli.add_argument("--engineering",action="store_true")
    cli.add_argument("--formal", type=Path, required=True, help="Run folder with evaluation/protocol.json and recovered/first_order_s{seed}")
    cli.add_argument("--nqrec-root", type=Path, required=True, help="Source and calibrated Round-6 device configuration")
    args=cli.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1 if args.engineering else 2)
    models,model_records=load_models(args.formal)
    rows=8 if args.engineering else 256;draws=32 if args.engineering else 2000
    latent_seed,count_seed=(2026112081,2026112082) if args.engineering else (2026112011,2026112012)
    registration=dict(archive_base=COMMIT,models=model_records,engineering=args.engineering,
                      latent_seed=latent_seed,count_seed=count_seed,rows=rows,bootstrap=draws,
                      sizes=list(SIZES),groups=GROUPS,budgets=list(BUDGETS),fixed_reads=FIXED_READS,
                      half_coverage=.5,bound=.25,protocol="independent_high_n_amplitude_20261120")
    regfile=args.output/"registration.json"
    if regfile.exists() and json.loads(regfile.read_text())!=registration:
        raise ValueError("Output directory belongs to different weights or a different panel")
    write_json(regfile,registration)
    cells={};start=time.monotonic()
    source=args.nqrec_root.resolve(strict=True)
    for n in ((32,256) if args.engineering else SIZES):
        for group in GROUPS:
            truth,base=generate_states(n,group,rows,source,latent_seed)
            for budget in (("fixed",) if args.engineering else BUDGETS):
                key=f"n{n}_{group}_{budget}";meta=args.output/f"{key}.json";panel=args.output/f"{key}.npz"
                if meta.exists() and panel.exists():
                    cells[key]=json.loads(meta.read_text());continue
                record,samples=cell(truth,base,n=n,group=group,budget=budget,models=models,source=source,count_seed=count_seed,draws=draws)
                pending=panel.with_suffix(".tmp.npz");np.savez_compressed(pending,**samples);pending.replace(panel)
                write_json(meta,record);cells[key]=record
                print(key,"complete",round(time.monotonic()-start,1),flush=True)
    all_records=[]
    for c in cells.values():
        for policy,ss in c["policies"].items():
            record={**c["config"],"policy":policy,**ss}
            for k in ("gain_ci95","harm_ci95"):
                record[k+"_lo"],record[k+"_hi"]=record.pop(k)
            all_records.append(record)
    with (args.output/"policy_cells.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(all_records[0]));writer.writeheader();writer.writerows(all_records)
    write_json(args.output/"result.json",dict(registration=registration,complete_cells=len(cells),cells=cells))


if __name__=="__main__":
    main()
