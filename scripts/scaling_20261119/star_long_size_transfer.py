"""Fixed-total READ transfer of terminal corrected STAR models to two higher sizes."""
from __future__ import annotations

import argparse
import json
from math import lcm
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"nqrec/src"))
import star_long as train

SIZES=(64,128,256)
CAPS=(2**27,2**30)
COMMON=lcm(*(2*(n-1) for n in SIZES))
BUDGETS=tuple(b//COMMON*COMMON for b in CAPS)
PARENT_SEED,FIXTURE_SEED=852641,852613
ROWS,DRAWS=1024,4000


def latent_parents(rows,seed):
    rng=np.random.default_rng(seed)
    amp=rng.uniform(.2,1.,(rows,max(SIZES)))
    phase=rng.uniform(-np.pi,np.pi,amp.shape);phase[:,0]=0.
    return amp*np.exp(1j*phase),rng.choice([.7,1.],rows)


def normalize(parent,n):
    mode=parent[:,:n]
    return mode/np.linalg.norm(mode,axis=1,keepdims=True)


def core(state):
    hidden=state["leaf_encoder.0.weight"].shape[0]
    rank=(state["leaf_decoder.2.bias"].numel()-1)//2
    model=train.Unified_star_density_net(hidden=hidden,rank_columns=rank,device=torch.device("cpu"))
    model.load_state_dict(state,strict=True)
    return model.eval()


def qualification(frozen):
    parent,vis=latent_parents(2,FIXTURE_SEED)
    report={}
    with torch.no_grad():
        for n in SIZES:
            mode=normalize(parent,n);c=torch.from_numpy(mode);v=torch.from_numpy(vis)
            pure=c[:,:,None]*c.conj()[:,None,:]
            rho=v[:,None,None]*pure+(1-v[:,None,None])*torch.diag_embed(c.abs().square())
            _,_,_,analyzers=train.short.impact.paired_states(n,seed=FIXTURE_SEED,rows=1)
            owner=train.short.impact.new_cells(rho,analyzers,.025,.002,"k1").numpy()
            probs=train.corrected_cells(mode,vis)
            diff=float(abs(probs-owner).max())
            np.testing.assert_allclose(probs,owner,rtol=0,atol=1e-13)
            err=0.
            for budget in BUDGETS:
                trials=budget//(2*(n-1))
                if trials*(2*(n-1))!=budget: raise ArithmeticError("Unmatched budget")
                counts=train.multinomial(probs,trials,np.random.default_rng([FIXTURE_SEED,n,budget]))
                if not np.all(counts.sum((1,2))==budget): raise ArithmeticError("Charge mismatch")
                x=torch.from_numpy(train.features(counts[...,[0,2,1,3]],trials))
                raw=(frozen(x)-rho).abs().square().sum((-2,-1))
                compact=train.state_risk(frozen,x,c,v)
                torch.testing.assert_close(raw,compact,rtol=1e-10,atol=1e-10)
                err=max(err,float((raw-compact).abs().max()))
            report[n]=dict(absolute_probability_error=diff,absolute_risk_error=err)
    return report


def bounds(x,w,family):
    b=w@x
    return dict(mean=float(x.mean()),pointwise_ci95=np.quantile(b,[.025,.975]).tolist(),
                family_ci95=np.quantile(b,[.025/family,1-.025/family]).tolist())


def evaluate(models, parent_seed=PARENT_SEED, seeds=train.SEEDS, rows=ROWS, draws=DRAWS, engineering=False):
    namespace = [parent_seed, 511073] if engineering else [parent_seed]
    parent,vis=latent_parents(rows,namespace);v=torch.from_numpy(vis)
    rng=np.random.default_rng([*namespace,877])
    w=np.zeros((draws,rows))
    for family in np.unique(vis):
        ix=np.flatnonzero(vis==family)
        w[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=draws)
    w/=rows
    result={};raw_arrays={"visibility":vis};base={}
    for n in SIZES:
        c_np=normalize(parent,n);c=torch.from_numpy(c_np);probs=train.corrected_cells(c_np,vis)
        prior=(v.square()+(1-v.square())*c.abs().pow(4).sum(1)-1/n).numpy()
        for budget in BUDGETS:
            trials=budget//(2*(n-1)); name=f"n{n}_r{budget}"
            counts=train.multinomial(probs,trials,np.random.default_rng([*namespace,n,budget,43]))
            if not np.all(counts.sum((1,2))==budget): raise ValueError("Charge mismatch")
            x=torch.from_numpy(train.features(counts[...,[0,2,1,3]],trials))
            risk={"prior":prior}
            with torch.no_grad():
                for key,model in models.items():
                    risk[key]=torch.cat([train.state_risk(model,x[i:i+32],c[i:i+32],v[i:i+32])
                                         for i in range(0,rows,32)]).numpy()
                    if not np.isfinite(risk[key]).all(): raise ValueError("Nonfinite risk")
            for k,a in risk.items(): raw_arrays[name+"_"+k]=a
            for arm in ("corrected","rehearsal"):
                risk[arm]=np.mean([risk[f"{arm}_s{s}"] for s in seeds],axis=0)
                base[n,budget,arm]=risk[arm]-prior
            paired={}
            for a,b in (("corrected","prior"),("corrected","frozen"),("corrected","rehearsal"),
                        ("rehearsal","prior"),("rehearsal","frozen")):
                dx=risk[a]-risk[b]
                paired[a+"_minus_"+b]={**bounds(dx,w,6),"fraction_worse":bounds((dx>0).astype(float),w,6)}
            result[name]=dict(n=n,paid_reads=budget,trials_per_setting=trials,settings=2*(n-1),
                              risks={k:float(a.mean()) for k,a in risk.items()},paired=paired)
    by_size={}
    for arm in ("corrected","rehearsal"):
        by_size[arm]={f"n{n}_r{b}":bounds(base[n,b,arm]-base[64,b,arm],w,4)
                      for n in (128,256) for b in BUDGETS}
    return result,by_size,raw_arrays


def main(argv=None):
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--frozen",type=Path,required=True);cli.add_argument("--terminal",type=Path)
    cli.add_argument("--output",type=Path,required=True);cli.add_argument("--qualification-only",action="store_true")
    cli.add_argument("--parent-seed", type=int, help="Independent transfer namespace for a new fixed-budget campaign")
    cli.add_argument("--reference-record", type=Path, help="Verified new archived-law reference.json")
    cli.add_argument("--engineering", action="store_true", help="Use development exports and a small separate transfer fixture")
    args=cli.parse_args(argv);torch.set_num_threads(1);start=time.perf_counter()
    frozen=core(torch.load(args.frozen,map_location="cpu",weights_only=True))
    checks=qualification(frozen);args.output.mkdir(exist_ok=True,parents=True)
    if args.qualification_only:
        train.atom_json(args.output/"size_transfer_qualification.json",dict(
            fixture_seed=FIXTURE_SEED,zero_updates=True,checks=checks,elapsed_s=time.perf_counter()-start))
        return
    if args.terminal is None: raise ValueError("A completed run directory is required for scientific evaluation")
    role = "development" if args.engineering else "confirmation"
    terminal_name = "engineering_result.json" if args.engineering else "result.json"
    prefix = "engineering_transfer" if args.engineering else "size_transfer"
    row_count = 16 if args.engineering else ROWS
    draw_count = 256 if args.engineering else DRAWS
    summary=json.loads((args.terminal/terminal_name).read_text(encoding="utf8"))
    identity = summary["identity"]
    fresh = identity["schema"] in ("collected_shared_star_v1", "corrected_shared_star_fixed_budget_v1")
    role_seeds = tuple(identity["seeds"])
    if len(set(role_seeds)) != 3:
        raise ValueError("The terminal size-transfer analysis requires three distinct fits")
    parent_seed = PARENT_SEED
    parent_sizes = [8,16,32,64,128,256]
    if fresh:
        if (args.parent_seed is None or args.parent_seed < 0
                or args.parent_seed in (PARENT_SEED, FIXTURE_SEED, identity["split_seed"])):
            raise ValueError("Use a new transfer parent seed independent of archived and adaptation panels")
        if identity["evaluation_role"] != role or identity["parent_sha256"] != train.file_sha(args.frozen):
            raise ValueError("Terminal models do not match the scientific run and frozen reference")
        parent_seed = args.parent_seed
        if args.reference_record is not None:
            reference = json.loads(args.reference_record.read_text())
            if (reference["sha256"] != identity["parent_sha256"] or reference["identity"]["engineering"] != args.engineering
                    or reference["identity"]["schema"] != "fresh_archived_star_reference_v1"):
                raise ValueError("Reference receipt does not match the declared scientific initialization")
            parent_sizes = reference["identity"]["sizes"]
        elif identity["parent_sha256"] != "36453b4d1afba9d500046be7710d5f6b35211902277c894fc9caa80b0163591d":
            raise ValueError("A new initialization requires its own reference receipt")
    elif identity["schema"] != "corrected_shared_star_long_v1" or role_seeds != tuple(train.SEEDS):
        raise ValueError("Unknown archived training geometry")
    elif args.engineering or args.parent_seed is not None or args.reference_record is not None:
        raise ValueError("Archived replication uses its registered parent panel and frozen baseline")
    checkpoint_file = "terminal_weights.pt" if identity["schema"] == "collected_shared_star_v1" else "latest.pt"
    weights=torch.load(args.terminal/checkpoint_file,weights_only=True,map_location="cpu")
    if summary["identity"]!=weights["identity"] or summary["step_per_arm_seed"]!=weights["step"]:
        raise ValueError("Terminal optimizer and terminal run record do not match")
    if tuple(weights["identity"]["sizes"])!=train.SIZES or (fresh and weights["step"] != identity["updates"]):
        raise ValueError("Wrong shared training geometry")
    study_id = dict(training=identity, parent_seed=parent_seed, budgets=BUDGETS, rows=row_count, draws=draw_count)
    registration = args.output / f"{prefix}_registration.json"
    normalized = json.loads(json.dumps(study_id))
    if fresh and registration.exists() and json.loads(registration.read_text()) != normalized:
        raise ValueError("This output directory belongs to a different transfer panel")
    if fresh and (args.output / f"{prefix}.json").exists():
        if not registration.exists():
            raise ValueError("Existing result has no matching new-study registration")
        return
    if fresh:
        train.atom_json(registration, normalized)
    models={"frozen":frozen}
    for arm in ("corrected","rehearsal"):
        for seed in role_seeds:
            k=f"{arm}_s{seed}";models[k]=core(weights["models"][k])
    cells,scale,rows=evaluate(models, parent_seed=parent_seed, seeds=role_seeds, rows=row_count,
                              draws=draw_count, engineering=args.engineering)
    tmp=args.output/f"{prefix}_states.tmp.npz";np.savez_compressed(tmp,**rows);tmp.replace(args.output/f"{prefix}_states.npz")
    train.atom_json(args.output/f"{prefix}.json",dict(evaluation_role=role,schema="terminal_shared_size_transfer_v1",
        parent_seed=parent_seed,rows=row_count,bootstrap_draws=draw_count,actual_new_optimizer_updates=0,
        adaptation_sizes=train.SIZES,original_frozen_training_sizes=parent_sizes,nominal_caps=CAPS,
        exact_matched_budgets=BUDGETS,terminal_updates_per_arm_seed=weights["step"],
        terminal_source=weights["identity"]["source"],numerical_checks=checks,cells=cells,
        prior_adjusted_high_n_minus_n64=scale,elapsed_seconds=time.perf_counter()-start))


if __name__=="__main__":
    main()
