"""Produce paired capacity, update and optimization comparisons from cloud-scored states."""
import argparse
from functools import lru_cache
import csv
import json
from pathlib import Path

import numpy as np

from training.large_recipes import RECIPES
from training.large_scale_evaluate import PANEL_NAMESPACE, ROWS
from training.large_scale_train import SOURCE
CAPS=("C-S","C-M","C-L","C-XL","C-XXL")
SEEDS=(7301,7302,7303)
BUDGETS=(1024,4096,16384)
DRAW_COUNT=10000


def write_table(root,name,rows,fields):
    path=root/name
    keys=list(dict.fromkeys([*fields,*(k for r in rows for k in r)]))
    temp=path.with_suffix(".tmp")
    with temp.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    temp.replace(path)


def canonical(cap,seed):
    return (f"joint_{cap.replace('-','').lower()}_32768_s{seed}" if cap in CAPS[:3]
            else f"{cap[2:].lower()}_joint_32768_s{seed}")


@lru_cache(maxsize=1000)
def state_data(path):
    with np.load(path) as data:
        return {key:data[key] for key in data.files}


@lru_cache(maxsize=6)
def boot_weights(families,n):
    f=np.frombuffer(families,dtype=np.int64)
    rng=np.random.default_rng([202611201831731,n,7309])
    mat=np.zeros((DRAW_COUNT,len(f)),dtype=np.float64)
    for fam in np.unique(f):
        ids=np.flatnonzero(f==fam)
        mat[:,ids]=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=DRAW_COUNT)
    return mat/len(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Study directory with runs/ and optional references/ scores")
    parser.add_argument("--classifier-root", type=Path, help="Original matched small/medium/large training runs")
    parser.add_argument("--source", default=SOURCE, help="Source identity used across the scored checkpoints")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    classifier_root = args.classifier_root or root.parent / "classifier"
    specs = {}
    for name, (cap, steps, seed, lr, rank, warmup) in RECIPES.items():
        role = ("matched_parameters_shape" if cap in {"C-XL-D", "C-XL-W"} else
                "optimizer_replication" if name in {"xl_lr_low_s7302", "xl_lr_low_s7303", "xxl_lr_low_s7301", "xxl_lr_high_s7301"} else
                "decoder_rank" if rank == 8 else "capacity" if "_joint_" in name else "optimizer")
        specs[name] = dict(capacity=cap, steps=steps, seed=seed, learning_rate=lr, factor_rank=rank,
                           warmup_steps=warmup, role=role)
    shapes = {k:v for k,v in specs.items() if v["role"] == "matched_parameters_shape"}
    replications = {k:v for k,v in specs.items() if v["role"] == "optimizer_replication"}
    cases = specs
    reg = dict(runs={k:v for k,v in specs.items() if k not in shapes and k not in replications},
               reference_runs=[canonical(cap, seed) for cap in CAPS[:3] for seed in SEEDS],
               primary_comparable_steps=[4096, 8192, 32768], sizes=[3,4,8], states_per_n=ROWS,
               extension_evaluation_namespace=PANEL_NAMESPACE, training_source=args.source)
    lookup={}; paired_inputs={}; summaries=[]; statuses=[]; loss=[]; quality=[]; measured_cost={}
    for group in ("runs","references"):
        for path in sorted((root/group).glob("*")):
            if not path.is_dir():continue
            name=path.name
            allowed=cases if group=="runs" else reg["reference_runs"]
            if name not in allowed:raise ValueError(f"unregistered model: {name}")
            if group=="runs":
                progress=path/"progress.json"
                if progress.exists():
                    info=json.loads(progress.read_text())
                    statuses.append(dict(run_id=name,capacity=cases[name]["capacity"],seed=cases[name]["seed"],
                                         **{k:info[k] for k in ("step","status","active_seconds","parameters",
                                                                  "gpu","gpu_peak_reserved_bytes")}))
                log=path/"training_metrics.jsonl"
                if log.exists():
                    for line in log.read_text().splitlines():
                        rec=json.loads(line)
                        for size,item in rec.get("by_size_stage",{}).items():
                            window={k:rec.get(k) for k in ("learning_rate","target_learning_rate","active_seconds",
                                    "elapsed_seconds","updates_per_second","data_wait_seconds","gpu_peak_reserved_bytes",
                                    "parameters","row_visits","training_read_uses")}
                            distinct=rec.get("distinct_states_per_n",{})
                            loss.append(dict(run_id=name,step=rec["step"],size_objective=size,
                                             updates=item["updates"],distinct_training_reads=4096*sum(distinct.values()),
                                             **window,**item["averages"]))
            for marker in sorted((path/"scores").glob("step_*/evaluation_done.json")):
                data=json.loads((marker.parent/"scale_scores.json").read_text())
                stamp=json.loads(marker.read_text())
                ident=data["identity"];step=data["step"];cap=ident["constructor"]["profile"]
                if (ident["lane"]!=name or ident["source"]!=reg["training_source"] or
                        data["namespace"]!=reg["extension_evaluation_namespace"] or
                        data["rows_per_n"]!=reg["states_per_n"] or marker.parent.name!=f"step_{step:09d}"):
                    raise ValueError(f"unmatched score: {marker}")
                if name in cases:
                    wanted=cases[name]
                    if (cap!=wanted["capacity"] or ident["seed"]!=wanted["seed"] or
                            ident["learning_rate"]!=wanted["learning_rate"] or ident["factor_rank"]!=wanted["factor_rank"]):
                        raise ValueError(f"wrong capacity or optimizer recipe: {name}")
                seconds=stamp.get("active_seconds")
                if group=="references":
                    match=classifier_root/name/"snapshots"/f"step_{step:09d}"/"metrics.json"
                    if match.is_file():seconds=json.loads(match.read_text()).get("active_training_seconds")
                if seconds is not None:
                    measured_cost[name,step]=float(seconds)
                for n in reg["sizes"]:
                    if group=="runs" and str(n) in data.get("development",{}):
                        ref=classifier_root/"joint_cs_32768_s7301/snapshots"/f"step_{step:09d}"/"metrics.json"
                        dev=data["development"][str(n)]
                        prior=None
                        if ref.is_file() and step>0:
                            matched=json.loads(ref.read_text())["development"][str(n)]
                            if matched["states"]!=dev["rows"] or matched["unique_train_states"]!=dev["distinct_train_states"]:
                                raise ValueError("DEV or visited TRAIN denominator differs across capacity controls")
                            prior=matched["train_prior_risk"]
                        quality.append(dict(run_id=name,n=n,step=step,dev_states=dev["rows"],dev_risk=dev["risk"],
                                            visited_train_states=dev["distinct_train_states"],train_prior_risk=prior,
                                            passes_five_percent=dev["risk"]<=0.95*prior if prior is not None else None,
                                            active_seconds=stamp.get("active_seconds"),
                                            evaluation_seconds=stamp.get("evaluation_seconds")))
                    fp=marker.parent/data["state_files"][str(n)]
                    arr=state_data(fp)
                    if not np.array_equal(arr["state_id"],np.arange(reg["states_per_n"])):
                        raise ValueError("state identifiers do not match")
                    lookup[name,step,n]=fp
                    for b in BUDGETS:
                        token=data["observations"][str(n)][str(b)]
                        if paired_inputs.setdefault((n,b),token)!=token:
                            raise ValueError("weights were evaluated on different counts, masks, or state parents")
                        for fam in (None,*np.unique(arr["family"])):
                            ix=slice(None) if fam is None else arr["family"]==fam
                            summaries.append(dict(run_id=name,parameter_tier=cap,seed=ident["seed"],
                               step=step,n=n,read_budget=b,family="all" if fam is None else int(fam),
                               states=arr["state_id"][ix].size,risk=float(arr[f"b{b}_risk"][ix].mean()),
                               diagonal_risk=float(arr[f"b{b}_diagonal_risk"][ix].mean()),
                               offdiagonal_risk=float(arr[f"b{b}_offdiagonal_risk"][ix].mean()),
                               query_nats_per_read=float(arr[f"b{b}_query_nats_per_read"][ix].mean()),
                               count_replacement_risk=float(arr[f"b{b}_command_count_risk"][ix].mean()),
                               public_command_risk=float(arr["public_command_risk"][ix].mean()),
                               decoder_rank_floor=float(arr["rank_floor"][ix].mean()),state_file=str(fp.relative_to(root))))
    contrasts=[]
    def cmp(label,left,right,step_a,step_b,n,b,simultaneous=False,subset=None):
        if not all((a,step_a,n) in lookup and (z,step_b,n) in lookup for a,z in zip(left,right)):return
        ds=[]; family=None
        for a,z in zip(left,right):
            x,y=state_data(lookup[a,step_a,n]),state_data(lookup[z,step_b,n])
            if not np.array_equal(x["family"],y["family"]):raise ValueError("paired parents differ")
            family=x["family"].astype(np.int64);ds.append(x[f"b{b}_risk"]-y[f"b{b}_risk"])
        d=np.mean(np.asarray(ds,dtype=np.float64),axis=0)
        if subset is not None:
            keep=family==subset; family,d=family[keep],d[keep]
        draws=boot_weights(family.tobytes(),n)@d
        alpha=.05/(6 if simultaneous else 1)
        lo,hi=np.quantile(draws,[alpha/2,1-alpha/2])
        contrasts.append(dict(comparison=label,left="|".join(left),right="|".join(right),step_left=step_a,
                              step_right=step_b,n=n,read_budget=b,family="all" if subset is None else subset,
                              states=len(d),fixed_seeds=len(ds),left_minus_right=float(d.mean()),
                              lower=float(lo),upper=float(hi),interval="simultaneous_6" if simultaneous else "pointwise"))
    for n in reg["sizes"]:
        for b in BUDGETS:
            for small,big in zip(CAPS,CAPS[1:]):
                for step in (4096,8192,16384,32768):
                    cmp(f"{big}_minus_{small}",[canonical(big,s) for s in SEEDS],
                        [canonical(small,s) for s in SEEDS],step,step,n,b,
                        b==4096 and step==8192 and big in CAPS[-2:])
            for cap in CAPS:
                names=[canonical(cap,s) for s in SEEDS]
                for earlier,later in ((4096,8192),(8192,16384),(8192,32768),(16384,32768)):
                    cmp(f"{cap}_more_updates",names,names,later,earlier,n,b)
            for name,spec in cases.items():
                if spec["role"]=="capacity":continue
                base_cap="C-XL" if name in shapes else spec["capacity"]
                reference=canonical(base_cap,spec["seed"])
                for step in (4096,8192):
                    cmp(spec["role"]+"_"+name,[name],[reference],step,step,n,b)
                    if spec["role"]=="decoder_rank" and n==8:
                        cmp("decoder_rank_dephased",[name],[canonical("C-XL",7301)],step,step,n,b,subset=3)
            for cap,later,earlier in (("C-XL",65536,32768),("C-XXL",32768,16384)):
                name=canonical(cap,7301);cmp(cap+"_long_seed7301",[name],[name],later,earlier,n,b)
            for step in (4096,8192):
                cmp("lower_lr_new_optimization_seeds",[f"xl_lr_low_s{s}" for s in SEEDS[1:]],
                    [canonical("C-XL",s) for s in SEEDS[1:]],step,step,n,b)
    write_table(root,"scale_curves.csv",summaries,["run_id","parameter_tier","step","n","read_budget","risk"])
    write_table(root,"paired_contrasts.csv",contrasts,["comparison","step_left","step_right","n","read_budget","left_minus_right"])
    write_table(root,"run_status.csv",statuses,["run_id","capacity","seed","step","status"])
    write_table(root,"optimization_trace.csv",loss,["run_id","step","size_objective","updates"])
    compute=[]
    for cap in CAPS:
        for seed in SEEDS:
            name=canonical(cap,seed)
            for allowance in (300,900,1800,3600,5400):
                eligible=[s for s in reg["primary_comparable_steps"]
                          if (name,s) in measured_cost and measured_cost[name,s]<=allowance]
                if not eligible:continue
                step=max(eligible)
                for n in reg["sizes"]:
                    arr=state_data(lookup[name,step,n])
                    compute.append(dict(capacity=cap,run_id=name,seed=seed,n=n,read_budget=4096,
                          allowed_active_seconds=allowance,observed_active_seconds=measured_cost[name,step],
                          last_common_rung=step,states=len(arr["state_id"]),risk=float(arr["b4096_risk"].mean())))
    write_table(root,"observed_time_frontier.csv",compute,["capacity","run_id","seed","n","allowed_active_seconds",
                "observed_active_seconds","last_common_rung","risk"])
    write_table(root,"development_quality.csv",quality,["run_id","n","step","dev_risk","train_prior_risk"])
    time_to_quality=[]
    for run in cases:
        for n in reg["sizes"]:
            curve=sorted((q for q in quality if q["run_id"]==run and q["n"]==n and q["step"]>0),key=lambda q:q["step"])
            if not curve:continue
            first=next(((a,b) for a,b in zip(curve,curve[1:])
                        if a["passes_five_percent"] and b["passes_five_percent"]),None)
            time_to_quality.append(dict(run_id=run,n=n,last_observed_step=curve[-1]["step"],
                     reached=first is not None,first_passing_step=first[0]["step"] if first else None,
                     confirmed_step=first[1]["step"] if first else None,
                     confirmed_active_seconds=first[1]["active_seconds"] if first else None))
    write_table(root,"time_to_quality.csv",time_to_quality,["run_id","n","last_observed_step","reached","confirmed_step"])
    obj=dict(schema="large_scale_summary_v1",registered=len(reg["runs"]),additional_shape_registered=len(shapes),
             optimizer_followup_registered=len(replications),
             terminal=sum(s["status"]=="training_complete" and s["run_id"] in reg["runs"] for s in statuses),
             additional_shape_terminal=sum(s["status"]=="training_complete" and s["run_id"] in shapes for s in statuses),
             optimizer_followup_terminal=sum(s["status"]=="training_complete" and s["run_id"] in replications for s in statuses),
             measured_runs=len(set(k[0] for k in lookup)),state_score_coordinates=len(lookup),
             complete_contrasts=len(contrasts),primary=[c for c in contrasts if c["interval"]=="simultaneous_6"])
    tmp=root/"summary.partial.json";tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n");tmp.replace(root/"summary.json")
    print(json.dumps({k:v for k,v in obj.items() if k!="primary"}))


if __name__=="__main__":main()
