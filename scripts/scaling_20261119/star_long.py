"""Matched multi-size STAR adaptation and rehearsal with restartable common minibatches."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

import star_corrected_adaptation as short
import star_data
from nqrec.physics.single_event_mc import calibrate_source_weights, four_cell_probabilities
from star_train import atom_json, atom_torch, features, file_sha, owner, state_risk
from unified_model import Unified_star_density_net

SIZES = (8, 16, 32, 64)
SEEDS = (83, 89, 97)
ARMS = ("corrected", "rehearsal", "fixed")
BASE_SEEDS = {"train": 811031, "development": 811051, "confirmation": 811073}
ROWS = {"train":4096, "development":192, "confirmation":1024}
STEPS_PER_BANK, BATCH = 1024, 64
MILESTONES = (16,128,1024,4096,16384,65536,131072,262144,524288,1048576)
RESERVE_SECONDS = 200
CALIBRATION = (.025,.20,.002,.03,.001)


def rng_for(role, n, epoch, stream, split_seed=None):
    role_seed = 914093 if role == "reference_train" else BASE_SEEDS[role]
    key = [role_seed, n, epoch, stream]
    if split_seed is not None:
        key.extend([1, split_seed])
    return np.random.default_rng(key)


def multinomial(p,trials,rng):
    """Vectorized stick-breaking with exactly the requested multinomial distribution."""
    remaining=np.full(p.shape[:-1],trials,dtype=np.int64)
    counts=np.empty(p.shape,dtype=np.int64)
    for j in range(3):
        prob=p[...,j]/p[...,j:].sum(-1)
        counts[...,j]=rng.binomial(remaining,prob)
        remaining=remaining-counts[...,j]
    counts[...,3]=remaining
    if not np.all(counts.sum(-1)==trials): raise ValueError("Invalid READ sum")
    return counts


def corrected_cells(mode,v):
    """Exact two-mode contractions of the shared K<=1 effective-event owner."""
    t=lambda x:torch.tensor(x,dtype=torch.float64)
    q,es,bs,ei,bi=CALIBRATION
    joint=calibrate_source_weights(t(q),t(es),t(bs),double_probability=t(0.))["joint"]
    ref,leaf=mode[:,:1],mode[:,1:]
    coherent=np.stack((abs(ref+leaf)**2,abs(ref+1j*leaf)**2),-1)/2
    diagonal=(abs(ref)**2+abs(leaf)**2)[:,:,None]/2
    projection=(v[:,None,None]*coherent+(1-v[:,None,None])*diagonal).reshape(len(mode),-1)
    r1=torch.from_numpy(bi+(1-bi)*ei*projection)
    response=torch.stack((torch.full_like(r1,bi),r1,torch.zeros_like(r1)),-1)
    return four_cell_probabilities(joint,response[...,None].expand(-1,-1,-1,2)).numpy()


def bank(role, n, epoch, device, split_seed=None, *, rows=None, laws=("corrected", "archived")):
    rng=rng_for(role, n, epoch, 0, split_seed)
    if rows is None:
        rows=ROWS[role]
    amp=rng.uniform(.2,1.,(rows,n));phase=rng.uniform(-np.pi,np.pi,(rows,n));phase[:,0]=0
    mode=amp*np.exp(1j*phase)/np.linalg.norm(amp,axis=1,keepdims=True)
    vis=rng.choice([.7,1.],size=rows)
    data={"mode":torch.from_numpy(mode).to(device),"vis":torch.from_numpy(vis).to(device)}
    shots=8192*n
    for law in laws:
        i = {"corrected": 1, "archived": 2}[law]
        prob=corrected_cells(mode,vis) if i==1 else star_data.observation_probabilities(mode,vis)
        cnt=multinomial(prob,shots,rng_for(role, n, epoch, i, split_seed))
        if law=="corrected": cnt=cnt[..., [0,2,1,3]]
        data[law]=torch.from_numpy(features(cnt,shots)).to(device)
    return data


def model_key(arm,seed,n):
    return f"{arm}_s{seed}"+(f"_n{n}" if arm=="fixed" else "")


def qualify(model,device):
    for n in SIZES:
        c,v,rho,an=short.impact.paired_states(n,seed=811003,rows=4)
        expected=short.impact.new_cells(rho,an,.025,.002,"k1").numpy()
        np.testing.assert_allclose(corrected_cells(c,v),expected,rtol=0,atol=1e-13)
        counts=multinomial(expected,8192*n,np.random.default_rng(811019))
        x=torch.from_numpy(features(counts[..., [0,2,1,3]],8192*n)).to(device)
        compact=state_risk(model,x,torch.from_numpy(c).to(device),torch.from_numpy(v).to(device))
        dense=(model(x)-rho.to(device)).abs().square().sum((-2,-1))
        torch.testing.assert_close(compact,dense,rtol=1e-10,atol=1e-10)
        compact.mean().backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters()): raise ArithmeticError("nonfinite gradient")
        model.zero_grad(set_to_none=True)


def evaluate(models, frozen, role, device, seeds=SEEDS, split_seed=None):
    summary={}; arrays={}
    for n in SIZES:
        d=bank(role, n, 0, device, split_seed);cv=d["vis"].cpu().numpy();arrays[f"n{n}_visibility"]=cv
        pop=d["mode"].abs().square()
        prior=(d["vis"].square()+(1-d["vis"].square())*pop.square().sum(1)-1/n).cpu().numpy()
        todo={"frozen":frozen}
        for arm in ARMS:
            for s in seeds: todo[model_key(arm,s,n)]=models[model_key(arm,s,n)]
        with torch.no_grad():
            results={}
            for law in ("corrected","archived"):
                arrays[f"n{n}_{law}_prior"]=prior
                raw={"prior":prior}
                for key,model in todo.items():
                    model.eval()
                    score=torch.cat([state_risk(model,d[law][i:i+64],d["mode"][i:i+64],d["vis"][i:i+64])
                                     for i in range(0,len(prior),64)]).cpu().numpy()
                    if not np.isfinite(score).all(): raise ArithmeticError("Nonfinite risk")
                    arrays[f"n{n}_{law}_{key}"]=score
                    raw[key]=score
                for arm in ARMS: raw[arm]=np.mean([raw[model_key(arm,s,n)] for s in seeds],axis=0)
                comparisons={}
                for a,b in (("corrected","frozen"),("corrected","fixed"),("corrected","prior"),
                            ("rehearsal","corrected"),("rehearsal","frozen"),("fixed","frozen")):
                    delta=raw[a]-raw[b]; stat={"mean":float(delta.mean()),"fraction_worse":float((delta>0).mean())}
                    if role=="confirmation":
                        stat.update(short.estimate(delta,cv))
                        stat["fraction_worse_ci95"]=short.estimate((delta>0).astype(float),cv)["interval95"]
                    comparisons[a+"_minus_"+b]=stat
                results[law]={"mean_risk":{k:float(x.mean()) for k,x in raw.items()},"paired":comparisons}
            summary[str(n)]=results
    return summary,arrays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--deadline", help="Absolute cutoff with timezone, required for historical LPY runs")
    parser.add_argument("--updates", type=int, help="Fixed matched units per arm and sampling seed")
    parser.add_argument("--seed", type=int, help="Run a single optimizer seed")
    parser.add_argument("--split-seed", type=int, help="Fresh data namespace shared by all compared arms and seeds")
    parser.add_argument("--device", help="For a fixed-budget run, use cpu, cuda, or cuda:INDEX")
    parser.add_argument("--engineering", action="store_true",
                        help="Use only development panels; suppress scientific confirmation")
    parser.add_argument("--fit-only", action="store_true",
                        help="Export all final models and leave scoring to the three-seed collector")
    a = parser.parse_args(argv)
    fixed_budget = a.updates is not None
    if fixed_budget:
        if a.updates <= 0 or a.updates % len(SIZES):
            parser.error("--updates must be a positive multiple of the number of training sizes")
        if a.split_seed is None or a.split_seed < 0:
            parser.error("--updates requires a nonnegative --split-seed for new data")
        if a.seed is not None and a.seed < 0:
            parser.error("--seed must be nonnegative")
    elif a.deadline is None or a.engineering or a.fit_only or any(option is not None for option in (a.seed, a.split_seed, a.device)):
        parser.error("use --deadline for the historical run or --updates and --split-seed for a new run")
    end = float("inf")
    if a.deadline:
        cutoff = datetime.fromisoformat(a.deadline.replace("Z", "+00:00"))
        if fixed_budget and cutoff.tzinfo is None:
            parser.error("--deadline must include a timezone")
        end = cutoff.timestamp()
    seeds = SEEDS if a.seed is None else (a.seed,)
    fit_only = fixed_budget and (a.fit_only or not a.engineering)
    if fit_only and a.seed is None:
        parser.error("a deferred fit must specify one --seed per run directory")
    target = a.updates if fixed_budget else MILESTONES[-1]
    milestones = tuple(sorted({v for v in MILESTONES if v <= target} | {target}))
    torch.set_num_threads(2)
    device = torch.device(a.device or "cuda")
    if device.type not in ("cpu", "cuda"):
        parser.error("--device must be cpu, cuda, or cuda:INDEX")
    actual = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    if not fixed_budget and "4060 Ti" not in actual:
        raise RuntimeError(f"Unexpected device {actual}")
    a.output.mkdir(parents=True, exist_ok=True)
    with owner(a.output):
        initial = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
        hidden = initial["leaf_encoder.0.weight"].shape[0]
        rank = (initial["leaf_decoder.2.bias"].numel() - 1) // 2

        def make():
            model = Unified_star_density_net(hidden=hidden, rank_columns=rank, device=device)
            model.load_state_dict(initial, strict=True)
            return model

        identity = dict(schema="corrected_shared_star_long_v1", source=a.source, deadline=a.deadline,
                        parent=a.checkpoint.name, sizes=SIZES, seeds=seeds, arms=ARMS, rows=ROWS,
                        base_seeds=BASE_SEEDS, batch=BATCH, calibration=CALIBRATION,
                        rank=rank, hidden=hidden, device=actual, milestones=milestones)
        if fixed_budget:
            # The time limit can be extended when resuming; the scientific endpoint cannot.
            del identity["deadline"]
            identity.update(schema="corrected_shared_star_fixed_budget_v1", updates=target,
                            split_seed=a.split_seed, parent_sha256=file_sha(a.checkpoint),
                            evaluation_role="development" if a.engineering else "confirmation")
            if fit_only:
                identity.pop("device")
                identity["scoring"] = "deferred_three_seed"
        identity = json.loads(json.dumps(identity))
        reg = a.output / "registration.json"
        if reg.exists() and json.loads(reg.read_text()) != identity:
            raise ValueError("Changed source/registration")
        phase = "fit_terminal" if fit_only else "result"
        result_path = a.output / (("engineering_" if a.engineering else "") + phase + ".json")
        if result_path.exists():
            if json.loads(result_path.read_text())["identity"] != identity:
                raise ValueError("Changed result identity")
            return
        atom_json(reg, identity)
        frozen = make()
        qualify(frozen, device)
        models = {model_key(arm, seed, n): None for n in SIZES for arm in ARMS for seed in seeds}
        models = {key: make() for key in models}
        optim = {key: torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=0.)
                 for key, model in models.items()}
        step=0;elapsed=0.;checkpoint=a.output/"latest.pt"
        runtime_segments = []
        loss_buffer = []
        def store_loss_block(records):
            if records:
                dest = a.output / "training_loss"
                dest.mkdir(exist_ok=True)
                atom_json(dest / f"through_{records[-1]['step']:07d}.json", records)
        if checkpoint.exists():
            state=torch.load(checkpoint,map_location="cpu",weights_only=True)
            if state["identity"]!=identity: raise ValueError("Incompatible recovery")
            for k,m in models.items():
                m.load_state_dict(state["models"][k],strict=True);optim[k].load_state_dict(state["optimizers"][k])
            step=state["step"];elapsed=state["elapsed_seconds"]
            runtime_segments = state.get("runtime_segments", [])
            store_loss_block(state.get("last_loss_block", []))
            del state
        if fit_only:
            runtime_segments.append(dict(start_after_update=step, device=actual, cpu_threads=torch.get_num_threads()))
        last_saved=step; start=time.perf_counter(); last_saved_time=start
        def elapsed_now(): return elapsed+time.perf_counter()-start
        def save():
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            atom_torch(checkpoint,dict(identity=identity,step=step,elapsed_seconds=elapsed_now(),
                models={k:m.state_dict() for k,m in models.items()},optimizers={k:o.state_dict() for k,o in optim.items()},
                runtime_segments=runtime_segments,last_loss_block=loss_buffer))
            store_loss_block(loss_buffer)
            loss_buffer.clear()
            atom_json(a.output/"progress.json",dict(step_per_arm_seed=step,total_optimizer_updates=step*len(ARMS)*len(seeds),
                     elapsed_seconds=elapsed_now(),utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
                     device=actual,fit_only=fit_only,
                     memory_reserved_mib=torch.cuda.max_memory_reserved(device)/1048576 if device.type == "cuda" else 0))
        epoch=None; batches={}
        while step < target and (time.time() < end - RESERVE_SECONDS or step % len(SIZES) != 0):
            if fit_only and step % len(SIZES) == 0 and (a.output / "pause.request").exists():
                save()
                atom_json(a.output / "paused.json", dict(step=step, pid=os.getpid(),
                          reason="operator-requested checkpoint for migration", identity=identity))
                return
            if step//STEPS_PER_BANK!=epoch:
                epoch=step//STEPS_PER_BANK;batches={}
            n=SIZES[step % len(SIZES)]
            if n not in batches: batches[n]=bank("train", n, epoch, device, a.split_seed)
            data=batches[n]
            for seed in seeds:
                ids=torch.from_numpy(np.random.default_rng([811039,seed,step,n]).integers(ROWS["train"],size=BATCH)).to(device)
                corrected=data["corrected"][ids]; truth=data["mode"][ids];vis=data["vis"][ids]
                rehearsal=torch.cat((data["archived"][ids[:BATCH//4]],corrected[BATCH//4:]))
                losses = []
                for arm in ARMS:
                    key=model_key(arm,seed,n);model=models[key];opt=optim[key]
                    model.train();opt.zero_grad(set_to_none=True)
                    loss=state_risk(model,rehearsal if arm=="rehearsal" else corrected,truth,vis).mean()
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True);opt.step()
                    if fit_only:
                        losses.append(loss.detach())
                if fit_only:
                    loss_buffer.append(dict(step=step+1,n=n,seed=seed,elapsed_seconds=elapsed_now(),
                                            train_loss=dict(zip(ARMS,torch.stack(losses).cpu().tolist()))))
            step+=1
            if step in (1,4) or step in milestones or step-last_saved>=128 or time.perf_counter()-last_saved_time>=120:
                save();last_saved=step;last_saved_time=time.perf_counter()
            if step in milestones:
                where=a.output/"milestones"/f"u{step:07d}";where.mkdir(parents=True,exist_ok=True)
                atom_torch(where/"state.pt",dict(identity=identity,step=step,models={k:m.state_dict() for k,m in models.items()}))
                report,_=evaluate(models, frozen, "development", device, seeds, a.split_seed)
                dev=dict(step=step,elapsed_seconds=elapsed_now(),systems=report)
                atom_json(where/"development.json",dev);atom_json(a.output/"development.json",dev)
        if fixed_budget and step < target:
            save()
            raise RuntimeError(f"Cutoff reached at {step}/{target} matched updates; confirmation has not run")
        if step < len(SIZES):
            raise RuntimeError("Insufficient time for one full size rotation")
        save()
        if fit_only:
            atom_json(result_path, dict(identity=identity,step_per_arm_seed=step,
                      total_optimizer_updates=step*len(ARMS)*len(seeds),elapsed_seconds=elapsed_now(),
                      runtime_segments=runtime_segments,checkpoint="latest.pt",confirmation_scored=False))
            return
        role = "development" if a.engineering else "confirmation"
        final,arrays=evaluate(models, frozen, role, device, seeds, a.split_seed)
        panel = a.output / ("engineering_scores.npz" if a.engineering else "confirmation.npz")
        temp=panel.with_suffix(".tmp.npz");np.savez_compressed(temp,**arrays);temp.replace(panel)
        usage={str(n):dict(unique_train_states_accessible=((step-1)//STEPS_PER_BANK+1)*ROWS["train"],
                 visits_per_arm_seed=step//len(SIZES)*BATCH,paid_reads_per_arm_seed=step//len(SIZES)*BATCH*8192*n*2*(n-1)) for n in SIZES}
        atom_json(result_path,dict(identity=identity,step_per_arm_seed=step,total_optimizer_updates=step*len(ARMS)*len(seeds),
            elapsed_seconds=elapsed_now(),per_size_resource=usage,weights="latest.pt:model states",
            **{role: final, f"independent_{role}_states_per_n": ROWS[role]}))


if __name__=="__main__":
    main()
