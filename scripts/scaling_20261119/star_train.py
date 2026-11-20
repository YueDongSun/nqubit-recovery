"""Paired shared versus specialist training on integer-exposure STAR observations.

Use the retained unified_model.py beside this file. The exact archived decoder weights and layout are shared by every arm.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch.nn import functional as fn

import unified_model
import star_data


SIZES = (8, 16, 32, 64, 128, 256)
SEEDS = (61, 62, 63)
HIDDEN, RANK = 256, 4
BATCH = 64
LR = 3.e-4
MATCH_STEPS, LONG_STEPS = 20_000, 120_000
CHECK_EVERY = 600
QUAL_STEPS_PER_SIZE = 100
WALL_CAP = 8 * 3600


def atom_json(path, value):
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(value, allow_nan=False, indent=2), encoding="utf8")
    temp.replace(path)


def atom_torch(path, state):
    temp = path.with_suffix(".pt.tmp")
    torch.save(state, temp)
    temp.replace(path)


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as fd:
        for chunk in iter(lambda: fd.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@contextmanager
def owner(directory):
    with (directory / "owner.lock").open("a+b") as fd:
        if os.name == "nt":
            import msvcrt
            if not fd.tell():
                fd.write(b"0")
                fd.flush()
            fd.seek(0)
            msvcrt.locking(fd.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def density_factors(model, x):
    """Compact execution of Unified_star_density_net's decoder, with its own modules."""
    b, width = x.shape
    n = 1 + (width - 1) // 8
    e = x[:, -1]
    metadata = torch.stack((torch.full_like(e, math.log(n) / unified_model.LOG_SIZE_SCALE),
                            (e - math.log(8192*n))/unified_model.LOG_EXPOSURE_SCALE), -1)
    counts = x[:, :-1].reshape(b, n-1, 8) / torch.logaddexp(e, torch.zeros_like(e))[:, None, None]
    leaf = model.leaf_encoder(torch.cat((counts, metadata[:, None, :].expand(-1, n-1, -1)), -1))
    pooled = leaf.mean(1)
    decoded = model.leaf_decoder(torch.cat((leaf, pooled[:, None].expand_as(leaf)), -1))
    ref = model.reference_decoder(torch.cat((pooled, metadata), -1))
    values = torch.cat((ref[:, None], decoded), 1)
    rank = model.rank_columns
    return torch.complex(values[..., :rank], values[..., rank:2*rank]), fn.softplus(values[..., -1])


def state_risk(model, x, c, visibility):
    """Per-row squared Frobenius risk without forming the N x N Gram matrix."""
    a, d = density_factors(model, x)
    v = visibility.reshape(-1)
    row_power = a.abs().square().sum(-1)
    z = (row_power + d).sum(1)
    gram = a.mH @ a
    predicted_norm = (gram.abs().square().sum((1, 2)) + ((2*row_power + d)*d).sum(1)) / z.square()
    p = c.abs().square()
    truth_norm = v.square() + (1-v.square())*p.square().sum(1)
    overlap = ((c.conj().unsqueeze(1) @ a).abs().square().sum((1, 2))*v
               + (p*(d+(1-v[:, None])*row_power)).sum(1)) / z
    return predicted_norm + truth_norm - 2*overlap


def features(raw, exposure):
    x = np.empty((len(raw), raw.shape[1]*raw.shape[2]+1), dtype=np.float64)
    np.log1p(raw.reshape(len(raw), -1), out=x[:, :-1], dtype=np.float64)
    x[:, -1] = math.log(exposure)
    return x


def matched_rows(seed, n, local_update, count):
    """Replayable schedule independent of other Ns, machines and data-loading speed."""
    # Randomize each cycle of the nine nominal regimes. Duplicate actual E values
    # share the same bank while retaining each regime's prespecified weight.
    cycle, pos = divmod(local_update, 9)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([2026111903, seed, n, cycle])))
    label_index = int(rng.permutation(9)[pos])
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([2026111904, seed, n, local_update])))
    rows = rng.choice(count, size=BATCH, replace=False)
    return label_index, rows


def open_banks(directory, ns, qualify):
    banks, identities = {}, {}
    for label, role, expected in (("train", "qualification_train" if qualify else "train", 128 if qualify else 4096),
                                  ("validation", "qualification_validation" if qualify else "validation", 64 if qualify else 256)):
        for n in ns:
            manifest = directory / role / f"n{n}" / "manifest.json"
            identities[f"{role}/n{n}"] = file_sha(manifest)
            exps = list(star_data.exposures(n).values())
            record = json.loads(manifest.read_text(encoding="utf-8"))
            names = {"mode.npy", "visibility.npy"} | {f"counts_e{e}.npy" for e in exps}
            if (record.get("n") != n or record.get("role") != role or record.get("rows") != expected
                    or record.get("exposures") != star_data.exposures(n)
                    or not names.issubset(record["files"])):
                raise ValueError(f"Incompatible bank manifest {manifest}")
            for name in names:
                if file_sha(manifest.parent / name) != record["files"][name]:
                    raise ValueError(f"Transfer integrity failure for {manifest.parent/name}")
            if len(exps) != 9:
                raise ValueError("Expected three budget regimes of three exposures")
            arrays = {}
            for exposure in dict.fromkeys(exps):
                counts, mode, v = star_data.load_cell(directory, role, n, exposure, mmap=True)
                if (counts.shape != (expected, 2*(n-1), 4) or mode.shape != (expected, n)
                        or v.size != expected or counts.dtype != np.int32 or mode.dtype != np.complex128
                        or not np.isfinite(mode).all() or not np.all(np.isin(v, [1.0, 0.7]))
                        or not np.allclose(np.square(abs(mode)).sum(1), 1., atol=1.e-12, rtol=0)):
                    raise ValueError(f"Incompatible {role} N={n} E={exposure}")
                if np.any(counts < 0) or np.any(counts.sum(-1, dtype=np.int64) != exposure):
                    raise ValueError(f"Count conservation failed for {role} N={n} E={exposure}")
                arrays[exposure] = (counts, mode, v.reshape(-1))
            banks[label, n] = (exps, arrays)
    return banks, identities


def minibatch(arrays, exposure, rows, device):
    counts, c, v = arrays
    return tuple(torch.as_tensor(a, device=device) for a in (
        features(counts[rows], exposure), np.array(c[rows]), np.array(v[rows], dtype=np.float64)))


def do_update(model, optimizer, bank, seed, n, number, micro, device):
    exposures, arrays = bank
    index, rows = matched_rows(seed, n, number, len(next(iter(arrays.values()))[0]))
    e = exposures[index]
    inputs = minibatch(arrays[e], e, rows, device)
    optimizer.zero_grad(set_to_none=True)
    for j in range(0, BATCH, micro):
        xx, c, v = [t[j:j+micro] for t in inputs]
        (state_risk(model, xx, c, v).sum()/BATCH).backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
    optimizer.step()
    return norm


@torch.no_grad()
def evaluate(model, banks, ns, micro, device):
    model.eval()
    per_n = {}
    for n in ns:
        exps, arrays = banks["validation", n]
        scores = {}
        for e, data in arrays.items():
            count = len(data[0])
            out = torch.zeros((), dtype=torch.float64, device=device)
            for j in range(0, count, micro):
                vals = minibatch(data, e, slice(j, min(j+micro, count)), device)
                out += state_risk(model, *vals).sum()
            scores[str(e)] = float(out/count)
        per_n[str(n)] = {"risk": np.mean([scores[str(e)] for e in exps]).item(), "by_exposure": scores}
    model.train()
    return {"risk": np.mean([v["risk"] for v in per_n.values()]).item(), "by_n": per_n}


def clone_weights(model):
    return {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}


def new_best():
    return {"risk": float("inf"), "step": 0, "state_dict": None}


def run(args):
    qualify = args.mode == "qualify"
    ns = SIZES if args.n == 0 else (args.n,)
    cap = QUAL_STEPS_PER_SIZE*len(ns) if qualify else LONG_STEPS if args.n == 0 or args.seed==61 else MATCH_STEPS
    output = Path(args.output)
    if args.resume and not output.is_dir():
        raise ValueError("No existing job directory for resume")
    output.mkdir(parents=True, exist_ok=args.resume)
    with owner(output):
        if (output/"result.json").exists():
            raise ValueError("Run already completed")
        if args.resume:
            deadline = json.loads((output/"origin.json").read_text())["deadline_unix"]
        else:
            start = time.time()
            deadline = start + (1800 if qualify else WALL_CAP)
            atom_json(output/"origin.json", {"started_unix": start, "deadline_unix": deadline})
        torch.set_num_threads(1)
        device = torch.device("cuda")
        if torch.cuda.device_count()!=1:
            raise RuntimeError("Bind exactly one GPU using CUDA_VISIBLE_DEVICES")
        torch.manual_seed(args.seed)
        model = unified_model.Unified_star_density_net(hidden=HIDDEN, rank_columns=RANK).to(device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        banks, bank_ids = open_banks(Path(args.banks), ns, qualify)
        recipe = {"namespace":"star_matched_budget_20261119_v1","seed":args.seed,"sizes":list(ns),
                  "batch":BATCH,"microbatch":args.microbatch,"lr":LR,"step_cap":cap,
                  "bank_manifests":bank_ids,"sampling":"paired_counter_per_N_v1",
                  "architecture": model.architecture_metadata(),
                  "source":{n:file_sha(p) for n,p in (("trainer",Path(__file__)),("model",Path(unified_model.__file__)),
                                                       ("data",Path(star_data.__file__)))},
                  "torch":str(torch.__version__),"numpy":np.__version__,"python":platform.python_version(),
                  "cuda":str(torch.version.cuda),"device":torch.cuda.get_device_name(0)}
        if args.resume:
            ck = torch.load(output/"latest.pt", weights_only=True, map_location="cpu")
            if ck["recipe"] != recipe or ck["deadline_unix"] != deadline:
                raise ValueError("Runtime, bank, schedule, or deadline changed across resume")
            model.load_state_dict(ck["state_dict"])
            optimizer.load_state_dict(ck["adam"])
            torch.set_rng_state(ck["cpu_rng"])
            torch.cuda.set_rng_state(ck["cuda_rng"])
            step,best,history,timing=ck["step"],ck["best"],ck["history"],ck["timing"]
            del ck
        else:
            step, best, history, timing = 0, {"matched":new_best(),"full":new_best()}, [], {str(n):[] for n in ns}
        if qualify:
            for n in ns:
                e = next(iter(banks["train",n][1]))
                xx,c,v = minibatch(banks["train",n][1][e],e,slice(0,2),device)
                direct = model(xx)
                p = c.abs().square()
                truth = v[:,None,None]*(c.unsqueeze(-1)*c.conj().unsqueeze(1)) + (1-v[:,None,None])*torch.diag_embed(p)
                explicit = (direct-truth).abs().square().sum((1,2))
                compact = state_risk(model,xx,c,v)
                torch.testing.assert_close(compact,explicit,rtol=1e-11,atol=1e-13)
                grad1=torch.autograd.grad(explicit.mean(),tuple(model.parameters()))
                grad2=torch.autograd.grad(compact.mean(),tuple(model.parameters()))
                for g,h in zip(grad1,grad2):
                    torch.testing.assert_close(g,h,rtol=1e-8,atol=2e-12)
        def checkpoint():
            ck = {"recipe":recipe,"step":step,"next_per_n_update":[(step+len(ns)-1-i)//len(ns) for i in range(len(ns))],
                  "state_dict":clone_weights(model),"adam":optimizer.state_dict(),
                  "cpu_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state(),
                  "deadline_unix":deadline,"best":best,"history":history,"timing":timing}
            atom_torch(output/"latest.pt",ck)
            atom_json(output/"status.json",{"state":"running","step":step,"pid":os.getpid(),"deadline_unix":deadline})
        if not args.resume:
            checkpoint()
        while step<cap:
            if time.time()>=deadline:
                raise TimeoutError("Original job wall-clock budget reached")
            idx = step % len(ns)
            n = ns[idx]
            begin=time.perf_counter()
            do_update(model,optimizer,banks["train",n],args.seed,n,step//len(ns),args.microbatch,device)
            step+=1
            if qualify:
                torch.cuda.synchronize()
                timing[str(n)].append(time.perf_counter()-begin)
            matched_interval = 1000*len(ns)
            matched_cap = MATCH_STEPS*len(ns)
            matched_eval = step<=matched_cap and step%matched_interval==0
            full_eval = step%6000==0 and cap==LONG_STEPS
            if matched_eval or full_eval or step==cap:
                validation = evaluate(model,banks,ns,args.microbatch,device)
                record = {"step":step,**validation}
                history.append(record)
                for kind,allowed in (("matched",matched_eval or (qualify and step==cap)),
                                     ("full",full_eval or (qualify and step==cap))):
                    if allowed and validation["risk"] < best[kind]["risk"]:
                        best[kind] = {"step":step,"risk":validation["risk"],"state_dict":clone_weights(model)}
                if step==matched_cap or (qualify and step==cap):
                    atom_torch(output/"matched_selected.pt",{"recipe":recipe,**best["matched"]})
                    atom_torch(output/"matched_terminal.pt",{"recipe":recipe,"step":step,"state_dict":clone_weights(model)})
            if step%CHECK_EVERY==0 or step==matched_cap or step==cap:
                checkpoint()
        final_best = best["full" if cap==LONG_STEPS else "matched"]
        atom_torch(output/"selected.pt",{"recipe":recipe,**final_best})
        if qualify:
            ck=torch.load(output/"latest.pt",weights_only=True,map_location="cpu")
            target_weights=clone_weights(model)
            model.load_state_dict(ck["state_dict"])
            if any(not torch.equal(model.state_dict()[k].cpu(),v) for k,v in target_weights.items()):
                raise ValueError("Checkpoint reload diverged")
            def replay_one():
                model.load_state_dict(ck["state_dict"])
                optimizer.load_state_dict(torch.load(output/"latest.pt",weights_only=True,map_location="cpu")["adam"])
                torch.set_rng_state(ck["cpu_rng"])
                torch.cuda.set_rng_state(ck["cuda_rng"])
                next_n=ns[step % len(ns)]
                do_update(model,optimizer,banks["train",next_n],args.seed,next_n,step//len(ns),args.microbatch,device)
                return clone_weights(model)
            first, second=replay_one(), replay_one()
            if any(not torch.equal(first[k], second[k]) for k in first):
                raise ArithmeticError("Next optimizer update differed after fresh checkpoint reload")
        result={"state":"completed","step":step,"recipe":recipe,"best_step":final_best["step"],
                "matched_selected_step":best["matched"]["step"],"validation_history":history,
                "updates_by_n":{str(n):step//len(ns) for n in ns},"deadline_unix":deadline,
                "allocator_reserved_bytes":torch.cuda.max_memory_reserved(),
                "elapsed_s":time.time()-json.loads((output/"origin.json").read_text())["started_unix"]}
        if qualify:
            result["dense_value_gradient_parity"] = True
            result["next_update_replay_exact"] = True
            result["extra_replay_fixture_updates"] = 2
            result["median_step_seconds_by_n"]={n:float(np.median(s)) for n,s in timing.items()}
            result["mean_step_seconds_by_n"]={n:float(np.mean(s)) for n,s in timing.items()}
        atom_json(output/"result.json",result)
        atom_json(output/"status.json",{"state":"completed","step":step})
        print(json.dumps({k:result[k] for k in ("state","step","best_step","elapsed_s")}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--banks",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--mode",choices=("qualify","train"),required=True)
    parser.add_argument("--n",type=int,choices=(0,*SIZES),default=0,help="0 uses a shared model on all six sizes")
    parser.add_argument("--seed",type=int,choices=SEEDS,required=True)
    parser.add_argument("--microbatch",type=int,choices=(8,16,32,64),default=16)
    parser.add_argument("--resume",action="store_true")
    run(parser.parse_args())
