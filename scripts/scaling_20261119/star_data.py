"""Compact, arbitrary-integer exposure observations for the frozen positive-exposure STAR law.

The eta_s=1, zero-background prepared-click law reduces to one scalar h(q);
the joint singleton probability is gate*h(q). See parity_with_archive for an
independent check against the source law and its public operation ordering.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


N_VALUES = (8, 16, 32, 64, 128, 256)
ROLE_ROWS = {"qualification_train": 128, "qualification_validation": 64,
             "train": 4096, "validation": 256, "cal": 256, "test": 1024}
NAMESPACE = 2026111901
ROLE_IDS = {role: index for index, role in enumerate(ROLE_ROWS, 1)}
CHI = 0.01
GATE = 0.8404231092240892
# Per selected-photon-number downstream singleton probabilities from the canonical
# prepared instrument; conditional on the declared zero-dark, zero-background law.
SINGLETON = np.array([0., .30000000000000004, .4425000000000001, .6494693559556785,
    .7492219002861205, .8192689020113676, .8687923301668936, .90404510691837,
    .9293112119207227, .9475435206903963, .9607898333117629, .9704789821634618, .9776140308499541])
PHYSICS = {"chi": CHI, "eta_signal": 1., "eta_retrieval": .25, "eta_optical_path": .4,
           "eta_idler": .3, "noise": 0, "signal_gate": GATE, "selected_photon_limit": 12,
           "visible_cells": ["00","01","10","11"], "geometry":"STAR reference 0, ascending leaves, phases 0 then pi/2",
           "latent_prior":"iid uniform raw amplitude [0.2,1], iid uniform phase with reference 0, uniform V in {1,0.7}"}


def exposures(n):
    if n not in N_VALUES:
        raise ValueError("Unregistered N")
    return {**{f"write_{b}": b//(2*n) for b in (2**24, 2**27, 2**30)},
            **{f"read_{b}": b//(2*(n-1)) for b in (2**24, 2**27, 2**30)},
            **{f"native_{m}": 8192*n*m for m in (1,10,30)}}


def singleton_probability(q):
    d = 1 - CHI*(1-q)
    return (1-CHI)/d * np.polynomial.polynomial.polyval(CHI*q/d, SINGLETON)


def observation_probabilities(mode, visibility):
    """Visible probabilities only. The coherent/diagonal mixture is *after* the nonlinear detector."""
    mode = np.asarray(mode, dtype=np.complex128)
    visibility = np.asarray(visibility, dtype=np.float64)
    scale = PHYSICS["eta_retrieval"] * PHYSICS["eta_optical_path"] / 2
    z, leaf = mode[:, :1], mode[:, 1:]
    a = scale * abs(z + leaf)**2
    b = scale * abs(z + 1j*leaf)**2
    q = np.stack((a,b), -1).reshape(len(mode), -1)
    diagonal = np.repeat((abs(z)**2+abs(leaf)**2)*singleton_probability(scale), 2, axis=-1)
    h = visibility[:,None]*singleton_probability(q)+(1-visibility[:,None])*diagonal
    p = np.stack((1-GATE*CHI-(1-GATE)*h, (1-GATE)*h, GATE*(CHI-h), GATE*h), -1)
    if not np.isfinite(p).all() or p.min()<0 or not np.allclose(p.sum(-1),1,rtol=0,atol=2e-15):
        raise ArithmeticError("Invalid native probabilities")
    return p


def file_digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(2**20), b""):
            sha.update(block)
    return sha.hexdigest()


def load_cell(root, role, n, exposure, mmap=True):
    path = Path(root)/role/f"n{n}"
    if exposure not in exposures(n).values() or not (path/"manifest.json").is_file():
        raise ValueError("Unknown exposure or bank not finalized")
    return tuple(np.load(path/name, mmap_mode="r" if mmap else None, allow_pickle=False)
                 for name in (f"counts_e{exposure}.npy", "mode.npy", "visibility.npy"))


def generate_bank(root, role, n):
    rows, role_id = ROLE_ROWS[role], ROLE_IDS[role]
    dest = Path(root)/role/f"n{n}"
    dest.mkdir(exist_ok=False, parents=True)
    rng = np.random.default_rng(np.random.SeedSequence([NAMESPACE,role_id,n,0]))
    a = rng.uniform(.2, 1., size=(rows,n))
    phase = rng.uniform(-np.pi,np.pi, size=(rows,n))
    phase[:,0] = 0.
    mode = a*np.exp(1j*phase)/np.linalg.norm(a,axis=1,keepdims=True)
    v = np.array([1., .7])[rng.integers(2,size=rows)]
    np.save(dest/"mode.npy",mode)
    np.save(dest/"visibility.npy",v)
    p = observation_probabilities(mode, v)
    for e in sorted(set(exposures(n).values())):
        rng = np.random.default_rng(np.random.SeedSequence([NAMESPACE,role_id,n,1,e]))
        path = dest/f"counts_e{e}.npy"
        out = np.lib.format.open_memmap(path, mode="w+", dtype=np.int32, shape=p.shape)
        for start in range(0,rows,128):
            s = slice(start,start+128)
            # Draw rare cells first; deriving abundant n00 by subtraction avoids
            # cancellation in the tiny remaining probability of the last cell.
            prob = p[s][...,[1,3,2,0]].copy()
            counts = rng.multinomial(e,prob)
            out[s] = counts[...,[3,0,2,1]]
            if (out[s]<0).any() or not (out[s].sum(-1)==e).all():
                raise ArithmeticError("Visible count/exposure conservation failed")
        out.flush()
        del out
    meta = dict(n=n,role=role,rows=rows,namespace=NAMESPACE,role_id=role_id,physics=PHYSICS,
                exposures=exposures(n),files={p.name:file_digest(p) for p in sorted(dest.glob("*.npy"))})
    with (dest/"manifest.json").open("x",encoding="utf-8") as stream:
        json.dump(meta,stream,indent=2)
    print(role,n,rows,len(meta["files"]),flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root",type=Path)
    cli.add_argument("--roles",nargs="+",choices=ROLE_ROWS)
    cli.add_argument("--sizes",nargs="+",type=int,default=list(N_VALUES))
    args=cli.parse_args()
    if args.roles:
        if not args.root: cli.error("--root required")
        for role in args.roles:
            for n in args.sizes: generate_bank(args.root,role,n)
