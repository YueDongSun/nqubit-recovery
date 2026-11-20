"""Compute paired terminal risk/retention intervals across four registered sizes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

SIZES=(8,16,32,64)
SEEDS=(83,89,97)
DRAWS=6000


def interval(point, draws):
    pointwise = np.quantile(draws,[.025,.975])
    family = np.quantile(draws,[.025/len(SIZES),1-.025/len(SIZES)])
    return {"estimate":float(point),"pointwise95":pointwise.tolist(),"four_size95":family.tolist()}


def analyze(archive, seeds=SEEDS):
    out={}
    for n in SIZES:
        values={}; vis=archive[f"n{n}_visibility"]
        for law in ("corrected","archived"):
            for key in ("frozen","prior"):
                values[law,key]=archive[f"n{n}_{law}_{key}"]
            for key in ("corrected","rehearsal","fixed"):
                names=[f"{key}_s{s}"+(f"_n{n}" if key=="fixed" else "") for s in seeds]
                values[law,key]=np.mean([archive[f"n{n}_{law}_{name}"] for name in names],axis=0)
        rng=np.random.default_rng([811099,n])
        weight=np.zeros((DRAWS,len(vis)),dtype=np.float64)
        for v in np.unique(vis):
            indices=np.flatnonzero(vis==v)
            weight[:,indices]=rng.multinomial(len(indices),np.full(len(indices),1/len(indices)),size=DRAWS)
        weight/=len(vis)
        contrasts={}
        for law in ("corrected","archived"):
            contrasts[law]={}
            for a,b in (("corrected","fixed"),("corrected","frozen"),("corrected","prior"),
                        ("rehearsal","corrected"),("rehearsal","frozen"),("fixed","frozen")):
                delta=values[law,a]-values[law,b]
                contrasts[law][a+"_minus_"+b]=interval(np.mean(delta),weight@delta)
        def ratio(numerator,denominator):
            den=weight@denominator
            if np.quantile(den,.025/len(SIZES))<=0:
                return {"estimate":None,"reason":"denominator does not have a positive four-size lower bound"}
            return interval(numerator.mean()/denominator.mean(),(weight@numerator)/den)
        corrected_gain=values["corrected","frozen"]-values["corrected","corrected"]
        gain_rehearsed=values["corrected","frozen"]-values["corrected","rehearsal"]
        old_penalty=values["archived","corrected"]-values["archived","frozen"]
        old_avoided=values["archived","corrected"]-values["archived","rehearsal"]
        out[str(n)]={"contrasts":contrasts,"corrected_gain_retained_with_rehearsal":ratio(gain_rehearsed,corrected_gain),
                     "old_law_penalty_avoided_with_rehearsal":ratio(old_avoided,old_penalty)}
    return out


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("confirmation_archive",type=Path);cli.add_argument("output",type=Path)
    args=cli.parse_args()
    with np.load(args.confirmation_archive,allow_pickle=False) as ar:
        systems=analyze(ar)
    result={"scope":"Fixed seeds; paired resampling within visibility; Bonferroni over four N for each named contrast",
            "draws":DRAWS,"systems":systems}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf8")
