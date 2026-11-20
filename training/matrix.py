"""Generate the 61 classifier recipes used for the scale, component and independent bank studies."""

import argparse
import csv
from pathlib import Path

SPLIT = "compshare_scale_20261120"
CAPACITIES = {"cs": "C-S", "cm": "C-M", "cl": "C-L"}
SEEDS = (7301, 7302, 7303)
MODULES = ("noise_ce_control", "noise_moe", "rho8", "esm", "csm", "laplacian", "fd", "all_no_fd")


def row(run_id, study, cap, states, objective, seed, steps, *, lr=0.0002, profile="", bank=""):
    return dict(
        run_id=run_id, study=study, capacity=CAPACITIES[cap], train_states_per_n=states,
        objective=objective, optimizer_seed=seed, learning_rate=lr, effective_batch=32,
        train_reads_per_state=4096, terminal_updates=steps,
        switch_after_update=43690 if objective == "staged" else "", sizes="3|4|8",
        idea_profile=profile, train_namespace=bank,
    )


def recipes():
    for seed in SEEDS:
        for cap in CAPACITIES:
            for states in (1024, 8192, 32768):
                steps = 131072 if cap == "cs" and states != 8192 else 32768
                yield row(f"joint_{cap}_{states}_s{seed}", "main", cap, states, "joint", seed, steps)
        for objective in ("state", "staged"):
            yield row(f"{objective}_cs_32768_s{seed}", "main", "cs", 32768, objective, seed, 131072)
    for seed in SEEDS[:2]:
        for cap in ("cs", "cl"):
            for lr in (0.00005, 0.0008):
                yield row(f"lr_{cap}_{lr}_s{seed}", "lr_sensitivity", cap, 8192, "joint", seed, 4096, lr=lr)
    for study in ("module_attribution", "train_bank_replication"):
        replicate = study == "train_bank_replication"
        for seed in SEEDS[:2]:
            for profile in (("noise_ce_control", "csm") if replicate else MODULES):
                name = f"module_{profile}_s{seed}" + ("_bank_b" if replicate else "")
                yield row(name, study, "cs", 8192, "joint", seed, 8192, profile=profile,
                          bank=f"{SPLIT}_bank_b" if replicate else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = list(recipes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
