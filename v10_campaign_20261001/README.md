# V10 Track A/B training campaign (frozen research snapshot)

This directory is the frozen training code for the V10 research campaign of
2026-10-01/03 (`v10_campaign_20261001`), published as the provenance record
for the classifier and feedback results reported in the paper.

Each subdirectory is a **self-contained frozen snapshot** of one research
track at the exact source revision used for the reported runs. The two
tracks evolved independently, so the `ml_recon` trees inside `track_a/` and
`track_b/` are separate copies and are not interchangeable with the
repository-root `ml_recon/` package or with each other. Import paths are
relative to the snapshot root you are working in.

## Track A — research stream classifier training

Frozen at the branch that produced the reported classifier confirmation
results (stream plans `41762351` 1000/200 and `6d84fd31` 10000/200,
CRN init-seed panels, curriculum, checkpoint resume).

Entry points (run from `track_a/`, repository root on `sys.path`):

```sh
python -m ml_recon.physics_hybrid_v10.research_stream_training \
  --task-pool small_n_classifier --device cuda \
  --output-dir runs/stream --run-id <run-id> --seed 20261001 \
  --train-rows 1000 --validation-rows 200
python -m ml_recon.physics_hybrid_v10.research_ensemble_evaluate \
  --checkpoints runs/stream/<run-id>
```

`research_stream_training.py` mirrors the V10 trainer semantics (AdamW
3e-4/1e-5 cosine, AMP, clip 1.0, committed-step post-hook) and adds the
campaign-era extensions: `--init-seed` CRN panels, curriculum via
`--curriculum-standard-epochs`, and `--resume-from` checkpoint resume with
metrics bookkeeping rebuild. `loss_profiles.py` registers the loss-weight
profiles; the reported confirmation used `balanced_structure_v1`.

Training data comes from the registered rung plans in
`ml_recon/data_v2_4/rung_plan.py`. Historical stream stores (the 1k and 10k
classifier-cell persistent stores) were mounted under a private dataset
root during the campaign and are **not part of this snapshot**; the stream
loader validates any dataset root you supply against the same rung-plan
identity.

## Track B — RealFD four-cell likelihood ladder and long training

Frozen at the branch that produced the reported four-cell likelihood-ladder
fit (`gated_phys` 4Q, `anchor_only` 3Q) and the fixed-recipe long training
of the behavior model.

Entry points (run from `track_b/`):

```sh
python -m ml_recon.physics_hybrid_v10.trackb_realfd_v1.cli --help
python -m ml_recon.physics_hybrid_v10.trackb_realfd_v1.behavior_training_v2 --help
```

`scripts/track_b_ladder/` holds the self-contained ladder fitters behind
the reported held-out four-cell NLL table (v4 anchor-blend, v4b
`gated_phys`, validation-fold evaluation):

```sh
python scripts/track_b_ladder/research_trackb_v4b.py \
  --dataset-root <v24-root> --build-id <build-id> ...
```

These scripts require the external V2.4 count records and calibration
artifacts; they are published as the exact fitting procedure, not as a
turnkey dataset.

## Research-only scope

Everything in this snapshot is research code: it never writes hardware
state, uploads AWG sequences, programs the FPGA, or requests data
collection. Model selection, promotion, and any production gate remain
outside this snapshot. Checkpoint metadata recorded during the campaign
binds the internal branch names under which the runs were executed;
loading those checkpoints elsewhere requires supplying the same branch
identifier.
