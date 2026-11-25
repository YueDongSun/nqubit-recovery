# N-qubit recovery

Research code for estimating conditional single-excitation W states from photon counts and testing how much a bounded change in write amplitude and phase improves the next preparation.

## Run

Use Python 3.11 or later. From the repository root on Linux or macOS:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install torch numpy scipy psutil h5py PyYAML matplotlib
export PYTHONPATH=".:nqrec/src"

python -m training.check_forward
python -m experiments.scale.run_grid validate --n 3 4 8 16
python -m training.matrix runs/matrix.csv
```

To train one of the registered classifiers on a CUDA GPU:

```sh
python -m ml_recon.physics_hybrid_v10.long_projector_training \
  --matrix runs/matrix.csv --run-id joint_cs_32768_s7301 \
  --split-namespace compshare_scale_20261120 --source public-ff2a-projector-v1 \
  --output-root runs/classifier --device cuda
```

`training/` contains the 61 classifier recipes and 16 large-model recipes. `ml_recon/` and `nqrec/` implement the networks and four-outcome observation law. `experiments/` contains the supply, recovery, budget and measurement-replay studies. `scripts/` contains STAR training and high-dimensional amplitude experiments. Historical replay requires the corresponding external counts, calibrations or checkpoints; the synthetic forward check generates its own inputs.

## Reported simulation result

One frozen estimator trained jointly at N = 3, 4 and 8 was evaluated through N = 256. At 77,000 READs per trajectory, an initial centered log-amplitude RMS error of 0.35 and a scaled relative-phase RMS error of 0.70 rad, one bounded correction changed mean conditional W-state overlap by +0.1154 at N = 3 and −0.0008 at N = 256 relative to no action (128 paired simulated initial states per size).


## Experiment control, V10 training campaign, and FPGA firmware (2026-11-25)

Three additions published as the apparatus-and-training provenance for the
paper:

- `system_control/` — the closed-loop experiment control core (runner state
  machine, AWG upload, serial acquisition orchestration, reconstruction,
  optional ML feedback), with `AWGcontrol_port/` (PyVISA/WebSocket AWG
  transports) and `VerilogHDL/` (Cyclone III histogram firmware,
  75-cell/256-cell layouts).
- `ml_recon/` at the repository root now also carries the control-loop
  import closure (feedback models, tomography, physics-hybrid v4-v9
  stacks, V10 checkpoint/hardware contracts) that `system_control/`
  imports.
- `v10_campaign_20261001/` — the frozen Track A / Track B training
  campaign snapshot: per-track self-contained `ml_recon` trees at the exact
  source revisions of the reported classifier confirmation runs and the
  four-cell likelihood-ladder fit, plus the self-contained Track B ladder
  fitters.

Entrances: `python system_control/run_experiment_ui.py` for the unified
control UI (PyQt5); each `v10_campaign_20261001/track_*/README` documents
its own entry points. All campaign code is research-only: nothing here
writes hardware state, and the bench instrument defaults (AWG pair
`192.168.2.122::4011` / `192.168.2.110::3992`, serial port) are overridable
wiring documentation.
