# ml_recon support modules (control-loop closure)

This `ml_recon/` tree is the **import closure of the experiment control
core** (`../system_control/`), frozen at the same revision. It contains the
feedback models, tomography, physics-hybrid model stacks (v4-v9) and the
V10 campaign authority helpers that the control loop actually imports —
not the full research `ml_recon` development tree.

For the V10 classifier/feedback **training** code, see
`../v10_campaign_20261001/` (per-track frozen snapshots). For the
paper-reanalysis package, see `../nqrec/`.

Import chain of the control loop (illustrative):

```
system_control.experiment_control.runner
  -> ml_recon.feedback_model                 (trained feedback model interface)
  -> ml_recon.physics_hybrid_v5.*            (command contract, projective action)
  -> ml_recon.physics_hybrid_v7/v8/v9.*      (feedback bridges, count contracts)
  -> ml_recon.physics_hybrid_v10.*           (checkpoint schema, hardware contract)
  -> ml_recon.quantum_state / quantum_representation   (tomography, state families)
  -> ml_recon.aod_calibration / deep_models / measured_data / physics_identification
```

The package `__init__.py` is intentionally minimal; import submodules
directly (e.g. `from ml_recon.feedback_model import TrainedFeedbackModel`).
`ml_recon/aod_calibration/neural_net_correction.py` imports `torch`
unconditionally and raises if it is missing; everything else degrades
gracefully without optional dependencies.
