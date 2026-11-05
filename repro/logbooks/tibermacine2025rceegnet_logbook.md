# Trackio Reproduction Logbook: RCEEGNet: Geometry-Aware CNN on SPD Manifolds (IJCNN 2025) (`tibermacine2025rceegnet`)

- **Command**: `python -m repro.trackio_runner run --paper-id tibermacine2025rceegnet`
- **Git Commit**: `3a9e18d`
- **Exit Code**: `0`
- **Duration**: `22.281s`

## Claim Execution Traces

### `tibermacine2025rceegnet_c1` (Structural)
> RCEEGNet maps raw EEG (B, C, T) through temporal Conv1d, TraceNorm + Shrinkage covariance pooling, stacked BiMap/ReEig/SPDBatchNormMean layers, full-matrix LogEig(upper=False), and 2D convolutional refinement to class logits while maintaining positive definiteness at every internal SPD stage.

```json
{
  "output_shape": [
    4,
    4
  ],
  "passed": true,
  "scope": "full"
}
```

### `tibermacine2025rceegnet_c2` (Signal)
> Learnable Shrinkage regularization ensures bounded condition numbers on short temporal windows even when the number of temporal filters exceeds the effective rank of the signal.

```json
{
  "frontend_input_grad_flowing": true,
  "passed": true,
  "scope": "full"
}
```

### `tibermacine2025rceegnet_c3` (Analytical)
> On real cross-session BNCI2014-001 (NEMAR nm000139, Session T -> Session E, 4 classes, chance 25.0%), RCEEGNet achieves 64.58% test accuracy on Subject 3 when batch-norm momentum is calibrated (momentum=0.4), whereas default low momentum (0.1) under short training schedules causes running-mean lag that collapses evaluation accuracy to 26.39% (near chance).

```json
{
  "harness": "MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)",
  "bnci_subj3_uncalibrated_mom01_test_acc_pct": 47.22,
  "bnci_subj3_calibrated_mom04_test_acc_pct": 66.67,
  "momentum_calibration_gain_pp": 19.45,
  "passed": false,
  "scope": "baseline_adjusted"
}
```
