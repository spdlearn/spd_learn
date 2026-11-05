# Trackio Reproduction Logbook: GyroSPD: Gyrovector Space Neural Networks on Matrix Manifolds (ICML 2023) (`nguyen2023gyrospd`)

- **Command**: `python -m repro.trackio_runner run --paper-id nguyen2023gyrospd`
- **Git Commit**: `3a9e18d`
- **Exit Code**: `0`
- **Duration**: `0.630s`

## Claim Execution Traces

### `nguyen2023gyrospd_c1` (Structural)
> GyroSPDLayer combines Stiefel bilinear projection W^T X W with left gyro-translation B oplus P = B^{1/2} P B^{1/2} under AIM or exp(log(B) + log(P)) under LEM, preserving SPD structure.

```json
{
  "min_eig_output": 0.35524412989616394,
  "passed": true,
  "scope": "full"
}
```

### `nguyen2023gyrospd_c2` (Signal)
> Unlike un-biased orthogonal BiMap which cannot shift the log-determinant or trace beyond the subspace projection of X, GyroSPDLayer's learnable SPD bias B translates the output spectrum to arbitrary positive scales.

```json
{
  "input_trace": 8.028207778930664,
  "gyro_translated_trace": 55.426971435546875,
  "passed": true,
  "scope": "full"
}
```

### `nguyen2023gyrospd_c3` (Analytical)
> On real cross-session BNCI2014-001 (NEMAR nm000139, Session T -> Session E, 4 classes, chance 25.0%), GyroSPD achieves 72.92% test accuracy on Subject 3 when gyro-bias translation is combined with full-rank initialization (vs 55.56% under aggressive rank compression).

```json
{
  "harness": "MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)",
  "bnci_subj3_train_acc_pct": 60.07,
  "bnci_subj3_test_acc_pct": 55.21,
  "chance_level_pct": 25.0,
  "passed": false,
  "scope": "full"
}
```
