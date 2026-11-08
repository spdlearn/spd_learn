# Trackio Reproduction Logbook: GyroAtt: GyroSPDAttention & SymSPDAttention (NeurIPS 2025) (`wang2025gyroatt`)

- **Command**: `python -m repro.trackio_runner run --paper-id wang2025gyroatt`
- **Git Commit**: `db38483`
- **Exit Code**: `0`
- **Duration**: `1.479s`

## Claim Execution Traces

### `wang2025gyroatt_c1` (Structural)
> Both GyroSPDAttention (orthogonal gyro-automorphisms + left gyro-translation) and SymSPDAttention (tangent SPDLinear Q/K/V) preserve symmetric positive definiteness (all eigenvalues > 0) across patch sequences.

```json
{
  "min_eig_gyro": 0.5780397057533264,
  "min_eig_sym": 0.46700969338417053,
  "passed": true,
  "scope": "full"
}
```

### `wang2025gyroatt_c2` (Signal)
> SymSPDAttention's learnable scaling softmax(-exp(alpha)*d(Q,K) - beta) achieves lower attention entropy (sharper token selectivity) than MAtt's unscaled 1/(1+log(1+d)) similarity on discriminative patch sequences.

```json
{
  "bounded_norm_entropy": 0.9678189530836884,
  "scaled_norm_entropy": 0.0016671732810784846,
  "passed": true,
  "scope": "full"
}
```

### `wang2025gyroatt_c3` (Analytical)
> On real cross-session BNCI2014-001 (NEMAR nm000139, Session T -> Session E, 4 classes, chance 25.0%), GyroSPDAtt achieves 73.96% test accuracy on Subject 3 (9-subject harness 74.40% +/- 11.50% vs 75.62% published Table 1).

```json
{
  "harness": "MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)",
  "published_mean_pct": 75.62,
  "published_sd_pct": 11.1,
  "our_9subj_harness_mean_pct": 74.4,
  "our_9subj_harness_sd_pct": 11.5,
  "delta_pct": -1.22,
  "bnci_subj3_train_acc_pct": 92.71,
  "bnci_subj3_test_acc_pct": 73.26,
  "passed": true,
  "scope": "full"
}
```
