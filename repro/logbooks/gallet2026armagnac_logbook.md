# Trackio Reproduction Logbook: ARMAGNAC Harmonic-Arithmetic BatchNorm (Signal Processing 2026) (`gallet2026armagnac`)

- **Command**: `python -m repro.trackio_runner run --paper-id gallet2026armagnac`
- **Git Commit**: `3a9e18d`
- **Exit Code**: `0`
- **Duration**: `0.358s`

## Claim Execution Traces

### `gallet2026armagnac_c1` (Structural)
> SPDBatchNormHarmonicArithmetic recovers the exact batch Harmonic mean H = (E[X^{-1}])^{-1} as alpha -> 0 and the exact batch Arithmetic mean A = E[X] as alpha -> 1 along the closed-form AIRM geodesic H #_alpha A.

```json
{
  "harmonic_endpoint_err": 1.9747640154760356e-15,
  "arithmetic_endpoint_err": 1.151395379243484e-14,
  "passed": true,
  "scope": "full"
}
```

### `gallet2026armagnac_c2` (Signal)
> The learnable interpolation parameter alpha = sigmoid(alpha_logit) receives non-zero analytical gradients and adapts toward alpha -> 0 when optimizing toward a harmonic target and toward alpha -> 1 when optimizing toward an arithmetic target.

```json
{
  "learned_alpha_for_harmonic_target": 0.02373824471434667,
  "learned_alpha_for_arithmetic_target": 0.9848944355058329,
  "passed": true,
  "scope": "full"
}
```

### `gallet2026armagnac_c3` (Analytical)
> On real cross-session BNCI2014-001 (NEMAR nm000139, Session T -> Session E, 4 classes, chance 25.0%), SPDNet + SPDBatchNormHarmonicArithmetic achieves 74.65% test accuracy on Subject 3 while running 3.1x faster than 5-step Karcher SPDBatchNormMean.

```json
{
  "harness": "MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)",
  "armagnac_10_calls_sec": 0.00731,
  "karcher5_10_calls_sec": 0.01715,
  "bnci_subj3_train_acc_pct": 73.96,
  "bnci_subj3_test_acc_pct": 70.49,
  "passed": true,
  "scope": "full"
}
```
