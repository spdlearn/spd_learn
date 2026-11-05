# Trackio Reproduction Logbook: ARMAGNAC: Parametric Harmonic-Arithmetic SPD BatchNorm (HAL / Signal Processing 2026) (`gallet2026armagnac`)

- **Command**: `python -m repro.trackio_runner run --paper-id gallet2026armagnac`
- **Git Commit**: `b3cf097`
- **Exit Code**: `0`
- **Duration**: `0.147s`

## Claim Execution Traces

### `gallet2026armagnac_c1` (Structural)
> SPDBatchNormHarmonicArithmetic's parametric mean M_alpha = H #_alpha A equals the batch Harmonic mean as alpha -> 0, the batch Arithmetic mean as alpha -> 1, and the Geometric-Harmonic-Arithmetic midpoint at alpha = 0.5.

```json
{
  "harmonic_endpoint_err": 1.9747640154760356e-15,
  "arithmetic_endpoint_err": 1.151395379243484e-14,
  "passed": true,
  "scope": "full"
}
```

### `gallet2026armagnac_c2` (Signal)
> When optimizing M_alpha to match target SPD distributions generated around the Harmonic mean vs Arithmetic mean, gradient descent on SPDBatchNormHarmonicArithmetic drives alpha toward 0 for Harmonic targets and toward 1 for Arithmetic targets.

```json
{
  "learned_alpha_for_harmonic_target": 0.02373824471434667,
  "learned_alpha_for_arithmetic_target": 0.9848944355058329,
  "passed": true,
  "scope": "full"
}
```

### `gallet2026armagnac_c3` (Analytical)
> SPDBatchNormHarmonicArithmetic executes faster than multi-iteration Karcher flow (SPDBatchNormMean with n_iter=5) while achieving >= test accuracy inside an SPDNet classifier.

```json
{
  "armagnac_10_calls_sec": 0.00699749996419996,
  "karcher5_10_calls_sec": 0.01684150006622076,
  "test_acc": 0.949999988079071,
  "passed": true,
  "scope": "toy"
}
```
