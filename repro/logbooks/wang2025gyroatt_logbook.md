# Trackio Reproduction Logbook: GyroAtt: GyroSPDAttention & SymSPDAttention (NeurIPS 2025) (`wang2025gyroatt`)

- **Command**: `python -m repro.trackio_runner run --paper-id wang2025gyroatt`
- **Git Commit**: `b3cf097`
- **Exit Code**: `0`
- **Duration**: `0.609s`

## Claim Execution Traces

### `wang2025gyroatt_c1` (Structural)
> Both GyroSPDAttention (orthogonal gyro-automorphisms + left gyro-translation) and SymSPDAttention (tangent SPDLinear Q/K/V) strictly preserve symmetric positive definiteness (all eigenvalues > 0) across patch sequences.

```json
{
  "min_eig_gyro": 0.5778074264526367,
  "min_eig_sym": 0.5684083700180054,
  "passed": true,
  "scope": "full"
}
```

### `wang2025gyroatt_c2` (Signal)
> SymSPDAttention's learnable scaling softmax(-exp(alpha)*d(Q,K) - beta) achieves lower attention entropy (sharper token selectivity) than unscaled 1/(1+log(1+d)) attention on discriminative patch sequences.

```json
{
  "bounded_norm_entropy": 0.9678189530836884,
  "scaled_norm_entropy": 0.0016671732810784846,
  "passed": true,
  "scope": "full"
}
```

### `wang2025gyroatt_c3` (Analytical)
> GyroSPDAtt (both gyro and sym modes) trains end-to-end and achieves >0.75 test accuracy on synthetic motor imagery EEG decoding.

```json
{
  "test_acc": 1.0,
  "passed": true,
  "scope": "toy"
}
```
