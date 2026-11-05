# Trackio Reproduction Logbook: CorNet: Riemannian Networks over Full-Rank Correlation Matrices (ICML 2026) (`chen2026cornet`)

- **Command**: `python -m repro.trackio_runner run --paper-id chen2026cornet`
- **Git Commit**: `b3cf097`
- **Exit Code**: `0`
- **Duration**: `0.101s`

## Claim Execution Traces

### `chen2026cornet_c1` (Structural)
> CorNet preserves unit-diagonal positive-definiteness across intermediate correlation BiMap/ReEig layers and outputs strictly lower-triangular isometric features of dimension d(d-1)/2.

```json
{
  "tril_dim": 28,
  "expected_tril_dim": 28,
  "output_shape": [
    4,
    2
  ],
  "passed": true,
  "scope": "full"
}
```

### `chen2026cornet_c2` (Signal)
> CorNet forward and backward passes remain invariant to arbitrary positive diagonal congruence D S D on input covariance matrices (max logit difference <1e-4).

```json
{
  "max_logit_diff_under_diagonal_scaling": 2.086162567138672e-07,
  "passed": true,
  "scope": "full"
}
```

### `chen2026cornet_c3` (Analytical)
> CorNet converges on multi-class oscillatory EEG classification under electrode impedance / amplitude scaling shifts with test accuracy >0.75.

```json
{
  "test_acc": 0.949999988079071,
  "passed": true,
  "scope": "toy"
}
```
