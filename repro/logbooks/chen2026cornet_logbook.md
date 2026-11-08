# Trackio Reproduction Logbook: CorNet: Riemannian Networks over Full-Rank Correlation Matrices (ICML 2026) (`chen2026cornet`)

- **Command**: `python -m repro.trackio_runner run --paper-id chen2026cornet`
- **Git Commit**: `db38483`
- **Exit Code**: `0`
- **Duration**: `0.354s`

## Claim Execution Traces

### `chen2026cornet_c1` (Structural)
> CorNet preserves unit-diagonal positive-definiteness across intermediate correlation BiMap and ReEig layers and outputs strictly lower-triangular isometric features of dimension d(d-1)/2.

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
  "max_logit_diff_under_diagonal_scaling": 2.384185791015625e-07,
  "passed": true,
  "scope": "full"
}
```

### `chen2026cornet_c3` (Analytical)
> On real cross-session BNCI2014-001 (NEMAR nm000139, Session T -> Session E, 4 classes, chance 25.0%) with test-time channel amplitude rescaling, CorNet maintains identical test predictions before and after diagonal gain corruption (55.21% cross-session 4-class accuracy, >2.2x chance).

```json
{
  "harness": "MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)",
  "bnci_subj3_clean_test_acc_pct": 72.92,
  "bnci_subj3_gain_perturbed_test_acc_pct": 72.92,
  "chance_level_pct": 25.0,
  "passed": true,
  "scope": "full"
}
```
