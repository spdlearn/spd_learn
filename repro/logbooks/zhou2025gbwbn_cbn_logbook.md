# Trackio Reproduction Logbook: GBWBN & CBN: Bures-Wasserstein & Cholesky BatchNorm (CVPR / IEEE TNNLS 2025) (`zhou2025gbwbn_cbn`)

- **Command**: `python -m repro.trackio_runner run --paper-id zhou2025gbwbn_cbn`
- **Git Commit**: `3a9e18d`
- **Exit Code**: `0`
- **Duration**: `0.717s`

## Claim Execution Traces

### `zhou2025gbwbn_cbn_c1` (Structural)
> SPDBatchNormCholesky (CBN) standardizes the Log-Cholesky tangent representation to zero mean (<1e-5 error) and unit variance (<1e-3 error) when rebias=False without a single eigendecomposition.

```json
{
  "cholesky_tangent_mean_max_err": 6.476300976980079e-16,
  "cholesky_tangent_var_max_err": 4.060877515676964e-05,
  "passed": true,
  "scope": "full"
}
```

### `zhou2025gbwbn_cbn_c2` (Signal)
> On ill-conditioned SPD batches matching HDM05 condition numbers (kappa = 1e6), both SPDBatchNormBW and SPDBatchNormCholesky produce finite, non-NaN outputs and backward gradients.

```json
{
  "kappa_1e6_bwbn_grad_finite": true,
  "passed": true,
  "scope": "full"
}
```

### `zhou2025gbwbn_cbn_c3` (Analytical)
> On real cross-session BNCI2014-001 (NEMAR nm000139, Session T -> Session E, 4 classes, chance 25.0%), SPDNet equipped with SPDBatchNormBW and SPDBatchNormCholesky achieves 74.31% and 73.61% test accuracy on Subject 3, matching AIRM batch normalization without iterative Karcher flow.

```json
{
  "harness": "MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)",
  "bnci_subj3_gbwbn_test_acc_pct": 69.44,
  "bnci_subj3_cbn_test_acc_pct": 66.67,
  "passed": false,
  "scope": "full"
}
```
