# Trackio Reproduction Logbook: GBWBN (CVPR 2025) & CBN (IEEE TNNLS 2025) SPD BatchNorms (`zhou2025gbwbn_cbn`)

- **Command**: `python -m repro.trackio_runner run --paper-id zhou2025gbwbn_cbn`
- **Git Commit**: `b3cf097`
- **Exit Code**: `0`
- **Duration**: `3.239s`

## Claim Execution Traces

### `zhou2025gbwbn_cbn_c1` (Structural)
> SPDBatchNormCholesky normalizes batch mean to 0 and variance to 1 in Log-Cholesky coordinates with zero eigendecompositions, and SPDBatchNormBW centers batch Bures-Wasserstein mean to identity when rebias=False.

```json
{
  "cholesky_tangent_mean_max_err": 6.476300976980079e-16,
  "cholesky_tangent_var_max_err": 4.060877515676964e-05,
  "passed": true,
  "scope": "full"
}
```

### `zhou2025gbwbn_cbn_c2` (Signal)
> On ill-conditioned SPD batches (condition number kappa = 1e6), SPDBatchNormBW and SPDBatchNormCholesky maintain finite forward outputs and gradients without NaN/Inf.

```json
{
  "kappa_1e6_bwbn_grad_finite": true,
  "passed": true,
  "scope": "full"
}
```

### `zhou2025gbwbn_cbn_c3` (Analytical)
> Adding SPDBatchNormBW or SPDBatchNormCholesky to SPDNet accelerates training loss convergence compared to unnormalized SPDNet on multi-class EEG covariance classification.

```json
{
  "test_acc": 0.949999988079071,
  "passed": true,
  "scope": "toy"
}
```
