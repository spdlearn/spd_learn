# Trackio Reproduction Logbook: RHOP: Riemannian High-Order Pooling for BFMs (ICLR 2026) (`hu2026rhop`)

- **Command**: `python -m repro.trackio_runner run --paper-id hu2026rhop`
- **Git Commit**: `b3cf097`
- **Exit Code**: `0`
- **Duration**: `0.148s`

## Claim Execution Traces

### `hu2026rhop_c1` (Structural)
> GaussianSPDDescriptor embeds token mean mu in R^d and covariance Sigma in S++^d into a strictly symmetric positive definite (d+1)x(d+1) matrix across all batch sizes and token lengths T >= 1.

```json
{
  "descriptor_shape": [
    8,
    7,
    7
  ],
  "min_eig": 0.09966684877872467,
  "passed": true,
  "scope": "full"
}
```

### `hu2026rhop_c2` (Signal)
> When two token sequences have identical first-order token mean (mu_1 = mu_2 = 0) but distinct inter-feature covariance (Sigma_1 != Sigma_2), standard mean pooling fails completely (0.50 accuracy) whereas RHOP separates them with >0.95 accuracy.

```json
{
  "mean_pool_feature_norm": 4.2305705960643536e-07,
  "rhop_test_acc_zero_mean_cov_task": 1.0,
  "passed": true,
  "scope": "full"
}
```

### `hu2026rhop_c3` (Analytical)
> Attaching RiemannianHighOrderPooling to a Transformer encoder on EEG token sequences improves classification accuracy over mean-pooling / CLS-only pooling.

```json
{
  "eeg_token_test_acc": 0.949999988079071,
  "passed": true,
  "scope": "toy"
}
```
