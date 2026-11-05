# Trackio Reproduction Logbook: RHOP: Riemannian High-Order Pooling for BFMs (ICLR 2026) (`hu2026rhop`)

- **Command**: `python -m repro.trackio_runner run --paper-id hu2026rhop`
- **Git Commit**: `3a9e18d`
- **Exit Code**: `0`
- **Duration**: `0.527s`

## Claim Execution Traces

### `hu2026rhop_c1` (Structural)
> GaussianSPDDescriptor embeds token mean mu in R^d and covariance Sigma in S++^d into a symmetric positive definite (d+1)x(d+1) matrix across all batch sizes and token lengths T >= 1.

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
> When two token sequences have identical first-order token mean (mu_1 = mu_2 = 0) but distinct inter-feature covariance (Sigma_1 != Sigma_2), standard mean pooling fails at chance (50.0% accuracy) whereas RHOP separates them with >= 95.0% accuracy.

```json
{
  "mean_pool_feature_norm": 4.2305705960643536e-07,
  "rhop_zero_mean_cov_test_acc_pct": 100.0,
  "passed": true,
  "scope": "full"
}
```

### `hu2026rhop_c3` (Analytical)
> On real cross-session BNCI2014-001 (NEMAR nm000139, Session T -> Session E, 4 classes, chance 25.0%) where bandpass-filtered EEG tokens have near-zero temporal mean, RHOP (PEM, power=0.5) achieves 71.18% test accuracy on Subject 3 whereas first-order temporal mean pooling collapses to 27.08% (near 25.0% chance).

```json
{
  "harness": "MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)",
  "bnci_subj3_rhop_test_acc_pct": 63.89,
  "bnci_subj3_mean_pool_test_acc_pct": 23.96,
  "second_order_gain_over_mean_pool_pp": 39.93,
  "passed": false,
  "scope": "full"
}
```
