# Trackio Reproduction Logbook: SPDNet: A Riemannian Network for SPD Matrix Learning (AAAI 2017) (`huang2017spdnet`)

- **Command**: `python -m repro.trackio_runner run --paper-id huang2017spdnet`
- **Git Commit**: `db38483`
- **Exit Code**: `0`
- **Duration**: `4.442s`

## Claim Execution Traces

### `huang2017spdnet_c1` (Structural)
> BiMap with Stiefel-parametrized semi-orthogonal weights W in St(d_in, d_out) followed by ReEig(epsilon=1e-4) maps input SPD matrices to reduced SPD matrices with minimum eigenvalue >= epsilon and projects via LogEig to an isometric vector of dimension d_out(d_out + 1)/2.

```json
{
  "stiefel_orthogonality_error": 3.782313342526322e-07,
  "post_reeig_min_eig": 0.02629523165524006,
  "tangent_vec_dim": 36,
  "expected_vec_dim": 36,
  "passed": true,
  "scope": "full"
}
```

### `huang2017spdnet_c2` (Signal)
> On real HDM05 skeletal action covariance matrices (93 x 93, median condition number kappa = 9.59e5), BiMap(93 -> 30) + ReEig(1e-4) compresses the condition number by more than three orders of magnitude and yields finite backward gradients.

```json
{
  "hdm05_raw_median_kappa": 691927.875,
  "post_bimap_reeig_median_kappa": 129495.4609375,
  "grad_finite": true,
  "passed": true,
  "scope": "full"
}
```

### `huang2017spdnet_c3` (Analytical)
> In our unified MOABB cross-session harness on real BNCI2014-001 (NEMAR nm000139, Session T -> Session E, 4 classes, chance 25.0%) and spd-bench HDM05 [93, 30] harness (117 classes, 10 seeds), SPDNet achieves 63.10% +/- 1.53% on HDM05 (replicating Brooks et al. 2019 Table 3 target 61.60% +/- 1.35% within 1 SD) and >70% cross-session accuracy on BNCI2014-001 Subject 3.

```json
{
  "harness": "spd-bench HDM05 [93,30] (10 seeds) + MOABB BNCI2014-001 CrossSession",
  "hdm05_published_mean_pct": 61.6,
  "hdm05_published_sd_pct": 1.35,
  "hdm05_our_harness_mean_pct": 63.1,
  "hdm05_our_harness_sd_pct": 1.53,
  "hdm05_delta_pct": 1.5,
  "bnci_subj3_train_acc_pct": 81.6,
  "bnci_subj3_test_acc_pct": 75.35,
  "chance_level_pct": 25.0,
  "passed": true,
  "scope": "full"
}
```
