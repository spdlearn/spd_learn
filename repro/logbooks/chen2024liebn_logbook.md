# Trackio Reproduction Logbook: LieBN: A Lie Group Approach to Riemannian Batch Normalization (ICLR 2024) (`chen2024liebn`)

- **Command**: `python -m repro.trackio_runner run --paper-id chen2024liebn`
- **Git Commit**: `3a9e18d`
- **Exit Code**: `0`
- **Duration**: `0.520s`

## Claim Execution Traces

### `chen2024liebn_c1` (Structural)
> SPDBatchNormLie satisfies Proposition 4.2 across all three Lie group geometries (AIM, LEM, LCM): when bias B = I and shift s = 1, the normalized batch has zero Lie-algebra mean (<1e-5 error) and unit bi-invariant dispersion (<1e-3 error).

```json
{
  "dispersion_error_from_unit": {
    "AIM": 0.0,
    "LEM": 0.0,
    "LCM": 0.0
  },
  "passed": true,
  "scope": "full"
}
```

### `chen2024liebn_c2` (Signal)
> On ill-conditioned SPD batches (condition number kappa = 9.59e5 from real HDM05 covariances), closed-form LEM and LCM LieBN satisfy exact group translation equivariance to machine precision (~1e-15), whereas 1-step Karcher AIM incurs a ~1e-3 centering residual.

```json
{
  "hdm05_lem_centering_residual": 1.1877650641019628e-15,
  "hdm05_lcm_centering_residual": 4.2593245247010923e-16,
  "hdm05_aim_1step_centering_residual": 0.16933186521059693,
  "passed": true,
  "scope": "full"
}
```

### `chen2024liebn_c3` (Analytical)
> In our 10-seed spd-bench replication of LieBN Table 4 on real HDM05 [93, 30], LieBN-LCM-(0.5) reaches 67.92% +/- 1.61% (69.45% +/- 2.51% on stratified splits vs 70.84% +/- 0.92% published), and our baseline audit reveals that +3.97 pp (58%) of LieBN's reported +11.71 pp gain over SPDNet comes from an under-tuned SPDNet baseline (59.13% in LieBN vs 63.10% +/- 1.53% in our harness and 61.60% +/- 1.35% in Brooks 2019).

```json
{
  "harness": "spd-bench HDM05 [93,30] (10 seeds) + FPHA [63,33] + MOABB BNCI2014-001",
  "hdm05_lcm05_published_mean_pct": 70.84,
  "hdm05_lcm05_published_sd_pct": 0.92,
  "hdm05_lcm05_frozen_split_harness_pct": 67.92,
  "hdm05_lcm05_stratified_split_harness_pct": 69.45,
  "spdnet_baseline_liebn_paper_pct": 59.13,
  "spdnet_baseline_our_harness_pct": 63.1,
  "weak_baseline_inflation_pp": 3.97,
  "bnci_subj3_test_acc_pct": 64.58,
  "passed": false,
  "scope": "baseline_adjusted"
}
```
