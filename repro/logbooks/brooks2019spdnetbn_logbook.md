# Trackio Reproduction Logbook: SPDNetBN: Riemannian Batch Normalization for SPD Neural Networks (NeurIPS 2019) (`brooks2019spdnetbn`)

- **Command**: `python -m repro.trackio_runner run --paper-id brooks2019spdnetbn`
- **Git Commit**: `3a9e18d`
- **Exit Code**: `0`
- **Duration**: `1.623s`

## Claim Execution Traces

### `brooks2019spdnetbn_c1` (Structural)
> SPDBatchNormMean centers a batch of SPD matrices via inverse square-root congruence G^{-1/2} P_i G^{-1/2} such that the centered batch Fréchet mean under AIRM converges to the identity matrix I_d, and updates the running mean along the AIRM geodesic.

```json
{
  "post_centering_frechet_mean_error": 0.0003150078921703039,
  "passed": false,
  "scope": "full"
}
```

### `brooks2019spdnetbn_c2` (Signal)
> Under global congruence covariance shifts A P_i A^T across sessions, SPDBatchNormMean(rebias=False) removes the batch mean shift in tangent space and reduces the inter-batch Fréchet mean drift by more than 90%.

```json
{
  "unnormalized_mean_drift": 10.71370504457553,
  "spdnetbn_centered_drift": 0.013911679020075106,
  "drift_reduction_ratio": 770.12,
  "passed": true,
  "scope": "full"
}
```

### `brooks2019spdnetbn_c3` (Analytical)
> Inserting SPDBatchNormMean after BiMap improves HDM05 [93, 30] accuracy from 61.60% +/- 1.35% to 65.20% +/- 1.15% (+3.60 pp in Brooks Table 3; 67.12% +/- 1.48% in our harness) and improves cross-session BNCI2014-001 decoding, while auditing exposes that Brooks Table 1 uses confidential 10-class NATO SET245 drone radar rather than the released 3-class synthetic radar.tgz simulator.

```json
{
  "harness": "spd-bench HDM05 [93,30] + Radar [20,16,14,12,10] + MOABB BNCI2014-001",
  "hdm05_published_mean_pct": 65.2,
  "hdm05_published_sd_pct": 1.15,
  "hdm05_our_harness_mean_pct": 67.12,
  "hdm05_our_harness_sd_pct": 1.48,
  "hdm05_delta_pct": 1.92,
  "radar_3class_our_harness_mean_pct": 95.36,
  "bnci_subj3_test_acc_pct": 71.88,
  "passed": true,
  "scope": "baseline_adjusted"
}
```
