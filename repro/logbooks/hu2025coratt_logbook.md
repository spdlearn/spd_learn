# Trackio Reproduction Logbook: CorAtt: Correlation Manifold Self-Attention (IJCAI 2025) (`hu2025coratt`)

- **Command**: `python -m repro.trackio_runner run --paper-id hu2025coratt`
- **Git Commit**: `b3cf097`
- **Exit Code**: `0`
- **Duration**: `2.328s`

## Claim Execution Traces

### `hu2025coratt_c1` (Structural)
> Both Off-Log Metric (OLM) and Log-Scaled Metric (LSM) maps bijectively map full-rank correlation matrices to hollow symmetric matrices (zero diagonal) and invert back to unit-diagonal correlation matrices (<1e-4 max diagonal error).

```json
{
  "hollow_max_error": 0.0,
  "unit_diag_max_error": 0.0,
  "olm_roundtrip_rel_error": 3.2900991717626144e-07,
  "passed": true,
  "scope": "full"
}
```

### `hu2025coratt_c2` (Signal)
> Under channel-wise amplitude scaling artifacts (multiplying EEG channels by random gains in [0.1, 10.0]), cov_to_corr + CorrAttention produces strictly scale-invariant representations (relative drift <1e-4) whereas covariance-based AttentionManifold drifts significantly.

```json
{
  "corr_attention_scale_drift": 3.625081603786384e-07,
  "cov_attention_scale_drift": 10.430442810058594,
  "passed": true,
  "scope": "full"
}
```

### `hu2025coratt_c3` (Analytical)
> CorAtt trained end-to-end on EEG trials with random per-trial channel gain perturbations outperforms covariance-based MAtt in out-of-sample test accuracy.

```json
{
  "coratt_test_acc": 1.0,
  "matt_test_acc": 0.949999988079071,
  "passed": true,
  "scope": "toy"
}
```
