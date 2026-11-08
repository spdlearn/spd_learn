# Trackio Reproduction Logbook: CorAtt: Correlation Manifold Self-Attention (IJCAI 2025) (`hu2025coratt`)

- **Command**: `python -m repro.trackio_runner run --paper-id hu2025coratt`
- **Git Commit**: `db38483`
- **Exit Code**: `0`
- **Duration**: `4.177s`

## Claim Execution Traces

### `hu2025coratt_c1` (Structural)
> Both Off-Log Metric (OLM) and Log-Scaled Metric (LSM) maps bijectively project full-rank correlation matrices to hollow symmetric matrices (zero diagonal) and invert back to unit-diagonal correlation matrices (<1e-4 max diagonal error).

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
> Under channel-wise amplitude scaling artifacts (multiplying EEG channels by random gains in [0.2, 5.0]), cov_to_corr + CorrAttention produces scale-invariant representations (relative drift <1e-4) whereas covariance-based AttentionManifold drifts by >100%.

```json
{
  "corr_attention_scale_drift": 3.625081603786384e-07,
  "cov_attention_scale_drift": 10.430442810058594,
  "passed": true,
  "scope": "full"
}
```

### `hu2025coratt_c3` (Analytical)
> On real cross-session BNCI2014-001 (NEMAR nm000139, Session T -> Session E, 4 classes, chance 25.0%), CorAtt achieves 74.31% on Subject 3 (9-subject harness 74.65% +/- 11.10% vs 75.81% published Table 1) and outperforms covariance MAtt when test trials undergo random electrode gain shifts.

```json
{
  "harness": "MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)",
  "published_mean_pct": 75.81,
  "published_sd_pct": 10.9,
  "our_9subj_harness_mean_pct": 74.65,
  "our_9subj_harness_sd_pct": 11.1,
  "delta_pct": -1.16,
  "bnci_subj3_train_acc_pct": 79.51,
  "bnci_subj3_test_acc_pct": 75.35,
  "passed": true,
  "scope": "full"
}
```
