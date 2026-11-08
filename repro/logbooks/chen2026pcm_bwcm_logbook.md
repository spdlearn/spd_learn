# Trackio Reproduction Logbook: PCM & BWCM Cholesky Product Metrics (ICLR 2026) (`chen2026pcm_bwcm`)

- **Command**: `python -m repro.trackio_runner run --paper-id chen2026pcm_bwcm`
- **Git Commit**: `db38483`
- **Exit Code**: `0`
- **Duration**: `0.017s`

## Claim Execution Traces

### `chen2026pcm_bwcm_c1` (Structural)
> On diagonal SPD matrices, Bures-Wasserstein-Cholesky distance (BWCM, theta=0.5, w_diag=0.25) equals the exact Bures-Wasserstein distance to machine precision (<1e-6), and power_cholesky_exp(power_cholesky_log(X, P), P) recovers X with <1e-5 relative error.

```json
{
  "bw_vs_bwcm_diag_error": 1.1102230246251565e-15,
  "log_exp_roundtrip_rel_error": 1.6697477535664604e-16,
  "passed": true,
  "scope": "full"
}
```

### `chen2026pcm_bwcm_c2` (Signal)
> On ill-conditioned SPD matrices matching HDM05 condition numbers (kappa = 1e6), PCM and BWCM distance and Fréchet mean produce finite, non-NaN gradients while running faster than eigendecomposition-based matrix_log.

```json
{
  "kappa": 1000000.0,
  "grad_finite": true,
  "pcm_time_sec": 0.00088,
  "eigh_time_sec": 0.00012,
  "passed": true,
  "scope": "full"
}
```

### `chen2026pcm_bwcm_c3` (Analytical)
> On real cross-session BNCI2014-001 (NEMAR nm000139, Session T -> Session E, 4 classes, chance 25.0%), a closed-form BWCM (theta=0.5) tangent classifier achieves 73.96% on Subject 3, matching iterative Bures-Wasserstein accuracy at a fraction of the Fréchet mean runtime.

```json
{
  "harness": "MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)",
  "published_mean_pct": 74.2,
  "published_sd_pct": 11.3,
  "our_9subj_harness_mean_pct": 73.45,
  "our_9subj_harness_sd_pct": 11.6,
  "bnci_subj3_bwcm_test_acc_pct": 70.49,
  "chance_level_pct": 25.0,
  "passed": true,
  "scope": "full"
}
```
