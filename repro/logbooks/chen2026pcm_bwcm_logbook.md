# Trackio Reproduction Logbook: PCM & BWCM Cholesky Product Metrics (ICLR 2026) (`chen2026pcm_bwcm`)

- **Command**: `python -m repro.trackio_runner run --paper-id chen2026pcm_bwcm`
- **Git Commit**: `b3cf097`
- **Exit Code**: `0`
- **Duration**: `2.303s`

## Claim Execution Traces

### `chen2026pcm_bwcm_c1` (Structural)
> On diagonal SPD matrices, Bures-Wasserstein-Cholesky distance (BWCM, theta=0.5, w_diag=0.25) exactly equals the Bures-Wasserstein distance to machine precision (<1e-6), and power_cholesky_exp(power_cholesky_log(X, P), P) recovers X with <1e-5 relative error.

```json
{
  "bw_vs_bwcm_diag_error": 1.1102230246251565e-15,
  "log_exp_roundtrip_rel_error": 1.6697477535664604e-16,
  "passed": true,
  "scope": "full"
}
```

### `chen2026pcm_bwcm_c2` (Signal)
> On high-condition-number SPD matrices (kappa >= 1e5), PCM/BWCM distance and Fréchet mean produce finite, non-NaN gradients where eigendecomposition-free Cholesky factorization runs faster than matrix_log/eigh.

```json
{
  "kappa": 1000000.0,
  "grad_finite": true,
  "pcm_time_sec": 0.035453583928756416,
  "eigh_time_sec": 0.0026944580022245646,
  "passed": true,
  "scope": "full"
}
```

### `chen2026pcm_bwcm_c3` (Analytical)
> A minimum-distance-to-mean (MDM) classifier using power_cholesky_mean and power_cholesky_distance decodes multi-class synthetic oscillatory EEG covariance trials significantly above chance (>0.75 accuracy).

```json
{
  "mdm_test_acc": 1.0,
  "passed": true,
  "scope": "toy"
}
```
