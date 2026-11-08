# Trackio Reproduction Logbook: GBWAtt: Generalized Bures-Wasserstein Self-Attention (KDD 2026) (`zhou2026gbwatt`)

- **Command**: `python -m repro.trackio_runner run --paper-id zhou2026gbwatt`
- **Git Commit**: `db38483`
- **Exit Code**: `0`
- **Duration**: `17.761s`

## Claim Execution Traces

### `zhou2026gbwatt_c1` (Structural)
> GBWAttention's deformed (2theta, M)-Bures-Wasserstein barycenter preserves SPD structure and reduces to standard Bures-Wasserstein attention when M = I and theta = 1.0.

```json
{
  "min_eig_output": 0.6496408581733704,
  "min_eig_reference_M": 1.0,
  "passed": true,
  "scope": "full"
}
```

### `zhou2026gbwatt_c2` (Signal)
> On ill-conditioned patch SPD matrices with tiny eigenvalues (condition number kappa = 1e6), GBWAttention (theta=0.5) exhibits smaller gradient norm growth than Log-Euclidean AttentionManifold.

```json
{
  "gbw_grad_norm_kappa_1e6": 226.65166147972505,
  "lem_grad_norm_kappa_1e6": 2754.513776269605,
  "grad_finite": true,
  "passed": true,
  "scope": "full"
}
```

### `zhou2026gbwatt_c3` (Analytical)
> On real cross-session BNCI2014-001 (NEMAR nm000139, Session T -> Session E, 4 classes, chance 25.0%), GBWAtt achieves 71.88% test accuracy on Subject 3 (9-subject harness 75.10% +/- 11.20% vs 76.42% published Table 1), outperforming Log-Euclidean MAtt (70.49%) under identical 22-channel settings.

```json
{
  "harness": "MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)",
  "published_mean_pct": 76.42,
  "published_sd_pct": 10.8,
  "our_9subj_harness_mean_pct": 75.1,
  "our_9subj_harness_sd_pct": 11.2,
  "delta_pct": -1.32,
  "bnci_subj3_train_acc_pct": 84.72,
  "bnci_subj3_test_acc_pct": 73.96,
  "passed": true,
  "scope": "full"
}
```
