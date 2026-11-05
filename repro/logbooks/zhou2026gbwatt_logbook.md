# Trackio Reproduction Logbook: GBWAtt: Generalized Bures-Wasserstein Self-Attention (KDD 2026) (`zhou2026gbwatt`)

- **Command**: `python -m repro.trackio_runner run --paper-id zhou2026gbwatt`
- **Git Commit**: `b3cf097`
- **Exit Code**: `0`
- **Duration**: `0.935s`

## Claim Execution Traces

### `zhou2026gbwatt_c1` (Structural)
> GBWAttention's deformed (2theta, M)-Bures-Wasserstein barycenter preserves SPD structure and reduces to standard Bures-Wasserstein attention when M=I and theta=1.0.

```json
{
  "min_eig_output": 0.6079550981521606,
  "min_eig_reference_M": 1.0,
  "passed": true,
  "scope": "full"
}
```

### `zhou2026gbwatt_c2` (Signal)
> On ill-conditioned patch SPD matrices with tiny eigenvalues (lambda_min = 1e-6), GBWAttention (theta=0.5) exhibits significantly smaller gradient norm variance and higher SNR stability than Log-Euclidean AttentionManifold.

```json
{
  "gbw_grad_norm_kappa_1e6": 975.8703002929688,
  "lem_grad_norm_kappa_1e6": 2382.6689453125,
  "grad_finite": true,
  "passed": true,
  "scope": "full"
}
```

### `zhou2026gbwatt_c3` (Analytical)
> GBWAtt trained on noisy/ill-conditioned multi-patch EEG trials achieves higher test accuracy than standard Log-Euclidean MAtt.

```json
{
  "test_acc": 1.0,
  "passed": true,
  "scope": "toy"
}
```
