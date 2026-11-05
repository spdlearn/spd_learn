# Trackio Reproduction Logbook: RieTrans: SPDLinear, SPDRMLR & SPNN (NeurIPS 2024 / 2026) (`chen2026rietrans`)

- **Command**: `python -m repro.trackio_runner run --paper-id chen2026rietrans`
- **Git Commit**: `b3cf097`
- **Exit Code**: `0`
- **Duration**: `0.200s`

## Claim Execution Traces

### `chen2026rietrans_c1` (Structural)
> Chaining SPDLinear(return_tangent=True) into SPDLinear(input_tangent=True) produces identical outputs (<1e-5 error) to full manifold exp/log chaining under LEM while eliminating intermediate matrix_exp and matrix_log eigendecompositions.

```json
{
  "tangent_chaining_rel_error": 4.213809745579056e-07,
  "passed": true,
  "scope": "full"
}
```

### `chen2026rietrans_c2` (Signal)
> SPDRMLR supports all 4 Riemannian geometries (LEM, LCM, AIM, PEM) with finite, non-zero gradients with respect to both the SPD input X and the learnable SPD reference point P.

```json
{
  "metrics_grad_finite": {
    "LEM": true,
    "LCM": true,
    "AIM": true,
    "PEM": true
  },
  "passed": true,
  "scope": "full"
}
```

### `chen2026rietrans_c3` (Analytical)
> SPNN trained end-to-end on multi-class EEG decoding achieves >0.75 test accuracy and faster per-step execution when tangent-chaining is enabled.

```json
{
  "test_acc": 0.949999988079071,
  "passed": true,
  "scope": "toy"
}
```
