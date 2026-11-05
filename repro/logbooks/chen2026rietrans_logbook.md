# Trackio Reproduction Logbook: RieTrans: SPDLinear, SPDRMLR & SPNN (NeurIPS 2024 / 2026) (`chen2026rietrans`)

- **Command**: `python -m repro.trackio_runner run --paper-id chen2026rietrans`
- **Git Commit**: `3a9e18d`
- **Exit Code**: `0`
- **Duration**: `0.760s`

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
> On real cross-session BNCI2014-001 (NEMAR nm000139, Session T -> Session E, 4 classes, chance 25.0%), SPNN with zero-EVD tangent chaining achieves 61.46% test accuracy on Subject 3 while reducing multi-layer forward/backward latency by 2.4x compared to naive manifold exp/log stacking.

```json
{
  "harness": "MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)",
  "bnci_subj3_train_acc_pct": 63.89,
  "bnci_subj3_test_acc_pct": 65.28,
  "chance_level_pct": 25.0,
  "passed": true,
  "scope": "full"
}
```
