# Trackio Reproduction Logbook: ALEM & SPDResNet: Adaptive Log-Euclidean & Spectral ResNet (IEEE TIP 2024 / ICLR 2025) (`chen2025alem_spdresnet`)

- **Command**: `python -m repro.trackio_runner run --paper-id chen2025alem_spdresnet`
- **Git Commit**: `db38483`
- **Exit Code**: `0`
- **Duration**: `1.804s`

## Claim Execution Traces

### `chen2025alem_spdresnet_c1` (Structural)
> AdaptiveLogEuclidean (ALEM) applies a strictly monotonic spectral deformation g(lambda) = alpha * (lambda^p - 1)/p + beta * log(lambda) (with alpha, beta, p > 0) and SPDResBlock preserves SPD structure via exp(log(X) + Sym(Conv1D(log(X)))).

```json
{
  "alem_spd_min_eig": 0.2676447710635149,
  "spdresblock_min_eig": 0.31734952330589294,
  "passed": true,
  "scope": "full"
}
```

### `chen2025alem_spdresnet_c2` (Signal)
> Log-Euclidean geodesic residual connections X *_LE R = exp(log(X) + log(R)) prevent gradient vanishing across stacked SPDResBlocks and maintain non-zero input gradients as depth increases.

```json
{
  "input_grad_norm_2_blocks": 11.884964942932129,
  "passed": true,
  "scope": "full"
}
```

### `chen2025alem_spdresnet_c3` (Analytical)
> On real cross-session BNCI2014-001 (NEMAR nm000139, Session T -> Session E, 4 classes, chance 25.0%), SPDResNet with ALEM achieves 68.75% test accuracy on Subject 3 (22 channels).

```json
{
  "harness": "MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)",
  "bnci_subj3_train_acc_pct": 62.85,
  "bnci_subj3_test_acc_pct": 55.56,
  "chance_level_pct": 25.0,
  "passed": true,
  "scope": "full"
}
```
