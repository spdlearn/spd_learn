# Trackio Reproduction Logbook: STaRNet (Neural Networks 2024, Issue #32) (`wang2024starnet`)

- **Command**: `python -m repro.trackio_runner run --paper-id wang2024starnet`
- **Git Commit**: `b3cf097`
- **Exit Code**: `0`
- **Duration**: `4.172s`

## Claim Execution Traces

### `wang2024starnet_c1` (Structural)
> STaRNet's dual-branch architecture fuses depthwise patch-level Riemannian tangent vectors with Squeeze-and-Excitation Euclidean temporal features into a unified representation.

```json
{
  "output_shape": [
    4,
    2
  ],
  "passed": true,
  "scope": "full"
}
```

### `wang2024starnet_c2` (Signal)
> When discriminative information is split between spatial covariance (branch 1) and temporal envelope modulation (branch 2), STaRNet's dual-branch fusion outperforms either branch zeroed out in isolation.

```json
{
  "riemannian_bimap_grad_norm": 1.5506951808929443,
  "euclidean_se_grad_sum": 114.43330578505993,
  "passed": true,
  "scope": "full"
}
```

### `wang2024starnet_c3` (Analytical)
> STaRNet achieves >0.75 test accuracy on synthetic motor imagery EEG classification.

```json
{
  "test_acc": 1.0,
  "passed": true,
  "scope": "toy"
}
```
