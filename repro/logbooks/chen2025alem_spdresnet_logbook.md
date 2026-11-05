# Trackio Reproduction Logbook: ALEM & SPDResNet (IEEE TIP 2024 / ICLR 2025 / NeurIPS 2026) (`chen2025alem_spdresnet`)

- **Command**: `python -m repro.trackio_runner run --paper-id chen2025alem_spdresnet`
- **Git Commit**: `b3cf097`
- **Exit Code**: `0`
- **Duration**: `0.537s`

## Claim Execution Traces

### `chen2025alem_spdresnet_c1` (Structural)
> AdaptiveLogEuclidean converges to the standard matrix logarithm as power p -> 0 (with w=1, b=0), and SPDResBlock strictly preserves SPD structure via Log-Euclidean geodesic residual addition.

```json
{
  "alem_spd_min_eig": 0.2676447710635149,
  "spdresblock_min_eig": 0.2223249077796936,
  "passed": true,
  "scope": "full"
}
```

### `chen2025alem_spdresnet_c2` (Signal)
> Deep SPD networks with SPDResBlock residual connections maintain non-vanishing gradient norms in early BiMap layers as network depth increases, compared to plain stacked BiMap+ReEig blocks.

```json
{
  "input_grad_norm_2_blocks": 19.42245864868164,
  "passed": true,
  "scope": "full"
}
```

### `chen2025alem_spdresnet_c3` (Analytical)
> SPDResNet decodes both raw EEG and covariance inputs with >0.75 test accuracy.

```json
{
  "test_acc": 1.0,
  "passed": true,
  "scope": "toy"
}
```
