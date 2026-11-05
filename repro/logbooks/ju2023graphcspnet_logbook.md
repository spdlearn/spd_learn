# Trackio Reproduction Logbook: Graph-CSPNet & GraphBiMap (IEEE TNNLS 2023) (`ju2023graphcspnet`)

- **Command**: `python -m repro.trackio_runner run --paper-id ju2023graphcspnet`
- **Git Commit**: `b3cf097`
- **Exit Code**: `0`
- **Duration**: `0.279s`

## Claim Execution Traces

### `ju2023graphcspnet_c1` (Structural)
> GraphBiMap's row-stochastic graph propagation matrix P mixes node SPD matrices via convex combinations that strictly preserve positive definiteness before and after node-wise BiMap.

```json
{
  "min_eig_output": 0.6969856023788452,
  "output_shape": [
    4,
    6,
    5,
    5
  ],
  "passed": true,
  "scope": "full"
}
```

### `ju2023graphcspnet_c2` (Signal)
> When discriminative covariance structure shifts across adjacent time-frequency windows, GraphBiMap with neighborhood mixing outperforms isolated identity graph (P=I) in linear separability of tangent features.

```json
{
  "variance_unmixed": 0.01149129681289196,
  "variance_graph_mixed": 0.0031361300498247147,
  "passed": true,
  "scope": "full"
}
```

### `ju2023graphcspnet_c3` (Analytical)
> GraphCSPNet decodes both 3D raw EEG and 4D filter-bank EEG inputs end-to-end, achieving >0.75 test accuracy on time-frequency non-stationary EEG data.

```json
{
  "test_acc": 1.0,
  "passed": true,
  "scope": "toy"
}
```
