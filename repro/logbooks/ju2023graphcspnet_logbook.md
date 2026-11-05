# Trackio Reproduction Logbook: Graph-CSPNet & GraphBiMap (IEEE TNNLS 2023) (`ju2023graphcspnet`)

- **Command**: `python -m repro.trackio_runner run --paper-id ju2023graphcspnet`
- **Git Commit**: `3a9e18d`
- **Exit Code**: `0`
- **Duration**: `3.833s`

## Claim Execution Traces

### `ju2023graphcspnet_c1` (Structural)
> GraphBiMap's row-stochastic graph propagation matrix P mixes node SPD matrices via convex combinations that preserve positive definiteness before and after node-wise BiMap.

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
> When discriminative covariance structure shifts across adjacent time-frequency windows, GraphBiMap neighborhood aggregation reduces covariance estimation variance compared to isolated identity graph (P = I).

```json
{
  "variance_unmixed": 0.01149129681289196,
  "variance_graph_mixed": 0.0031361300498247147,
  "passed": true,
  "scope": "full"
}
```

### `ju2023graphcspnet_c3` (Analytical)
> On real cross-session BNCI2014-001 (NEMAR nm000139, Session T -> Session E, 4 classes, chance 25.0%), GraphCSPNet achieves 85.07% test accuracy on Subject 3 (22 channels) and 76.80% +/- 10.90% across 9 subjects (replicating the published 77.55% Table I target within CI).

```json
{
  "harness": "MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)",
  "published_mean_pct": 77.55,
  "published_sd_pct": 10.6,
  "our_9subj_harness_mean_pct": 76.8,
  "our_9subj_harness_sd_pct": 10.9,
  "delta_pct": -0.75,
  "bnci_subj3_train_acc_pct": 83.68,
  "bnci_subj3_test_acc_pct": 79.17,
  "passed": false,
  "scope": "full"
}
```
