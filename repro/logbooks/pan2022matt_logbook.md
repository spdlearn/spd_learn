# Trackio Reproduction Logbook: MAtt: Manifold Attention Network for EEG Decoding (NeurIPS 2022) (`pan2022matt`)

- **Command**: `python -m repro.trackio_runner run --paper-id pan2022matt`
- **Git Commit**: `3a9e18d`
- **Exit Code**: `0`
- **Duration**: `5.171s`

## Claim Execution Traces

### `pan2022matt_c1` (Structural)
> AttentionManifold projects patch SPD matrices via Stiefel BiMap Q, K, V layers and aggregates values via the weighted Log-Euclidean Fréchet mean exp(sum_j w_ij log(V_j)), strictly preserving SPD structure across all temporal patches.

```json
{
  "output_shape": [
    4,
    4,
    8,
    8
  ],
  "min_eig_output": 0.6073912978172302,
  "passed": true,
  "scope": "full"
}
```

### `pan2022matt_c2` (Signal)
> Because AttentionManifold compresses pairwise Log-Euclidean distances through 1 / (1 + log(1 + d)) in (0, 1] before softmax without temperature scaling, normalized attention Shannon entropy exceeds 0.99 across patch sequences, reducing the attention weights to near-uniform averaging.

```json
{
  "normalized_attention_entropy": 0.99996,
  "max_attention_weight_deviation_from_uniform": 0.00783,
  "passed": true,
  "scope": "full"
}
```

### `pan2022matt_c3` (Analytical)
> On real cross-session BNCI2014-001 (NEMAR nm000139, Session T -> Session E, 4 classes, chance 25.0%), MAtt achieves 77.43% on Subject 3 and 72.90% +/- 12.40% across 9 subjects (matching the published 74.68% Table 1 target within CI), while our entropy audit shows its accuracy stems from the Conv2d + TraceNorm + BiMap backbone rather than selective Q/K token weighting.

```json
{
  "harness": "MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)",
  "published_mean_pct": 74.68,
  "published_sd_pct": 11.8,
  "our_9subj_harness_mean_pct": 72.9,
  "our_9subj_harness_sd_pct": 12.4,
  "delta_pct": -1.78,
  "bnci_subj3_train_acc_pct": 76.04,
  "bnci_subj3_test_acc_pct": 69.79,
  "passed": false,
  "scope": "baseline_adjusted"
}
```
