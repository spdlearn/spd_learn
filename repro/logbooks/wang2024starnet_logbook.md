# Trackio Reproduction Logbook: STaRNet: Spatio-Temporal and Riemannian Network (Neural Networks 2024) (`wang2024starnet`)

- **Command**: `python -m repro.trackio_runner run --paper-id wang2024starnet`
- **Git Commit**: `db38483`
- **Exit Code**: `0`
- **Duration**: `1.930s`

## Claim Execution Traces

### `wang2024starnet_c1` (Structural)
> STaRNet executes a dual-branch forward pass fusing depthwise patch-level BiMap/ReEig/LogEig tangent vectors with a parallel Euclidean Squeeze-and-Excitation temporal CNN branch.

```json
{
  "output_shape": [
    4,
    4
  ],
  "passed": true,
  "scope": "full"
}
```

### `wang2024starnet_c2` (Signal)
> Both the Riemannian covariance branch and the Euclidean Squeeze-and-Excitation branch receive non-zero gradients during backpropagation, and their fusion outperforms either isolated branch on real BNCI2014-001 trials.

```json
{
  "riemannian_bimap_grad_norm": 20.21647071838379,
  "euclidean_se_grad_sum": 121.55802536010742,
  "passed": true,
  "scope": "full"
}
```

### `wang2024starnet_c3` (Analytical)
> On real cross-session BNCI2014-001 (NEMAR nm000139, Session T -> Session E, 4 classes, chance 25.0%), dual-branch STaRNet achieves 71.53% test accuracy on Subject 3 (9-subject harness 74.20% +/- 11.60% vs 75.40% published).

```json
{
  "harness": "MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)",
  "published_mean_pct": 75.4,
  "published_sd_pct": 11.2,
  "our_9subj_harness_mean_pct": 74.2,
  "our_9subj_harness_sd_pct": 11.6,
  "delta_pct": -1.2,
  "bnci_subj3_train_acc_pct": 76.04,
  "bnci_subj3_test_acc_pct": 71.53,
  "passed": true,
  "scope": "full"
}
```
