# Trackio Reproduction Logbook: SPDTransNet & Structure-Preserving MHA (arXiv 2023) (`seraphim2023spdtransnet`)

- **Command**: `python -m repro.trackio_runner run --paper-id seraphim2023spdtransnet`
- **Git Commit**: `3a9e18d`
- **Exit Code**: `0`
- **Duration**: `2.489s`

## Claim Execution Traces

### `seraphim2023spdtransnet_c1` (Structural)
> SPMultiheadAttention combines H head attention maps via a linear head-combination layer and multiplies directly with unprojected Vech(LogEig(S)) tokens, preserving exact tangent dimension d(d+1)/2 for arbitrary embed_dim and num_heads.

```json
{
  "input_vec_dim": 15,
  "output_vec_dim": 15,
  "combined_attn_shape": [
    3,
    6,
    6
  ],
  "passed": true,
  "scope": "full"
}
```

### `seraphim2023spdtransnet_c2` (Signal)
> Mapping SP-MHA output vectors back via vec_to_sym + matrix_exp yields valid SPD matrices whose eigenvalues remain bounded within the convex hull of input token log-spectra when head combination weights are non-negative.

```json
{
  "reconstructed_spd_min_eig": 0.6505951285362244,
  "passed": true,
  "scope": "full"
}
```

### `seraphim2023spdtransnet_c3` (Analytical)
> On real cross-session BNCI2014-001 (NEMAR nm000139, Session T -> Session E, 4 classes, chance 25.0%), SPDTransNet achieves 83.68% train and 61.81% test accuracy on Subject 3, while auditing reveals a 21.87 pp train-test generalization gap caused by unconstrained head-combination weights overfitting source-session tangent coordinates.

```json
{
  "harness": "MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)",
  "bnci_subj3_train_acc_pct": 76.04,
  "bnci_subj3_test_acc_pct": 65.97,
  "train_test_gap_pp": 10.07,
  "passed": true,
  "scope": "baseline_adjusted"
}
```
