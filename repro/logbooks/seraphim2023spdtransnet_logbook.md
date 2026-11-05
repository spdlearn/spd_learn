# Trackio Reproduction Logbook: SPDTransNet & Structure-Preserving MHA (arXiv 2023) (`seraphim2023spdtransnet`)

- **Command**: `python -m repro.trackio_runner run --paper-id seraphim2023spdtransnet`
- **Git Commit**: `b3cf097`
- **Exit Code**: `0`
- **Duration**: `0.543s`

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
> Mapping SP-MHA output vectors back via vec_to_sym + matrix_exp yields valid SPD matrices whose eigenvalues remain bounded within the convex hull of input token log-spectra when head weights are non-negative.

```json
{
  "reconstructed_spd_min_eig": 0.6505951285362244,
  "passed": true,
  "scope": "full"
}
```

### `seraphim2023spdtransnet_c3` (Analytical)
> SPDTransNet trained end-to-end on multi-patch EEG sequences achieves >0.75 test accuracy.

```json
{
  "test_acc": 1.0,
  "passed": true,
  "scope": "toy"
}
```
