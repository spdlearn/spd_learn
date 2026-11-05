# Trackio Reproduction Logbook: SPD-MetaFormer (arXiv Oct 2026) (`wang2026spdmetaformer`)

- **Command**: `python -m repro.trackio_runner run --paper-id wang2026spdmetaformer`
- **Git Commit**: `b3cf097`
- **Exit Code**: `0`
- **Duration**: `0.501s`

## Claim Execution Traces

### `wang2026spdmetaformer_c1` (Structural)
> In MAtt's AttentionManifold, the bounded similarity 1/(1+log(1+d)) produces near-uniform attention weights across patches (normalized Shannon entropy > 0.95), confirming the paper's motivation that learned Q/K attention collapses toward uniform mixing.

```json
{
  "matt_normalized_attention_entropy": 0.9999536871910095,
  "passed": true,
  "scope": "full"
}
```

### `wang2026spdmetaformer_c2` (Signal)
> SPDMetaFormerBlock preserves SPD structure across uniform Fréchet token mixing and spectral FFN stages while using fewer trainable parameters than Q/K/V BiMap attention + FFN.

```json
{
  "min_eig_output": 0.43210482597351074,
  "spd_metaformer_block_params": 74,
  "qkv_plus_ffn_params": 180,
  "passed": true,
  "scope": "full"
}
```

### `wang2026spdmetaformer_c3` (Analytical)
> In a small-sample EEG training regime (N <= 48 training trials), SPDMetaFormer matches or exceeds MAtt test accuracy with lower generalization gap.

```json
{
  "small_sample_test_acc": 1.0,
  "passed": true,
  "scope": "toy"
}
```
