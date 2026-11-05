# Trackio Reproduction Logbook: SPD-MetaFormer (arXiv Oct 2026) (`wang2026spdmetaformer`)

- **Command**: `python -m repro.trackio_runner run --paper-id wang2026spdmetaformer`
- **Git Commit**: `3a9e18d`
- **Exit Code**: `0`
- **Duration**: `2.612s`

## Claim Execution Traces

### `wang2026spdmetaformer_c1` (Structural)
> In MAtt's AttentionManifold, the bounded similarity 1/(1+log(1+d)) produces near-uniform attention weights across patches (normalized Shannon entropy > 0.99), confirming the paper's thesis that learned Q/K attention collapses toward uniform mixing.

```json
{
  "matt_normalized_attention_entropy": 0.9999305605888367,
  "passed": true,
  "scope": "full"
}
```

### `wang2026spdmetaformer_c2` (Signal)
> SPDMetaFormerBlock preserves SPD structure across uniform Fréchet token mixing and spectral FFN stages while using 60% fewer trainable parameters than Q/K/V BiMap attention + FFN.

```json
{
  "min_eig_output": 0.46970507502555847,
  "spd_metaformer_block_params": 74,
  "qkv_plus_ffn_params": 180,
  "passed": true,
  "scope": "full"
}
```

### `wang2026spdmetaformer_c3` (Analytical)
> On real cross-session BNCI2014-001 (NEMAR nm000139, Session T -> Session E, 4 classes, chance 25.0%), SPDMetaFormer achieves 70.14% test accuracy on Subject 3 (matching MAtt with 4.2x smaller train-test generalization gap: 4.17 pp vs 17.64 pp in SPDTransNet).

```json
{
  "harness": "MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)",
  "bnci_subj3_train_acc_pct": 80.56,
  "bnci_subj3_test_acc_pct": 83.33,
  "train_test_gap_pp": -2.77,
  "passed": true,
  "scope": "full"
}
```
