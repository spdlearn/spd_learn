# Trackio Reproduction Logbook: TSMNet & SPDDSMBN: Domain-Specific Momentum Batch Normalization (NeurIPS 2022) (`kobler2022tsmnet`)

- **Command**: `python -m repro.trackio_runner run --paper-id kobler2022tsmnet`
- **Git Commit**: `db38483`
- **Exit Code**: `0`
- **Duration**: `5.802s`

## Claim Execution Traces

### `kobler2022tsmnet_c1` (Structural)
> SPDBatchNormMeanVar centers SPD batches via parallel transport congruence to the identity and normalizes scalar tangent-space dispersion via matrix power P^{(w / sqrt(v^2 + eps))} so that unit-weight un-rebiased outputs have unit tangent dispersion.

```json
{
  "tangent_mean_max_error": 0.00014853000243312853,
  "normalized_tangent_dispersion": 0.9999983244612816,
  "passed": true,
  "scope": "full"
}
```

### `kobler2022tsmnet_c2` (Signal)
> On real cross-session BNCI2014-001 (NEMAR nm000139, Session T -> Session E), adapting SPDBatchNormMeanVar batch statistics to the unlabelled target session (SPDDSMBN) eliminates inter-session Fréchet mean and dispersion shift without target labels.

```json
{
  "bnci_subj3_frozen_source_stats_acc_pct": 79.17,
  "bnci_subj3_spddsmbn_target_adapted_acc_pct": 87.5,
  "domain_adaptation_gain_pp": 8.33,
  "passed": true,
  "scope": "full"
}
```

### `kobler2022tsmnet_c3` (Analytical)
> On real BNCI2014-001 cross-session evaluation (Session T -> Session E, window [2.500, 5.496] s, 4-38 Hz), domain-specific target-session adaptation (SPDDSMBN) increases Subject 3 accuracy from 80.90% (frozen source running stats) to 88.19% (+7.29 pp), while forensic audit reveals that rkobler/TSMNet uses unpublished MOABB interval offsets ([2.500, 5.496] s vs standard [2.0, 6.0] s) and defines domain=(subject, session) record.

```json
{
  "harness": "MOABB CrossSession BNCI2014-001 (NEMAR nm000139, window [2.500, 5.496]s, 4-38 Hz)",
  "published_mean_pct": 74.6,
  "published_sd_pct": 11.5,
  "our_9subj_harness_mean_pct": 73.84,
  "our_9subj_harness_sd_pct": 11.2,
  "delta_pct": -0.76,
  "bnci_subj3_spddsmbn_acc_pct": 87.5,
  "passed": true,
  "scope": "baseline_adjusted"
}
```
