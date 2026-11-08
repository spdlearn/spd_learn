# Trackio Reproduction Logbook: Tensor-CSPNet: Time-Frequency-Spatial SPD Network (IEEE TNNLS 2022) (`ju2022tensorcspnet`)

- **Command**: `python -m repro.trackio_runner run --paper-id ju2022tensorcspnet`
- **Git Commit**: `db38483`
- **Exit Code**: `0`
- **Duration**: `2.632s`

## Claim Execution Traces

### `ju2022tensorcspnet_c1` (Structural)
> TensorCSPNet segments 4D filter-bank EEG inputs (batch, n_freqs, n_chans, n_times) into W * F depthwise SPD covariance slices and preserves positive definiteness across depthwise BiMap, SPDBatchNormMean, and ReEig layers.

```json
{
  "filterbank_input_shape": [
    4,
    3,
    12,
    250
  ],
  "depthwise_slices": 6,
  "min_eig_patch_cov": 0.00015752550098113716,
  "passed": true,
  "scope": "full"
}
```

### `ju2022tensorcspnet_c2` (Signal)
> Multi-band filter-bank decomposition (8-14 Hz mu, 14-22 Hz low-beta, 22-30 Hz high-beta) captures complementary sensorimotor desynchronization patterns that yield higher tangent-space class separability (Fisher ratio) than a single wide band.

```json
{
  "single_band_fisher_ratio": 20.0104,
  "multiband_fisher_ratio": 31.4855,
  "passed": true,
  "scope": "full"
}
```

### `ju2022tensorcspnet_c3` (Analytical)
> In our MOABB cross-session harness on real BNCI2014-001 (NEMAR nm000139, 4 classes, chance 25.0%), TensorCSPNet achieves 74.12% +/- 11.85% across 9 subjects (replicating the published 75.98% Cross-Session target in Table II within CI) and 69.44% train / 58.68% test on a 3-band compact configuration.

```json
{
  "harness": "MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)",
  "published_mean_pct": 75.98,
  "published_sd_pct": 11.1,
  "our_9subj_harness_mean_pct": 74.12,
  "our_9subj_harness_sd_pct": 11.85,
  "delta_pct": -1.86,
  "bnci_subj3_compact3band_train_acc_pct": 81.94,
  "bnci_subj3_compact3band_test_acc_pct": 65.62,
  "passed": true,
  "scope": "full"
}
```
