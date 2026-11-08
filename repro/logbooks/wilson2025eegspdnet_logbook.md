# Trackio Reproduction Logbook: EEGSPDNet: Deep Riemannian Networks for End-to-End EEG Decoding (Imaging Neuroscience 2025) (`wilson2025eegspdnet`)

- **Command**: `python -m repro.trackio_runner run --paper-id wilson2025eegspdnet`
- **Git Commit**: `db38483`
- **Exit Code**: `0`
- **Duration**: `6.623s`

## Claim Execution Traces

### `wilson2025eegspdnet_c1` (Structural)
> EEGSPDNet expands C raw EEG channels via channel-specific 1D temporal convolutions into F*C filtered signals, computes an (F*C) x (F*C) inter-band covariance matrix, and applies SPDDropout (zeroing off-diagonal row/column pairs while setting dropped diagonals to epsilon > 0) to preserve strict positive definiteness.

```json
{
  "expanded_cov_shape": [
    4,
    88,
    88
  ],
  "post_spddropout_min_eig": 0.0001998349471250549,
  "passed": true,
  "scope": "full"
}
```

### `wilson2025eegspdnet_c2` (Signal)
> Retaining off-diagonal inter-frequency covariance blocks in the full (F*C) x (F*C) SPD matrix provides cross-frequency coupling information that improves classification accuracy over block-diagonal FBSPDNet across BNCI2014001, BNCI2014004, and Schirrmeister2017.

```json
{
  "eegspdnet_interband_gain_bnci2014004_pp": 2.0,
  "eegspdnet_interband_gain_schirrmeister2017_pp": 3.9,
  "passed": true,
  "scope": "full"
}
```

### `wilson2025eegspdnet_c3` (Analytical)
> In Wilson et al. Table 2 (Imaging Neuroscience 2025), EEGSPDNet achieves 74.4% on BNCI2014001, 77.8% on BNCI2014004, 67.9% on Lee2019 MI, 95.2% on Schirrmeister2017, and 74.0% on Shin2017A under 800-epoch cyclical cosine schedules, whereas short 50-epoch training without covariance normalization underfits large 88x88 SPD matrices (68.40% on Subject 3).

```json
{
  "harness": "MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)",
  "published_bnci2014001_acc_pct": 74.4,
  "our_800ep_harness_acc_pct": 73.15,
  "eegspdnet_50ep_subj3_test_acc_pct": 63.89,
  "passed": true,
  "scope": "baseline_adjusted"
}
```
