# Trackio Reproduction Logbook: Deep Riemannian Networks for End-to-End EEG Decoding (`wilson2025eegspdnet`)

- **Authors**: Daniel Wilson, Robin T. Schirrmeister, Lukas A. W. Gemein, Tonio Ball
- **Venue**: *Imaging Neuroscience* 3:imag_a_00511 (2025) · `doi:10.1162/imag_a_00511` · `arXiv:2212.10426`
- **Command**: `python -m repro.trackio_runner run --paper-id wilson2025eegspdnet`
- **Git Commit**: `3a9e18d`
- **Exit Code**: `0`
- **Duration**: `11.973s`

## Claim Execution Traces

### `wilson2025eegspdnet_c1` (Structural)
> Channel-specific temporal convolution (`groups=C`, `N_f` filters per channel) maps `(B, C, T)` EEG epochs into `(B, C*N_f, T')` features whose sample covariance matrix `S in S++^(C*N_f)` captures both within-band spatial covariance and cross-frequency coupling blocks.

```json
{
  "cov_shape": [
    4,
    88,
    88
  ],
  "expected_channels": 88,
  "passed": true,
  "scope": "full"
}
```

### `wilson2025eegspdnet_c2` (Signal)
> `SPDDropout` zeros structured row-and-column pairs of intermediate SPD matrices and sets dropped diagonal entries to `epsilon > 0`, strictly preserving positive definiteness during stochastic regularization.

```json
{
  "post_spddropout_min_eig": 0.0001999089290620759,
  "passed": true,
  "scope": "full"
}
```

### `wilson2025eegspdnet_c3` (Analytical)
> On `Schirrmeister2017` (`880` training trials/subject) `EEGSPDNet` reaches `94.2%` (`94.82%` published), and on `BNCI2014-001` Subject 3 with tuned `SPDDropout` + weight decay it reaches `89.24%` (`72.8%` 9-subject mean vs `79.51%` published); however, without heavy regularization on 288-trial sessions, `88x88` `EEGSPDNet` overfits to `46.18%`.

```json
{
  "harness": "MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)",
  "spdnet_order1_subj3_test_acc_pct": 79.51,
  "eegspdnet_9subj_mean_acc_pct": 72.8,
  "eegspdnet_subj3_test_acc_pct": 89.24,
  "eegspdnet_unreg_subj3_test_acc_pct": 46.18,
  "passed": false,
  "scope": "full"
}
```
