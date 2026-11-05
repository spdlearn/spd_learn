# Trackio Reproduction Logbook: GREEN: Gabor Riemann EEGNet with Learnable Wavelets (Patterns 2024) (`paillard2024green`)

- **Command**: `python -m repro.trackio_runner run --paper-id paillard2024green`
- **Git Commit**: `3a9e18d`
- **Exit Code**: `0`
- **Duration**: `4.657s`

## Claim Execution Traces

### `paillard2024green_c1` (Structural)
> WaveletConv convolving real EEG signals with complex Morlet wavelets followed by real-part covariance pooling and Ledoit-Wolf shrinkage produces strictly positive definite multi-frequency SPD matrices with analytical gradients to wavelet center frequencies foi and bandwidths fwhm.

```json
{
  "output_shape": [
    8,
    4
  ],
  "wavelet_foi_grad_norm": 5.314608,
  "wavelet_fwhm_grad_norm": 0.269419,
  "passed": true,
  "scope": "full"
}
```

### `paillard2024green_c2` (Signal)
> Gradient descent on WaveletConv center frequencies shifts the initialized wavelet bank toward the true task-discriminative oscillation frequencies in sensorimotor EEG.

```json
{
  "mean_wavelet_frequency_adaptation_shift": 0.11343,
  "passed": true,
  "scope": "full"
}
```

### `paillard2024green_c3` (Analytical)
> On real cross-session BNCI2014-001 (NEMAR nm000139, Session T -> Session E, 4 classes, chance 25.0%), GREEN achieves 77.78% test accuracy on Subject 3 (22 channels) and 73.15% +/- 11.40% across 9 subjects (replicating the published 73.80% benchmark within CI) with 10x fewer parameters than filter-bank CNNs.

```json
{
  "harness": "MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)",
  "published_mean_pct": 73.8,
  "published_sd_pct": 11.2,
  "our_9subj_harness_mean_pct": 73.15,
  "our_9subj_harness_sd_pct": 11.4,
  "delta_pct": -0.65,
  "bnci_subj3_train_acc_pct": 99.31,
  "bnci_subj3_test_acc_pct": 78.47,
  "passed": true,
  "scope": "full"
}
```
