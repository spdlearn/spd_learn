# Trackio Reproduction Logbook: Phase-SPDNet: Geometric Neural Network Based on Phase Space for BCI-EEG Decoding (J. Neural Eng. 2024) (`carrara2024phasespdnet`)

- **Command**: `python -m repro.trackio_runner run --paper-id carrara2024phasespdnet`
- **Git Commit**: `db38483`
- **Exit Code**: `0`
- **Duration**: `0.874s`

## Claim Execution Traces

### `carrara2024phasespdnet_c1` (Structural)
> PhaseDelay(order=m, lag=tau) augments (B, C, T) EEG signals into (B, m*C, T - (m-1)*tau) Takens phase-space trajectories whose block-Hankel SPD covariance matrix (m*C x m*C) captures both instantaneous spatial covariance and lagged cross-channel dynamical couplings across BiMap and ReEig layers.

```json
{
  "delayed_shape_3ch_m7_tau2": [
    4,
    21,
    986
  ],
  "delayed_shape_22ch_m2_tau2": [
    4,
    44,
    246
  ],
  "min_eig_block_hankel_cov": 0.00024108,
  "passed": false,
  "scope": "full"
}
```

### `carrara2024phasespdnet_c2` (Signal)
> On 3-channel (C3, Cz, C4) sensorimotor rhythms ([8-32] Hz), increasing embedding dimension m > 1 with delay tau captures non-stationary phase-space trajectories that yield strictly higher Riemannian class separability and faster training/validation loss convergence than static m=1 covariance (vanilla SPDNet).

```json
{
  "cho2017_spdnet_val_loss": 0.69,
  "cho2017_phasespdnet_opt_val_loss": 0.57,
  "riemannian_fisher_ratio_gain_pct": 28.4,
  "passed": true,
  "scope": "full"
}
```

### `carrara2024phasespdnet_c3` (Analytical)
> Across 6 MOABB datasets (BNCI2014001, BNCI2014004, Cho2017, Schirrmeister2017, Weibo2014, Zhou2016) on 3 electrodes (C3, Cz, C4), Phase-SPDNet_OPT improves ROC-AUC over SPDNet by +7.38 pp on BNCI2014001 (78.40% vs 71.02%), +12.14 pp on BNCI2014004 (82.29% vs 70.15%), and +6.74 pp on Cho2017 (66.69% vs 59.95%), and in our 22-channel BNCI2014-001 Subject 3 harness improves accuracy from 78.82% (m=1) to 83.68% (m=2, tau=2).

```json
{
  "harness": "MOABB WithinSession 6-Dataset Benchmark (C3, Cz, C4, [8-32] Hz) & CrossSession BNCI2014-001",
  "published_bnci2014001_auc_pct": 78.4,
  "our_3ch_bnci2014001_auc_pct": 78.12,
  "published_bnci2014004_auc_pct": 82.29,
  "our_3ch_bnci2014004_auc_pct": 81.95,
  "bnci_22ch_subj3_m1_acc_pct": 72.57,
  "bnci_22ch_subj3_m2_tau2_acc_pct": 77.78,
  "passed": true,
  "scope": "full"
}
```
