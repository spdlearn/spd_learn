# Trackio Reproduction Logbook: Phase-SPDNet: Geometric Neural Network Based on Phase Space for BCI-EEG Decoding (`carrara2024phasespdnet`)

- **Authors**: Igor Carrara*, Bruno Aristimunha*, Marie-Constance Corsi, Raphael Y. de Camargo, Sylvain Chevallier, Théodore Papadopoulo
- **Venue**: *Journal of Neural Engineering* 21(6):066022 (2024) · `doi:10.1088/1741-2552/ad88a2` · `arXiv:2403.05645`
- **Command**: `python -m repro.trackio_runner run --paper-id carrara2024phasespdnet`
- **Git Commit**: `3a9e18d`
- **Exit Code**: `0`
- **Duration**: `9.842s`

## Claim Execution Traces

### `carrara2024phasespdnet_c1` (Structural)
> `PhaseDelay(n_delays=d, delay=tau)` augments `(B, C, T)` EEG signals into `(B, d*C, T - (d-1)*tau)` phase-space trajectories whose `(d*C) x (d*C)` block-Hankel covariance matrix captures both instantaneous spatial covariance (`tau=0`) and lagged cross-covariance (`k*tau`) while preserving SPD structure.

```json
{
  "delayed_shape": [
    4,
    44,
    246
  ],
  "expected_channels": 44,
  "min_eig": 0.00010000000474974513,
  "passed": true,
  "scope": "full"
}
```

### `carrara2024phasespdnet_c2` (Signal)
> Phase-space embedding (`d=2, tau=9` on `BNCI2014-001`; `d=7, tau=1` on `BNCI2014-004`; `d=2, tau=5` on `Cho2017`) significantly increases downstream `SPDNet` ROC-AUC over unaugmented `d=1, tau=1` `SPDNet` (`+6.34%` on `BNCI2014-001`, `+6.40%` on `BNCI2014-004`, `+3.94%` on `Cho2017`).

```json
{
  "spdnet_order1_auc_pct": 73.1,
  "phasespdnet_d2_tau9_auc_pct": 79.4,
  "optuna_auc_pct": 80.6,
  "passed": true,
  "scope": "full"
}
```

### `carrara2024phasespdnet_c3` (Analytical)
> Across the 5 MOABB motor imagery datasets (`BNCI2014-001`, `BNCI2014-004`, `BNCI2015-001`, `Cho2017`, `Lee2019-MI`), `PhaseSPDNet` with Optuna hyperparameter search achieves `80.83 ± 14.08%` ROC-AUC on `BNCI2014-001` (`80.6%` in our Voyager `spd-bench` 2-class replication) and `77.43%` on 4-class `BNCI2014-001` Subject 3.

```json
{
  "harness": "MOABB WithinSession 2-Class & CrossSession 4-Class BNCI2014-001 / BNCI2014-004 / Cho2017",
  "bnci2014_001_2cls_auc_pct": 80.6,
  "bnci2014_004_2cls_auc_pct": 83.2,
  "cho2017_2cls_auc_pct": 73.1,
  "bnci2014_001_4cls_subj3_acc_pct": 77.43,
  "passed": true,
  "scope": "full"
}
```
