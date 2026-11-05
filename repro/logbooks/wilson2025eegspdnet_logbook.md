# Trackio Reproduction Logbook: EEGSPDNet & PhaseSPDNet: Channel-Specific Conv & Phase-Space Delay SPD Networks (2024-2025) (`wilson2025eegspdnet`)

- **Command**: `python -m repro.trackio_runner run --paper-id wilson2025eegspdnet`
- **Git Commit**: `3a9e18d`
- **Exit Code**: `0`
- **Duration**: `11.973s`

## Claim Execution Traces

### `wilson2025eegspdnet_c1` (Structural)
> PhaseDelay(order=m, lag=tau) augments (B, C, T) signals into (B, m*C, T - m*tau) phase-space trajectories whose block-Hankel covariance matrix captures both instantaneous spatial covariance and lagged cross-covariance while preserving SPD structure.

```json
{
  "delayed_shape": [
    4,
    44,
    246
  ],
  "expected_channels": 44,
  "passed": true,
  "scope": "full"
}
```

### `wilson2025eegspdnet_c2` (Signal)
> SPDDropout zeros structured row-and-column pairs of intermediate SPD matrices and sets dropped diagonal entries to epsilon > 0, strictly preserving positive definiteness during stochastic regularization.

```json
{
  "post_spddropout_min_eig": 0.0001999089290620759,
  "passed": true,
  "scope": "full"
}
```

### `wilson2025eegspdnet_c3` (Analytical)
> On real cross-session BNCI2014-001 (NEMAR nm000139, Session T -> Session E, 4 classes, chance 25.0%), PhaseSPDNet (order=2, lag=2) increases Subject 3 test accuracy from 78.82% (order=1 static SPDNet) to 83.68% (+4.86 pp) and EEGSPDNet achieves 68.40% on Subject 3.

```json
{
  "harness": "MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)",
  "spdnet_order1_subj3_test_acc_pct": 79.51,
  "phasespdnet_order2_subj3_test_acc_pct": 72.22,
  "eegspdnet_subj3_test_acc_pct": 46.18,
  "passed": false,
  "scope": "full"
}
```
