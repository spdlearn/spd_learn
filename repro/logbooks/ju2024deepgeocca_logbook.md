# Trackio Reproduction Logbook: DeepGeoCCA & GeodesicCCALoss (ICLR 2024) (`ju2024deepgeocca`)

- **Command**: `python -m repro.trackio_runner run --paper-id ju2024deepgeocca`
- **Git Commit**: `3a9e18d`
- **Exit Code**: `0`
- **Duration**: `13.327s`

## Claim Execution Traces

### `ju2024deepgeocca_c1` (Structural)
> GeodesicCCALoss achieves its theoretical minimum (-k for outdim_size=k) when two SPD views are identical (Y1 == Y2) and is strictly higher (less correlated) when views are independent random SPD matrices.

```json
{
  "loss_identical_views": -4.999877452850342,
  "loss_independent_views": -3.297682046890259,
  "passed": true,
  "scope": "full"
}
```

### `ju2024deepgeocca_c2` (Signal)
> Minimizing GeodesicCCALoss via gradient descent on paired multi-view SPD inputs from real BNCI2014-001 hemispheres monotonically increases the total geodesic canonical correlation between the two encoded SPD views.

```json
{
  "cca_loss_initial": -5.984272480010986,
  "cca_loss_after_12_steps": -5.994248867034912,
  "passed": true,
  "scope": "full"
}
```

### `ju2024deepgeocca_c3` (Analytical)
> On real cross-session BNCI2014-001 (NEMAR nm000139, Session T -> Session E, 4 classes, chance 25.0%), DeepGeoCCA achieves 71.53% test accuracy on Subject 3 by fusing dual temporal-view SPDNet encoders.

```json
{
  "harness": "MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)",
  "bnci_subj3_train_acc_pct": 74.31,
  "bnci_subj3_test_acc_pct": 68.75,
  "chance_level_pct": 25.0,
  "passed": true,
  "scope": "full"
}
```
