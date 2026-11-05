# Trackio Reproduction Logbook: DeepGeoCCA & GeodesicCCALoss (ICLR 2024) (`ju2024deepgeocca`)

- **Command**: `python -m repro.trackio_runner run --paper-id ju2024deepgeocca`
- **Git Commit**: `b3cf097`
- **Exit Code**: `0`
- **Duration**: `1.860s`

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
> Minimizing GeodesicCCALoss via gradient descent on paired multi-view SPD inputs monotonically increases the total geodesic canonical correlation between the two encoded SPD views.

```json
{
  "cca_loss_initial": -3.884921073913574,
  "cca_loss_after_15_steps": -3.9905545711517334,
  "passed": true,
  "scope": "full"
}
```

### `ju2024deepgeocca_c3` (Analytical)
> DeepGeoCCA decodes multi-class EEG signals using its dual-view SPDNet encoders with >0.75 test accuracy.

```json
{
  "test_acc": 1.0,
  "passed": true,
  "scope": "toy"
}
```
