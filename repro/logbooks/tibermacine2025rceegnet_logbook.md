# Trackio Reproduction Logbook: RCEEGNet (IJCNN 2025, Issue #31) (`tibermacine2025rceegnet`)

- **Command**: `python -m repro.trackio_runner run --paper-id tibermacine2025rceegnet`
- **Git Commit**: `b3cf097`
- **Exit Code**: `0`
- **Duration**: `0.350s`

## Claim Execution Traces

### `tibermacine2025rceegnet_c1` (Structural)
> RCEEGNet preserves SPD structure across its BiMap/ReEig/SPDBatchNormMean stages and projects to a 2D symmetric tangent map (upper=False) for 2D convolutional feature extraction.

```json
{
  "output_shape": [
    4,
    2
  ],
  "passed": true,
  "scope": "full"
}
```

### `tibermacine2025rceegnet_c2` (Signal)
> RCEEGNet's 2D tangent-space convolution captures localized off-diagonal channel-pair interactions with non-zero gradients flowing back to the temporal Conv1d frontend.

```json
{
  "frontend_input_grad_flowing": true,
  "passed": true,
  "scope": "full"
}
```

### `tibermacine2025rceegnet_c3` (Analytical)
> RCEEGNet trains end-to-end on multi-class oscillatory EEG signals and achieves >0.75 test accuracy.

```json
{
  "test_acc": 1.0,
  "passed": true,
  "scope": "toy"
}
```
