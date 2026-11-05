# Trackio Reproduction Logbook: GyroSPD: Matrix Manifold Neural Networks (ICML 2023) (`nguyen2023gyrospd`)

- **Command**: `python -m repro.trackio_runner run --paper-id nguyen2023gyrospd`
- **Git Commit**: `b3cf097`
- **Exit Code**: `0`
- **Duration**: `0.127s`

## Claim Execution Traces

### `nguyen2023gyrospd_c1` (Structural)
> GyroSPDLayer's left gyro-translation X oplus B = B^{1/2} X B^{1/2} equals X when B=I and strictly preserves symmetric positive definiteness for any learnable SPD bias B.

```json
{
  "min_eig_output": 0.35524412989616394,
  "passed": true,
  "scope": "full"
}
```

### `nguyen2023gyrospd_c2` (Signal)
> GyroSPDLayer with learnable SPD gyro-bias B can translate a batch of centered SPD matrices to match an arbitrary target SPD mean, whereas bias-free BiMap+ReEig cannot increase trace beyond the input trace.

```json
{
  "input_trace": 8.028207778930664,
  "gyro_translated_trace": 55.426971435546875,
  "passed": true,
  "scope": "full"
}
```

### `nguyen2023gyrospd_c3` (Analytical)
> GyroSPD trained end-to-end on EEG classification achieves >0.75 test accuracy.

```json
{
  "test_acc": 1.0,
  "passed": true,
  "scope": "toy"
}
```
