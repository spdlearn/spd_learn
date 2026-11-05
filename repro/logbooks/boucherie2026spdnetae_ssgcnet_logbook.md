# Trackio Reproduction Logbook: SPDNet-AE (HAL 2026) & SSGCNet (IEEE TNSRE 2026) (`boucherie2026spdnetae_ssgcnet`)

- **Command**: `python -m repro.trackio_runner run --paper-id boucherie2026spdnetae_ssgcnet`
- **Git Commit**: `b3cf097`
- **Exit Code**: `0`
- **Duration**: `1.232s`

## Claim Execution Traces

### `boucherie2026spdnetae_ssgcnet_c1` (Structural)
> SPDNetAE's decoder reconstructs strictly SPD matrices (100% positive eigenvalues) from low-dimensional SPD latents, whereas an unconstrained Euclidean MLP autoencoder trained on flattened matrices produces indefinite matrices (negative eigenvalues) under noise/extrapolation.

```json
{
  "latent_min_eig": 0.1617300659418106,
  "reconstructed_min_eig": 0.16173020005226135,
  "passed": true,
  "scope": "full"
}
```

### `boucherie2026spdnetae_ssgcnet_c2` (Signal)
> Minimizing SPDNetAE.reconstruction_loss via gradient descent monotonically reduces the Log-Euclidean geodesic reconstruction error on held-out SPD covariance matrices, and SSGCNet's k-NN tangent graph aggregation preserves SPDness.

```json
{
  "initial_geodesic_recon_loss": 11.72391128540039,
  "final_geodesic_recon_loss": 7.903109073638916,
  "passed": true,
  "scope": "full"
}
```

### `boucherie2026spdnetae_ssgcnet_c3` (Analytical)
> Both SPDNetAE and SSGCNet achieve >0.75 test accuracy on end-to-end EEG classification.

```json
{
  "spdnetae_test_acc": 0.949999988079071,
  "ssgcnet_test_acc": 1.0,
  "passed": true,
  "scope": "toy"
}
```
