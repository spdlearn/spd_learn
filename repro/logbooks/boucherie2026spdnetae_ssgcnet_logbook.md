# Trackio Reproduction Logbook: SPDNet-AE & SSGCNet: Riemannian Autoencoding & Stiefel-SPD Graph Conv (HAL / IEEE TNSRE 2026) (`boucherie2026spdnetae_ssgcnet`)

- **Command**: `python -m repro.trackio_runner run --paper-id boucherie2026spdnetae_ssgcnet`
- **Git Commit**: `3a9e18d`
- **Exit Code**: `0`
- **Duration**: `7.162s`

## Claim Execution Traces

### `boucherie2026spdnetae_ssgcnet_c1` (Structural)
> SPDNetAE's transposed congruence decoder W Z W^T + eps * I guarantees strictly positive definite reconstructed covariance matrices S_rec in S++^C across all latent bottleneck dimensions d_lat < C.

```json
{
  "latent_min_eig": 0.12310659885406494,
  "reconstructed_min_eig": 0.12310611456632614,
  "passed": true,
  "scope": "full"
}
```

### `boucherie2026spdnetae_ssgcnet_c2` (Signal)
> Minimizing the Log-Euclidean geodesic reconstruction loss ||log(S_in) - log(S_rec)||_F^2 via gradient descent on real BNCI2014-001 EEG covariances monotonically reduces manifold reconstruction error.

```json
{
  "initial_geodesic_recon_loss": 65.17560577392578,
  "final_geodesic_recon_loss": 40.52341842651367,
  "passed": true,
  "scope": "full"
}
```

### `boucherie2026spdnetae_ssgcnet_c3` (Analytical)
> On real cross-session BNCI2014-001 (NEMAR nm000139, Session T -> Session E, 4 classes, chance 25.0%), joint supervised + geodesic autoencoder training in SPDNetAE achieves 65.97% test accuracy and SSGCNet achieves 66.32% test accuracy on Subject 3.

```json
{
  "harness": "MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)",
  "spdnetae_bnci_subj3_test_acc_pct": 67.01,
  "ssgcnet_bnci_subj3_test_acc_pct": 68.4,
  "chance_level_pct": 25.0,
  "passed": true,
  "scope": "full"
}
```
