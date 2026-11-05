# SPD-Learn ICML 2026 / NEMAR Open Reproductions Leaderboard

- **Judge**: `spd-learn-logbook-judge-1.0` (Untrusted-Narrative Execution Trace Auditor)
- **Audited Timestamp**: `2026-11-05T14:48:29Z`
- **Papers Audited**: `18`
- **Claims Audited**: `54` (`18 Structural`, `18 Signal`, `18 Analytical`)
- **Verdict Breakdown**: `Verified: 36` (2 pts) | `Toy: 18` (1 pt) | `Falsified: 0` (2 pts) | `Inconclusive: 0` (0 pts)
- **Total Reproducibility Score**: **`90 / 90`**

## Per-Paper Reproducibility Scorecard

| Paper ID | Model / Method | Venue | Structural (C1) | Signal (C2) | Analytical (C3) | Points |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `chen2026pcm_bwcm` | **PCM & BWCM Cholesky Product Metrics (ICLR 2026)** | ICLR 2026 | `Verified` | `Verified` | `Toy` | **5 / 5** |
| `hu2025coratt` | **CorAtt: Correlation Manifold Self-Attention (IJCAI 2025)** | IJCAI 2025 | `Verified` | `Verified` | `Toy` | **5 / 5** |
| `chen2026cornet` | **CorNet: Riemannian Networks over Full-Rank Correlation Matrices (ICML 2026)** | ICML 2026 | `Verified` | `Verified` | `Toy` | **5 / 5** |
| `chen2026rietrans` | **RieTrans: SPDLinear, SPDRMLR & SPNN (NeurIPS 2024 / 2026)** | NeurIPS 2024 / NeurIPS 2026 | `Verified` | `Verified` | `Toy` | **5 / 5** |
| `wang2025gyroatt` | **GyroAtt: GyroSPDAttention & SymSPDAttention (NeurIPS 2025)** | NeurIPS 2025 | `Verified` | `Verified` | `Toy` | **5 / 5** |
| `ju2023graphcspnet` | **Graph-CSPNet & GraphBiMap (IEEE TNNLS 2023)** | IEEE TNNLS 2023 | `Verified` | `Verified` | `Toy` | **5 / 5** |
| `zhou2026gbwatt` | **GBWAtt: Generalized Bures-Wasserstein Self-Attention (KDD 2026)** | KDD 2026 | `Verified` | `Verified` | `Toy` | **5 / 5** |
| `seraphim2023spdtransnet` | **SPDTransNet & Structure-Preserving MHA (arXiv 2023)** | arXiv 2023 | `Verified` | `Verified` | `Toy` | **5 / 5** |
| `wang2026spdmetaformer` | **SPD-MetaFormer (arXiv Oct 2026)** | arXiv 2026 | `Verified` | `Verified` | `Toy` | **5 / 5** |
| `hu2026rhop` | **RHOP: Riemannian High-Order Pooling for BFMs (ICLR 2026)** | ICLR 2026 | `Verified` | `Verified` | `Toy` | **5 / 5** |
| `ju2024deepgeocca` | **DeepGeoCCA & GeodesicCCALoss (ICLR 2024)** | ICLR 2024 | `Verified` | `Verified` | `Toy` | **5 / 5** |
| `gallet2026armagnac` | **ARMAGNAC: Parametric Harmonic-Arithmetic SPD BatchNorm (HAL / Signal Processing 2026)** | HAL / Signal Processing 2026 | `Verified` | `Verified` | `Toy` | **5 / 5** |
| `zhou2025gbwbn_cbn` | **GBWBN (CVPR 2025) & CBN (IEEE TNNLS 2025) SPD BatchNorms** | CVPR 2025 / IEEE TNNLS 2025 | `Verified` | `Verified` | `Toy` | **5 / 5** |
| `tibermacine2025rceegnet` | **RCEEGNet (IJCNN 2025, Issue #31)** | IJCNN 2025 | `Verified` | `Verified` | `Toy` | **5 / 5** |
| `wang2024starnet` | **STaRNet (Neural Networks 2024, Issue #32)** | Neural Networks 2024 | `Verified` | `Verified` | `Toy` | **5 / 5** |
| `chen2025alem_spdresnet` | **ALEM & SPDResNet (IEEE TIP 2024 / ICLR 2025 / NeurIPS 2026)** | IEEE TIP 2024 / ICLR 2025 | `Verified` | `Verified` | `Toy` | **5 / 5** |
| `nguyen2023gyrospd` | **GyroSPD: Matrix Manifold Neural Networks (ICML 2023)** | ICML 2023 | `Verified` | `Verified` | `Toy` | **5 / 5** |
| `boucherie2026spdnetae_ssgcnet` | **SPDNet-AE (HAL 2026) & SSGCNet (IEEE TNSRE 2026)** | HAL 2026 / IEEE TNSRE 2026 | `Verified` | `Verified` | `Toy` | **5 / 5** |

## Detailed Claim Verdicts & Audited Metrics

| Claim ID | Category | Verdict | Pts | Key Audited Metrics |
| :--- | :--- | :---: | :---: | :--- |
| `chen2026pcm_bwcm_c1` | Structural | **Verified** | 2 | `bw_vs_bwcm_diag_error=1.11e-15`, `log_exp_roundtrip_rel_error=1.67e-16` |
| `chen2026pcm_bwcm_c2` | Signal | **Verified** | 2 | `kappa=1e+06`, `grad_finite=True`, `pcm_time_sec=0.03545`, `eigh_time_sec=0.002694` |
| `chen2026pcm_bwcm_c3` | Analytical | **Toy** | 1 | `mdm_test_acc=1` |
| `hu2025coratt_c1` | Structural | **Verified** | 2 | `hollow_max_error=0`, `unit_diag_max_error=0`, `olm_roundtrip_rel_error=3.29e-07` |
| `hu2025coratt_c2` | Signal | **Verified** | 2 | `corr_attention_scale_drift=3.625e-07`, `cov_attention_scale_drift=10.43` |
| `hu2025coratt_c3` | Analytical | **Toy** | 1 | `coratt_test_acc=1`, `matt_test_acc=0.95` |
| `chen2026cornet_c1` | Structural | **Verified** | 2 | `tril_dim=28`, `expected_tril_dim=28`, `output_shape=[4, 2]` |
| `chen2026cornet_c2` | Signal | **Verified** | 2 | `max_logit_diff_under_diagonal_scaling=2.086e-07` |
| `chen2026cornet_c3` | Analytical | **Toy** | 1 | `test_acc=0.95` |
| `chen2026rietrans_c1` | Structural | **Verified** | 2 | `tangent_chaining_rel_error=4.214e-07` |
| `chen2026rietrans_c2` | Signal | **Verified** | 2 | `metrics_grad_finite={'LEM': True, 'LCM': True, 'AIM': True, 'PEM': True}` |
| `chen2026rietrans_c3` | Analytical | **Toy** | 1 | `test_acc=0.95` |
| `wang2025gyroatt_c1` | Structural | **Verified** | 2 | `min_eig_gyro=0.5778`, `min_eig_sym=0.5684` |
| `wang2025gyroatt_c2` | Signal | **Verified** | 2 | `bounded_norm_entropy=0.9678`, `scaled_norm_entropy=0.001667` |
| `wang2025gyroatt_c3` | Analytical | **Toy** | 1 | `test_acc=1` |
| `ju2023graphcspnet_c1` | Structural | **Verified** | 2 | `min_eig_output=0.697`, `output_shape=[4, 6, 5, 5]` |
| `ju2023graphcspnet_c2` | Signal | **Verified** | 2 | `variance_unmixed=0.01149`, `variance_graph_mixed=0.003136` |
| `ju2023graphcspnet_c3` | Analytical | **Toy** | 1 | `test_acc=1` |
| `zhou2026gbwatt_c1` | Structural | **Verified** | 2 | `min_eig_output=0.608`, `min_eig_reference_M=1` |
| `zhou2026gbwatt_c2` | Signal | **Verified** | 2 | `gbw_grad_norm_kappa_1e6=975.9`, `lem_grad_norm_kappa_1e6=2383`, `grad_finite=True` |
| `zhou2026gbwatt_c3` | Analytical | **Toy** | 1 | `test_acc=1` |
| `seraphim2023spdtransnet_c1` | Structural | **Verified** | 2 | `input_vec_dim=15`, `output_vec_dim=15`, `combined_attn_shape=[3, 6, 6]` |
| `seraphim2023spdtransnet_c2` | Signal | **Verified** | 2 | `reconstructed_spd_min_eig=0.6506` |
| `seraphim2023spdtransnet_c3` | Analytical | **Toy** | 1 | `test_acc=1` |
| `wang2026spdmetaformer_c1` | Structural | **Verified** | 2 | `matt_normalized_attention_entropy=1` |
| `wang2026spdmetaformer_c2` | Signal | **Verified** | 2 | `min_eig_output=0.4321`, `spd_metaformer_block_params=74`, `qkv_plus_ffn_params=180` |
| `wang2026spdmetaformer_c3` | Analytical | **Toy** | 1 | `small_sample_test_acc=1` |
| `hu2026rhop_c1` | Structural | **Verified** | 2 | `descriptor_shape=[8, 7, 7]`, `min_eig=0.09967` |
| `hu2026rhop_c2` | Signal | **Verified** | 2 | `mean_pool_feature_norm=4.231e-07`, `rhop_test_acc_zero_mean_cov_task=1` |
| `hu2026rhop_c3` | Analytical | **Toy** | 1 | `eeg_token_test_acc=0.95` |
| `ju2024deepgeocca_c1` | Structural | **Verified** | 2 | `loss_identical_views=-5`, `loss_independent_views=-3.298` |
| `ju2024deepgeocca_c2` | Signal | **Verified** | 2 | `cca_loss_initial=-3.885`, `cca_loss_after_15_steps=-3.991` |
| `ju2024deepgeocca_c3` | Analytical | **Toy** | 1 | `test_acc=1` |
| `gallet2026armagnac_c1` | Structural | **Verified** | 2 | `harmonic_endpoint_err=1.975e-15`, `arithmetic_endpoint_err=1.151e-14` |
| `gallet2026armagnac_c2` | Signal | **Verified** | 2 | `learned_alpha_for_harmonic_target=0.02374`, `learned_alpha_for_arithmetic_target=0.9849` |
| `gallet2026armagnac_c3` | Analytical | **Toy** | 1 | `armagnac_10_calls_sec=0.006997`, `karcher5_10_calls_sec=0.01684`, `test_acc=0.95` |
| `zhou2025gbwbn_cbn_c1` | Structural | **Verified** | 2 | `cholesky_tangent_mean_max_err=6.476e-16`, `cholesky_tangent_var_max_err=4.061e-05` |
| `zhou2025gbwbn_cbn_c2` | Signal | **Verified** | 2 | `kappa_1e6_bwbn_grad_finite=True` |
| `zhou2025gbwbn_cbn_c3` | Analytical | **Toy** | 1 | `test_acc=0.95` |
| `tibermacine2025rceegnet_c1` | Structural | **Verified** | 2 | `output_shape=[4, 2]` |
| `tibermacine2025rceegnet_c2` | Signal | **Verified** | 2 | `frontend_input_grad_flowing=True` |
| `tibermacine2025rceegnet_c3` | Analytical | **Toy** | 1 | `test_acc=1` |
| `wang2024starnet_c1` | Structural | **Verified** | 2 | `output_shape=[4, 2]` |
| `wang2024starnet_c2` | Signal | **Verified** | 2 | `riemannian_bimap_grad_norm=1.551`, `euclidean_se_grad_sum=114.4` |
| `wang2024starnet_c3` | Analytical | **Toy** | 1 | `test_acc=1` |
| `chen2025alem_spdresnet_c1` | Structural | **Verified** | 2 | `alem_spd_min_eig=0.2676`, `spdresblock_min_eig=0.2223` |
| `chen2025alem_spdresnet_c2` | Signal | **Verified** | 2 | `input_grad_norm_2_blocks=19.42` |
| `chen2025alem_spdresnet_c3` | Analytical | **Toy** | 1 | `test_acc=1` |
| `nguyen2023gyrospd_c1` | Structural | **Verified** | 2 | `min_eig_output=0.3552` |
| `nguyen2023gyrospd_c2` | Signal | **Verified** | 2 | `input_trace=8.028`, `gyro_translated_trace=55.43` |
| `nguyen2023gyrospd_c3` | Analytical | **Toy** | 1 | `test_acc=1` |
| `boucherie2026spdnetae_ssgcnet_c1` | Structural | **Verified** | 2 | `latent_min_eig=0.1617`, `reconstructed_min_eig=0.1617` |
| `boucherie2026spdnetae_ssgcnet_c2` | Signal | **Verified** | 2 | `initial_geodesic_recon_loss=11.72`, `final_geodesic_recon_loss=7.903` |
| `boucherie2026spdnetae_ssgcnet_c3` | Analytical | **Toy** | 1 | `spdnetae_test_acc=0.95`, `ssgcnet_test_acc=1` |
