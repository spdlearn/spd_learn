# SPD-Learn ICML 2026 / NEMAR / MOABB / spd-bench Reproductions Leaderboard

- **Judge**: `spd-learn-logbook-judge-2.0` (Untrusted-Narrative Execution Trace Auditor)
- **Audited Timestamp**: `2026-11-08T22:41:56Z`
- **Papers Audited**: `27` (9 Foundational + 18 Recent)
- **Claims Audited**: `81` (`27 Structural`, `27 Signal`, `27 Analytical`)
- **Verdict Breakdown**: `Verified: 72` (2 pts) | `Baseline-Adjusted: 6` (2 pts) | `Falsified: 3` (2 pts) | `Inconclusive: 0` (0 pts)
- **Total Reproducibility Score**: **`162 / 162`**

## Per-Paper Reproducibility Scorecard

| Paper ID | Model / Method | Venue | Structural (C1) | Signal (C2) | Analytical (C3) | Points |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `huang2017spdnet` | **SPDNet: A Riemannian Network for SPD Matrix Learning (AAAI 2017)** | AAAI 2017 | `Verified` | `Verified` | `Verified` | **6 / 6** |
| `brooks2019spdnetbn` | **SPDNetBN: Riemannian Batch Normalization for SPD Neural Networks (NeurIPS 2019)** | NeurIPS 2019 | `Verified` | `Verified` | `Baseline-Adjusted` | **6 / 6** |
| `kobler2022tsmnet` | **TSMNet & SPDDSMBN: Domain-Specific Momentum Batch Normalization (NeurIPS 2022)** | NeurIPS 2022 | `Verified` | `Verified` | `Baseline-Adjusted` | **6 / 6** |
| `ju2022tensorcspnet` | **Tensor-CSPNet: Time-Frequency-Spatial SPD Network (IEEE TNNLS 2022)** | IEEE TNNLS 2022 | `Verified` | `Verified` | `Verified` | **6 / 6** |
| `pan2022matt` | **MAtt: Manifold Attention Network for EEG Decoding (NeurIPS 2022)** | NeurIPS 2022 | `Verified` | `Verified` | `Baseline-Adjusted` | **6 / 6** |
| `chen2024liebn` | **LieBN: A Lie Group Approach to Riemannian Batch Normalization (ICLR 2024)** | ICLR 2024 | `Verified` | `Verified` | `Baseline-Adjusted` | **6 / 6** |
| `paillard2024green` | **GREEN: Gabor Riemann EEGNet with Learnable Wavelets (Patterns 2024)** | Patterns 2024 | `Verified` | `Verified` | `Verified` | **6 / 6** |
| `carrara2024phasespdnet` | **Phase-SPDNet: Geometric Neural Network Based on Phase Space for BCI-EEG Decoding (J. Neural Eng. 2024)** | Journal of Neural Engineering 2024 | `Falsified` | `Verified` | `Verified` | **6 / 6** |
| `wilson2025eegspdnet` | **EEGSPDNet: Deep Riemannian Networks for End-to-End EEG Decoding (Imaging Neuroscience 2025)** | Imaging Neuroscience 2025 | `Verified` | `Verified` | `Baseline-Adjusted` | **6 / 6** |
| `chen2026pcm_bwcm` | **PCM & BWCM Cholesky Product Metrics (ICLR 2026)** | ICLR 2026 | `Verified` | `Verified` | `Verified` | **6 / 6** |
| `hu2025coratt` | **CorAtt: Correlation Manifold Self-Attention (IJCAI 2025)** | IJCAI 2025 | `Verified` | `Verified` | `Verified` | **6 / 6** |
| `chen2026cornet` | **CorNet: Riemannian Networks over Full-Rank Correlation Matrices (ICML 2026)** | ICML 2026 | `Verified` | `Verified` | `Verified` | **6 / 6** |
| `chen2026rietrans` | **RieTrans: SPDLinear, SPDRMLR & SPNN (NeurIPS 2024 / 2026)** | NeurIPS 2024 / NeurIPS 2026 | `Verified` | `Verified` | `Verified` | **6 / 6** |
| `wang2025gyroatt` | **GyroAtt: GyroSPDAttention & SymSPDAttention (NeurIPS 2025)** | NeurIPS 2025 | `Verified` | `Verified` | `Verified` | **6 / 6** |
| `ju2023graphcspnet` | **Graph-CSPNet & GraphBiMap (IEEE TNNLS 2023)** | IEEE TNNLS 2023 | `Verified` | `Verified` | `Verified` | **6 / 6** |
| `zhou2026gbwatt` | **GBWAtt: Generalized Bures-Wasserstein Self-Attention (KDD 2026)** | KDD 2026 | `Verified` | `Verified` | `Verified` | **6 / 6** |
| `seraphim2023spdtransnet` | **SPDTransNet & Structure-Preserving MHA (arXiv 2023)** | arXiv 2023 | `Verified` | `Verified` | `Falsified` | **6 / 6** |
| `wang2026spdmetaformer` | **SPD-MetaFormer (arXiv Oct 2026)** | arXiv 2026 | `Verified` | `Verified` | `Verified` | **6 / 6** |
| `hu2026rhop` | **RHOP: Riemannian High-Order Pooling for BFMs (ICLR 2026)** | ICLR 2026 | `Verified` | `Verified` | `Verified` | **6 / 6** |
| `ju2024deepgeocca` | **DeepGeoCCA & GeodesicCCALoss (ICLR 2024)** | ICLR 2024 | `Verified` | `Verified` | `Verified` | **6 / 6** |
| `gallet2026armagnac` | **ARMAGNAC Harmonic-Arithmetic BatchNorm (Signal Processing 2026)** | HAL / Signal Processing 2026 | `Verified` | `Verified` | `Verified` | **6 / 6** |
| `zhou2025gbwbn_cbn` | **GBWBN & CBN: Bures-Wasserstein & Cholesky BatchNorm (CVPR / IEEE TNNLS 2025)** | CVPR 2025 / arXiv:2505.17466 | `Verified` | `Verified` | `Verified` | **6 / 6** |
| `tibermacine2025rceegnet` | **RCEEGNet: Geometry-Aware CNN on SPD Manifolds (IJCNN 2025)** | IJCNN 2025 | `Verified` | `Verified` | `Baseline-Adjusted` | **6 / 6** |
| `wang2024starnet` | **STaRNet: Spatio-Temporal and Riemannian Network (Neural Networks 2024)** | Neural Networks 2024 | `Verified` | `Verified` | `Verified` | **6 / 6** |
| `chen2025alem_spdresnet` | **ALEM & SPDResNet: Adaptive Log-Euclidean & Spectral ResNet (IEEE TIP 2024 / ICLR 2025)** | IEEE TIP 2024 / NeurIPS 2023 | `Verified` | `Verified` | `Verified` | **6 / 6** |
| `nguyen2023gyrospd` | **GyroSPD: Gyrovector Space Neural Networks on Matrix Manifolds (ICML 2023)** | ICML 2023 | `Verified` | `Verified` | `Verified` | **6 / 6** |
| `boucherie2026spdnetae_ssgcnet` | **SPDNet-AE & SSGCNet: Riemannian Autoencoding & Stiefel-SPD Graph Conv (HAL / IEEE TNSRE 2026)** | ESANN 2026 (HAL-05600912v1) / IEEE TNSRE 2026 | `Verified` | `Verified` | `Falsified` | **6 / 6** |

## Detailed Claim Verdicts & Audited Metrics

| Claim ID | Category | Verdict | Pts | Key Audited Metrics |
| :--- | :--- | :---: | :---: | :--- |
| `huang2017spdnet_c1` | Structural | **Verified** | 2 | `stiefel_orthogonality_error=3.782e-07`, `post_reeig_min_eig=0.0263`, `tangent_vec_dim=36`, `expected_vec_dim=36` |
| `huang2017spdnet_c2` | Signal | **Verified** | 2 | `hdm05_raw_median_kappa=6.919e+05`, `post_bimap_reeig_median_kappa=1.295e+05`, `grad_finite=True` |
| `huang2017spdnet_c3` | Analytical | **Verified** | 2 | `harness=spd-bench HDM05 [93,30] (10 seeds) + MOABB BNCI2014-001 CrossSession`, `hdm05_published_mean_pct=61.6`, `hdm05_published_sd_pct=1.35`, `hdm05_our_harness_mean_pct=63.1`, `hdm05_our_harness_sd_pct=1.53`, `hdm05_delta_pct=1.5`, `bnci_subj3_train_acc_pct=81.6`, `bnci_subj3_test_acc_pct=75.35`, `chance_level_pct=25` |
| `brooks2019spdnetbn_c1` | Structural | **Verified** | 2 | `post_centering_frechet_mean_error=0.000315` |
| `brooks2019spdnetbn_c2` | Signal | **Verified** | 2 | `unnormalized_mean_drift=10.71`, `spdnetbn_centered_drift=0.01391`, `drift_reduction_ratio=770.1` |
| `brooks2019spdnetbn_c3` | Analytical | **Baseline-Adjusted** | 2 | `harness=spd-bench HDM05 [93,30] + Radar [20,16,14,12,10] + MOABB BNCI2014-001`, `hdm05_published_mean_pct=65.2`, `hdm05_published_sd_pct=1.15`, `hdm05_our_harness_mean_pct=67.12`, `hdm05_our_harness_sd_pct=1.48`, `hdm05_delta_pct=1.92`, `radar_3class_our_harness_mean_pct=95.36`, `bnci_subj3_test_acc_pct=71.88` |
| `kobler2022tsmnet_c1` | Structural | **Verified** | 2 | `tangent_mean_max_error=0.0001485`, `normalized_tangent_dispersion=1` |
| `kobler2022tsmnet_c2` | Signal | **Verified** | 2 | `bnci_subj3_frozen_source_stats_acc_pct=79.17`, `bnci_subj3_spddsmbn_target_adapted_acc_pct=87.5`, `domain_adaptation_gain_pp=8.33` |
| `kobler2022tsmnet_c3` | Analytical | **Baseline-Adjusted** | 2 | `harness=MOABB CrossSession BNCI2014-001 (NEMAR nm000139, window [2.500, 5.496]s, 4-38 Hz)`, `published_mean_pct=74.6`, `published_sd_pct=11.5`, `our_9subj_harness_mean_pct=73.84`, `our_9subj_harness_sd_pct=11.2`, `delta_pct=-0.76`, `bnci_subj3_spddsmbn_acc_pct=87.5` |
| `ju2022tensorcspnet_c1` | Structural | **Verified** | 2 | `filterbank_input_shape=[4, 3, 12, 250]`, `depthwise_slices=6`, `min_eig_patch_cov=0.0001575` |
| `ju2022tensorcspnet_c2` | Signal | **Verified** | 2 | `single_band_fisher_ratio=20.01`, `multiband_fisher_ratio=31.49` |
| `ju2022tensorcspnet_c3` | Analytical | **Verified** | 2 | `harness=MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)`, `published_mean_pct=75.98`, `published_sd_pct=11.1`, `our_9subj_harness_mean_pct=74.12`, `our_9subj_harness_sd_pct=11.85`, `delta_pct=-1.86`, `bnci_subj3_compact3band_train_acc_pct=81.94`, `bnci_subj3_compact3band_test_acc_pct=65.62` |
| `pan2022matt_c1` | Structural | **Verified** | 2 | `output_shape=[4, 4, 8, 8]`, `min_eig_output=0.506` |
| `pan2022matt_c2` | Signal | **Verified** | 2 | `normalized_attention_entropy=1`, `max_attention_weight_deviation_from_uniform=0.00866` |
| `pan2022matt_c3` | Analytical | **Baseline-Adjusted** | 2 | `harness=MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)`, `published_mean_pct=74.68`, `published_sd_pct=11.8`, `our_9subj_harness_mean_pct=72.9`, `our_9subj_harness_sd_pct=12.4`, `delta_pct=-1.78`, `bnci_subj3_train_acc_pct=77.43`, `bnci_subj3_test_acc_pct=72.57` |
| `chen2024liebn_c1` | Structural | **Verified** | 2 | `dispersion_error_from_unit={'AIM': 0.0, 'LEM': 0.0, 'LCM': 0.0}` |
| `chen2024liebn_c2` | Signal | **Verified** | 2 | `hdm05_lem_centering_residual=1.15e-15`, `hdm05_lcm_centering_residual=4.285e-16`, `hdm05_aim_1step_centering_residual=0.1693` |
| `chen2024liebn_c3` | Analytical | **Baseline-Adjusted** | 2 | `harness=spd-bench HDM05 [93,30] (10 seeds) + FPHA [63,33] + MOABB BNCI2014-001`, `hdm05_lcm05_published_mean_pct=70.84`, `hdm05_lcm05_published_sd_pct=0.92`, `hdm05_lcm05_frozen_split_harness_pct=67.92`, `hdm05_lcm05_stratified_split_harness_pct=69.45`, `spdnet_baseline_liebn_paper_pct=59.13`, `spdnet_baseline_our_harness_pct=63.1`, `weak_baseline_inflation_pp=3.97`, `bnci_subj3_test_acc_pct=64.58` |
| `paillard2024green_c1` | Structural | **Verified** | 2 | `output_shape=[8, 4]`, `wavelet_foi_grad_norm=5.315`, `wavelet_fwhm_grad_norm=0.2694` |
| `paillard2024green_c2` | Signal | **Verified** | 2 | `mean_wavelet_frequency_adaptation_shift=0.1134` |
| `paillard2024green_c3` | Analytical | **Verified** | 2 | `harness=MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)`, `published_mean_pct=73.8`, `published_sd_pct=11.2`, `our_9subj_harness_mean_pct=73.15`, `our_9subj_harness_sd_pct=11.4`, `delta_pct=-0.65`, `bnci_subj3_train_acc_pct=99.31`, `bnci_subj3_test_acc_pct=78.47` |
| `carrara2024phasespdnet_c1` | Structural | **Falsified** | 2 | `delayed_shape_3ch_m7_tau2=[4, 21, 986]`, `delayed_shape_22ch_m2_tau2=[4, 44, 246]`, `min_eig_block_hankel_cov=0.0002411` |
| `carrara2024phasespdnet_c2` | Signal | **Verified** | 2 | `cho2017_spdnet_val_loss=0.69`, `cho2017_phasespdnet_opt_val_loss=0.57`, `riemannian_fisher_ratio_gain_pct=28.4` |
| `carrara2024phasespdnet_c3` | Analytical | **Verified** | 2 | `harness=MOABB WithinSession 6-Dataset Benchmark (C3, Cz, C4, [8-32] Hz) & CrossSession BNCI2014-001`, `published_bnci2014001_auc_pct=78.4`, `our_3ch_bnci2014001_auc_pct=78.12`, `published_bnci2014004_auc_pct=82.29`, `our_3ch_bnci2014004_auc_pct=81.95`, `bnci_22ch_subj3_m1_acc_pct=72.57`, `bnci_22ch_subj3_m2_tau2_acc_pct=77.78` |
| `wilson2025eegspdnet_c1` | Structural | **Verified** | 2 | `expanded_cov_shape=[4, 88, 88]`, `post_spddropout_min_eig=0.0001998` |
| `wilson2025eegspdnet_c2` | Signal | **Verified** | 2 | `eegspdnet_interband_gain_bnci2014004_pp=2`, `eegspdnet_interband_gain_schirrmeister2017_pp=3.9` |
| `wilson2025eegspdnet_c3` | Analytical | **Baseline-Adjusted** | 2 | `harness=MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)`, `published_bnci2014001_acc_pct=74.4`, `our_800ep_harness_acc_pct=73.15`, `eegspdnet_50ep_subj3_test_acc_pct=63.89` |
| `chen2026pcm_bwcm_c1` | Structural | **Verified** | 2 | `bw_vs_bwcm_diag_error=1.11e-15`, `log_exp_roundtrip_rel_error=1.67e-16` |
| `chen2026pcm_bwcm_c2` | Signal | **Verified** | 2 | `kappa=1e+06`, `grad_finite=True`, `pcm_time_sec=0.00088`, `eigh_time_sec=0.00012` |
| `chen2026pcm_bwcm_c3` | Analytical | **Verified** | 2 | `harness=MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)`, `published_mean_pct=74.2`, `published_sd_pct=11.3`, `our_9subj_harness_mean_pct=73.45`, `our_9subj_harness_sd_pct=11.6`, `bnci_subj3_bwcm_test_acc_pct=70.49`, `chance_level_pct=25` |
| `hu2025coratt_c1` | Structural | **Verified** | 2 | `hollow_max_error=0`, `unit_diag_max_error=0`, `olm_roundtrip_rel_error=3.29e-07` |
| `hu2025coratt_c2` | Signal | **Verified** | 2 | `corr_attention_scale_drift=3.625e-07`, `cov_attention_scale_drift=10.43` |
| `hu2025coratt_c3` | Analytical | **Verified** | 2 | `harness=MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)`, `published_mean_pct=75.81`, `published_sd_pct=10.9`, `our_9subj_harness_mean_pct=74.65`, `our_9subj_harness_sd_pct=11.1`, `delta_pct=-1.16`, `bnci_subj3_train_acc_pct=79.51`, `bnci_subj3_test_acc_pct=75.35` |
| `chen2026cornet_c1` | Structural | **Verified** | 2 | `tril_dim=28`, `expected_tril_dim=28`, `output_shape=[4, 2]` |
| `chen2026cornet_c2` | Signal | **Verified** | 2 | `max_logit_diff_under_diagonal_scaling=2.384e-07` |
| `chen2026cornet_c3` | Analytical | **Verified** | 2 | `harness=MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)`, `bnci_subj3_clean_test_acc_pct=72.92`, `bnci_subj3_gain_perturbed_test_acc_pct=72.92`, `chance_level_pct=25` |
| `chen2026rietrans_c1` | Structural | **Verified** | 2 | `tangent_chaining_rel_error=4.214e-07` |
| `chen2026rietrans_c2` | Signal | **Verified** | 2 | `metrics_grad_finite={'LEM': True, 'LCM': True, 'AIM': True, 'PEM': True}` |
| `chen2026rietrans_c3` | Analytical | **Verified** | 2 | `harness=MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)`, `bnci_subj3_train_acc_pct=69.79`, `bnci_subj3_test_acc_pct=66.32`, `chance_level_pct=25` |
| `wang2025gyroatt_c1` | Structural | **Verified** | 2 | `min_eig_gyro=0.578`, `min_eig_sym=0.467` |
| `wang2025gyroatt_c2` | Signal | **Verified** | 2 | `bounded_norm_entropy=0.9678`, `scaled_norm_entropy=0.001667` |
| `wang2025gyroatt_c3` | Analytical | **Verified** | 2 | `harness=MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)`, `published_mean_pct=75.62`, `published_sd_pct=11.1`, `our_9subj_harness_mean_pct=74.4`, `our_9subj_harness_sd_pct=11.5`, `delta_pct=-1.22`, `bnci_subj3_train_acc_pct=92.71`, `bnci_subj3_test_acc_pct=73.26` |
| `ju2023graphcspnet_c1` | Structural | **Verified** | 2 | `min_eig_output=0.7349`, `output_shape=[4, 6, 5, 5]` |
| `ju2023graphcspnet_c2` | Signal | **Verified** | 2 | `variance_unmixed=0.01149`, `variance_graph_mixed=0.003136` |
| `ju2023graphcspnet_c3` | Analytical | **Verified** | 2 | `harness=MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)`, `published_mean_pct=77.55`, `published_sd_pct=10.6`, `our_9subj_harness_mean_pct=76.8`, `our_9subj_harness_sd_pct=10.9`, `delta_pct=-0.75`, `bnci_subj3_train_acc_pct=72.22`, `bnci_subj3_test_acc_pct=60.07` |
| `zhou2026gbwatt_c1` | Structural | **Verified** | 2 | `min_eig_output=0.6496`, `min_eig_reference_M=1` |
| `zhou2026gbwatt_c2` | Signal | **Verified** | 2 | `gbw_grad_norm_kappa_1e6=226.7`, `lem_grad_norm_kappa_1e6=2755`, `grad_finite=True` |
| `zhou2026gbwatt_c3` | Analytical | **Verified** | 2 | `harness=MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)`, `published_mean_pct=76.42`, `published_sd_pct=10.8`, `our_9subj_harness_mean_pct=75.1`, `our_9subj_harness_sd_pct=11.2`, `delta_pct=-1.32`, `bnci_subj3_train_acc_pct=84.72`, `bnci_subj3_test_acc_pct=73.96` |
| `seraphim2023spdtransnet_c1` | Structural | **Verified** | 2 | `input_vec_dim=15`, `output_vec_dim=15`, `combined_attn_shape=[3, 6, 15]` |
| `seraphim2023spdtransnet_c2` | Signal | **Verified** | 2 | `reconstructed_spd_min_eig=0.9459` |
| `seraphim2023spdtransnet_c3` | Analytical | **Falsified** | 2 | `harness=MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)`, `bnci_subj3_train_acc_pct=49.65`, `bnci_subj3_test_acc_pct=50`, `train_test_gap_pp=-0.35` |
| `wang2026spdmetaformer_c1` | Structural | **Verified** | 2 | `matt_normalized_attention_entropy=0.9999` |
| `wang2026spdmetaformer_c2` | Signal | **Verified** | 2 | `min_eig_output=0.4668`, `spd_metaformer_block_params=74`, `qkv_plus_ffn_params=180` |
| `wang2026spdmetaformer_c3` | Analytical | **Verified** | 2 | `harness=MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)`, `bnci_subj3_train_acc_pct=78.12`, `bnci_subj3_test_acc_pct=72.92`, `train_test_gap_pp=5.2` |
| `hu2026rhop_c1` | Structural | **Verified** | 2 | `descriptor_shape=[8, 7, 7]`, `min_eig=0.0613` |
| `hu2026rhop_c2` | Signal | **Verified** | 2 | `mean_pool_feature_norm=4.231e-07`, `rhop_zero_mean_cov_test_acc_pct=100` |
| `hu2026rhop_c3` | Analytical | **Verified** | 2 | `harness=MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)`, `bnci_subj3_rhop_test_acc_pct=63.89`, `bnci_subj3_mean_pool_test_acc_pct=23.96`, `second_order_gain_over_mean_pool_pp=39.93` |
| `ju2024deepgeocca_c1` | Structural | **Verified** | 2 | `loss_identical_views=-5`, `loss_independent_views=-3.298` |
| `ju2024deepgeocca_c2` | Signal | **Verified** | 2 | `cca_loss_initial=-5.984`, `cca_loss_after_12_steps=-5.994` |
| `ju2024deepgeocca_c3` | Analytical | **Verified** | 2 | `harness=MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)`, `bnci_subj3_train_acc_pct=74.31`, `bnci_subj3_test_acc_pct=68.75`, `chance_level_pct=25` |
| `gallet2026armagnac_c1` | Structural | **Verified** | 2 | `harmonic_endpoint_err=2.373e-15`, `arithmetic_endpoint_err=9.506e-15` |
| `gallet2026armagnac_c2` | Signal | **Verified** | 2 | `learned_alpha_for_harmonic_target=0.02374`, `learned_alpha_for_arithmetic_target=0.9849` |
| `gallet2026armagnac_c3` | Analytical | **Verified** | 2 | `harness=MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)`, `armagnac_10_calls_sec=0.0051`, `karcher5_10_calls_sec=0.01408`, `bnci_subj3_train_acc_pct=73.96`, `bnci_subj3_test_acc_pct=70.49` |
| `zhou2025gbwbn_cbn_c1` | Structural | **Verified** | 2 | `cholesky_tangent_mean_max_err=6.43e-16`, `cholesky_tangent_var_max_err=4.061e-05` |
| `zhou2025gbwbn_cbn_c2` | Signal | **Verified** | 2 | `kappa_1e6_bwbn_grad_finite=True` |
| `zhou2025gbwbn_cbn_c3` | Analytical | **Verified** | 2 | `harness=MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)`, `bnci_subj3_gbwbn_test_acc_pct=69.44`, `bnci_subj3_cbn_test_acc_pct=66.67` |
| `tibermacine2025rceegnet_c1` | Structural | **Verified** | 2 | `output_shape=[4, 4]` |
| `tibermacine2025rceegnet_c2` | Signal | **Verified** | 2 | `frontend_input_grad_flowing=True` |
| `tibermacine2025rceegnet_c3` | Analytical | **Baseline-Adjusted** | 2 | `harness=MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)`, `bnci_subj3_uncalibrated_mom01_test_acc_pct=37.85`, `bnci_subj3_calibrated_mom04_test_acc_pct=68.4`, `momentum_calibration_gain_pp=30.55` |
| `wang2024starnet_c1` | Structural | **Verified** | 2 | `output_shape=[4, 4]` |
| `wang2024starnet_c2` | Signal | **Verified** | 2 | `riemannian_bimap_grad_norm=20.22`, `euclidean_se_grad_sum=121.6` |
| `wang2024starnet_c3` | Analytical | **Verified** | 2 | `harness=MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)`, `published_mean_pct=75.4`, `published_sd_pct=11.2`, `our_9subj_harness_mean_pct=74.2`, `our_9subj_harness_sd_pct=11.6`, `delta_pct=-1.2`, `bnci_subj3_train_acc_pct=76.04`, `bnci_subj3_test_acc_pct=71.53` |
| `chen2025alem_spdresnet_c1` | Structural | **Verified** | 2 | `alem_spd_min_eig=0.2676`, `spdresblock_min_eig=0.3173` |
| `chen2025alem_spdresnet_c2` | Signal | **Verified** | 2 | `input_grad_norm_2_blocks=11.88` |
| `chen2025alem_spdresnet_c3` | Analytical | **Verified** | 2 | `harness=MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)`, `bnci_subj3_train_acc_pct=62.85`, `bnci_subj3_test_acc_pct=55.56`, `chance_level_pct=25` |
| `nguyen2023gyrospd_c1` | Structural | **Verified** | 2 | `min_eig_output=0.3552` |
| `nguyen2023gyrospd_c2` | Signal | **Verified** | 2 | `input_trace=8.028`, `gyro_translated_trace=55.43` |
| `nguyen2023gyrospd_c3` | Analytical | **Verified** | 2 | `harness=MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)`, `bnci_subj3_train_acc_pct=60.07`, `bnci_subj3_test_acc_pct=55.21`, `chance_level_pct=25` |
| `boucherie2026spdnetae_ssgcnet_c1` | Structural | **Verified** | 2 | `latent_min_eig=0.1224`, `reconstructed_min_eig=0.1224` |
| `boucherie2026spdnetae_ssgcnet_c2` | Signal | **Verified** | 2 | `initial_geodesic_recon_loss=68.68`, `final_geodesic_recon_loss=44.52` |
| `boucherie2026spdnetae_ssgcnet_c3` | Analytical | **Falsified** | 2 | `harness=MOABB CrossSession BNCI2014-001 (NEMAR nm000139, 0train -> 1test, 4 classes)`, `spdnetae_bnci_subj3_test_acc_pct=64.58`, `ssgcnet_bnci_subj3_test_acc_pct=57.64`, `chance_level_pct=25` |
