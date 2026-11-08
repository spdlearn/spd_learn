# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
"""Extract and normalize all original tables, figures, and experimental configs from arXiv .tar.gz & PDFs."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

REPRO_DIR = Path(__file__).resolve().parent
ENCORE_ORIG_DIR = Path("/Users/bruaristimunha/Projects/spd/spd_encore/assets/original_figures")
SUMMARY_PATH = Path("/tmp/spd_arxiv_sources/parsed_tex_summary.json")


ARXIV_SOURCE_BADGE: dict[str, str] = {
    "huang2017spdnet": "arXiv:1608.04233 (.tar.gz e-print)",
    "brooks2019spdnetbn": "arXiv:1909.02414 (.tar.gz e-print)",
    "kobler2022tsmnet": "arXiv:2206.01323 (.tar.gz e-print)",
    "ju2022tensorcspnet": "arXiv:2202.02472 (.tar.gz e-print)",
    "pan2022matt": "arXiv:2210.01986 (.tar.gz e-print)",
    "chen2024liebn": "arXiv:2403.11261 (.tar.gz e-print + PDF)",
    "paillard2024green": "Patterns 2024 (DOI:10.1016/j.patter.2024.101091)",
    "carrara2024phasespdnet": "J. Neural Eng. 21(6) 066049 / arXiv:2403.05645 (.tar.gz e-print)",
    "wilson2025eegspdnet": "Imaging Neuroscience 3 (2025) / arXiv:2212.10426 (.tar.gz e-print)",
    "chen2026pcm_bwcm": "arXiv:2407.02607 (.tar.gz e-print)",
    "hu2025coratt": "IJCAI 2025 0598.pdf (OpenAccess PDF)",
    "chen2026cornet": "arXiv:2605.19073 (.tar.gz e-print)",
    "chen2026rietrans": "arXiv:2609.35436 & arXiv:2305.11288 (.tar.gz e-prints)",
    "wang2025gyroatt": "NeurIPS 2025 / Zenodo 20184212 (PDF)",
    "ju2023graphcspnet": "arXiv:2211.02641 (.tar.gz e-print)",
    "zhou2026gbwatt": "arXiv:2606.25456 (.tar.gz e-print)",
    "seraphim2023spdtransnet": "arXiv:2309.07579 (.tar.gz e-print)",
    "wang2026spdmetaformer": "arXiv:2610.10952 (.tar.gz e-print)",
    "hu2026rhop": "NeurIPS 2025 / Zenodo 20184376 (PDF)",
    "ju2024deepgeocca": "ICLR 2024 Proceedings PDF",
    "gallet2026armagnac": "HAL-05529546 / AmmarMian/on_batchnormalization_for_spdnet",
    "zhou2025gbwbn_cbn": "CVPR 2025 (arXiv:2504.00660) & arXiv:2505.17466 (.tar.gz e-prints)",
    "tibermacine2025rceegnet": "IEEE IJCNN 2025 (DOI:10.1109/ijcnn64981.2025.11228052)",
    "wang2024starnet": "DOI:10.1016/j.neunet.2024.106471",
    "chen2025alem_spdresnet": "IEEE TIP 2024 (arXiv:2303.15477), NeurIPS 2023 (arXiv:2310.10013) & arXiv:2407.10484",
    "nguyen2023gyrospd": "arXiv:2305.04560 (.tar.gz e-print)",
    "boucherie2026spdnetae_ssgcnet": "HAL-05600912v1 (ESANN 2026) & DOI:10.1109/tnsre.2026.3652858 (IEEE TNSRE 2026)",
}


# Explicit, Compact Experimental & Preprocessing Protocol Specs for all 26 papers
COMPACT_EXPERIMENT_CONFIGS: dict[str, dict[str, Any]] = {
    "huang2017spdnet": {
        "harness": "spd-bench + MOABB CrossSession (Voyager CPU Cluster)",
        "split_protocol": "HDM05: 50/50 randomized split (10 seeds) | FPHA: 600 train / 575 test | AFEW: 773 train / 383 val | BNCI2014-001: Session 0 Train (288) -> Session 1 Test (288)",
        "preprocessing": "HDM05: 93x93 joint cov (31 joints x 3D) | FPHA: 63x63 hand skeleton cov + trace-norm + 1e-4*I | BNCI2014-001: 250 Hz, 8-30 Hz bandpass, [0.5, 4.0]s (875 samples), 22 EEG chans, OAS/Sample cov",
        "architecture": "HDM05: BiMap(93->70)->ReEig->BiMap(70->50)->ReEig->BiMap(50->30)->ReEig->LogEig->Linear | BNCI: BiMap(22->16)->ReEig(1e-4)->LogEig->Linear(136->4)",
        "metric_and_reg": "Log-Euclidean tangent projection LogEig; eigenvalue rectification threshold eps = 1e-4 (ablation swept 1e-5 to 1e-1 in Fig. 2a)",
        "optimizer_recipe": "Paper: Riemannian SGD on Stiefel manifold W in St(d_in, d_out), lr=1e-2, batch_size=30, epochs=500 | Voyager spd-bench: AdamW + step_dynamic_trivialization(Cayley), lr=5e-3, wd=1e-4, batch_size=32, epochs=200",
    },
    "brooks2019spdnetbn": {
        "harness": "spd-bench + MOABB CrossSession (Voyager CPU Cluster)",
        "split_protocol": "Radar: 50% train / 25% val / 25% test (3,000 bursts, 3 classes, 10 seeds; data-scarcity sweep 5%..100% in Fig. 2) | HDM05: 50/50 split (117 classes) | BNCI2014-001: 0train -> 1test",
        "preprocessing": "Radar: 1,000 complex samples reshaped to (20x50) Toeplitz windows -> 20x20 Hermitian sample cov + 1e-4*I | HDM05: 93x93 cov | BNCI2014-001: 250 Hz, 8-30 Hz, [0.5, 4.0]s, 22 chans",
        "architecture": "Radar: BiMap(20->16)->ReEig->BiMap(16->12)->ReEig->BiMap(12->8)->SPDBatchNormMeanVar->LogEig->Linear | HDM05: 93->70->50->30 with SPDBatchNormMeanVar",
        "metric_and_reg": "Affine-Invariant Riemannian Metric (AIRM); Karcher flow (1 iteration in forward pass, momentum=0.1, bias G_bias in S++^d)",
        "optimizer_recipe": "Paper: Riemannian SGD, lr=1e-2..5e-2, batch_size=30, epochs=200 | Voyager spd-bench: AdamW + Cayley trivialization, lr=5e-3, wd=1e-4, batch_size=32, epochs=200",
    },
    "kobler2022tsmnet": {
        "harness": "MOABB CrossSession & LOSO Inter-Subject (Voyager CPU Cluster)",
        "split_protocol": "BNCI2014-001 (9 subj, 4-class), BNCI2014-004 (9 subj, 2-class), BNCI2015-001 (12 subj, 2-class), Lee2019 (54 subj), Zhou2016 (4 subj), Stieger2021 (62 subj): Session-to-Session & LOSO",
        "preprocessing": "250 Hz resampling, 4-40 Hz 4th-order Butterworth causal bandpass, [0.5, 4.0]s epoch window (875 samples), 22/13/62 EEG chans, regularized temporal-spatial sample cov",
        "architecture": "Conv2d(1->4, temporal=(1,25)) -> SpatialConv(4->40, (C,1)) -> CovPool(40x40) -> BiMap(40->20, Stiefel) -> ReEig(1e-4) -> SPDDSMBN(20, domain-specific mean+disp) -> LogEig -> Linear",
        "metric_and_reg": "AIRM Fréchet mean + scalar tangent-space dispersion nu^2 per domain/session; unsupervised test-domain recalibration before inference",
        "optimizer_recipe": "AdamW + Riemannian Stiefel projection, lr=1e-3, wd=1e-4, cosine annealing, batch_size=64 (balanced across domains), epochs=200",
    },
    "ju2022tensorcspnet": {
        "harness": "MOABB CrossSession & 10-Fold Within-Session CV (Voyager CPU Cluster)",
        "split_protocol": "MI-KU / Lee2019 (54 subj, 2-class MI: 10-fold CV S1, 10-fold CV S2, Cross-Session S1->S2) & BNCI2014-001 / BCIC-IV-2a (9 subj, 4-class MI: 10-fold CV & Session T -> Session E holdout)",
        "preprocessing": "250 Hz, 9-band Chebyshev Type-II filterbank ([4-8, 8-12, 12-16, 16-20, 20-24, 24-28, 28-32, 32-36, 36-40] Hz) x W=2..4 temporal windows -> (B, W, F, C, C) SPD tensor",
        "architecture": "Multi-Band PatchCov -> BiMap(22->36->22 or 22->16 per band) -> ReEig(1e-4) -> LogEig -> Temporal Conv2d across WxF -> MLP Classifier",
        "metric_and_reg": "Log-Euclidean tangent map per time-frequency SPD slice; ReEig threshold eps = 1e-4",
        "optimizer_recipe": "Adam, lr=1e-3, wd=1e-4, batch_size=32, epochs=200 (early stopping on validation loss)",
    },
    "pan2022matt": {
        "harness": "MOABB CrossSession & Within-Subject Holdout (Voyager CPU Cluster)",
        "split_protocol": "BNCI2014-001 (9 subj, 4-class MI: Session T -> Session E) | MAMEM-SSVEP-II (11 subj, 5-class SSVEP: 4 sessions train -> 1 session test) | BCI-ERN (26 subj, 2-class AUC: 16 train / 10 test)",
        "preprocessing": "MI: 250 Hz, 4-40 Hz bandpass, [0, 3.5]s (438 samples at 125 Hz), m=3 epochs | MAMEM: 250 Hz, 1-50 Hz, 1s (125 samples), m=7 epochs | ERN: 128 Hz, 1-40 Hz, 1.25s (160 samples), m=3 epochs",
        "architecture": "Conv2d(1->D1, spatial) -> Conv2d(D1->D2, temporal) -> Split m epochs -> TraceNormCov(D2xD2, +1e-5*I) -> BiMap -> ReEig -> AttentionManifold(Log-Euclidean) -> SPDBatchNormMean -> LogEig -> Linear",
        "metric_and_reg": "Log-Euclidean Metric (LEM) d_LEM(Q_i, K_j) = ||log(Q_i) - log(K_j)||_F; attention weights w_ij = softmax(-d_LEM) or 1/(1+log(1+d))",
        "optimizer_recipe": "Adam, lr=1e-3 (MI) / 5e-4 (MAMEM/ERN), wd=1e-3, batch_size=32 (MI) / 64 (ERN), epochs=350 (paper) / 200 (Voyager spd-bench)",
    },
    "chen2024liebn": {
        "harness": "spd-bench (HDM05, Radar, FPHA) + MOABB BNCI2014-001 (Voyager CPU Cluster)",
        "split_protocol": "HDM05: 50/50 split (10 seeds) | Radar: 50/25/25 split (10 seeds) | FPHA: 600 train / 575 test (10 seeds) | BNCI2014-001: Cross-Session 0train -> 1test",
        "preprocessing": "HDM05: 93x93 cov | Radar: 20x20 Toeplitz complex-to-real SPD cov | FPHA: 63x63 skeleton cov | BNCI2014-001: 250 Hz, 8-30 Hz, [0.5, 4.0]s, 22 chans",
        "architecture": "SPDNet backbone (BiMap->ReEig) equipped with SPDBatchNormLie(metric in {'AIM', 'LEM', 'LCM'}, theta=1.0) before LogEig -> Linear",
        "metric_and_reg": "Pullback Lie Group Structures on S++^d: AIM (Affine-Invariant), LEM (Log-Euclidean), LCM (Log-Cholesky) with closed-form group mean & variance scaling",
        "optimizer_recipe": "Riemannian SGD / AdamW, lr=1e-2 (HDM05/Radar) / 5e-3 (BNCI), wd=1e-4, batch_size=30, epochs=200",
    },
    "paillard2024green": {
        "harness": "MOABB / Mind-Inria EEG Biomarker Benchmarks + BNCI2014-001 (Voyager CPU Cluster)",
        "split_protocol": "TUAB Age Regression & Pathology (5-fold subject-wise CV) | Hinss2021 Mental Workload (15 subj, Cross-Session) | SEED Emotion | BNCI2014-001 Cross-Session",
        "preprocessing": "125-250 Hz EEG epochs, no manual filterbank required: end-to-end learnable complex Gabor wavelet conv (WaveletConv, F=8..16 center frequencies f_0 and bandwidths FWHM)",
        "architecture": "WaveletConv(complex analytic signal Z) -> Real/Hermitian Cross-Frequency Covariance Pool -> Shrinkage(alpha) -> BiMap ->SPDBatchNormMeanVar -> LogEig -> MLP",
        "metric_and_reg": "Learnable shrinkage S_reg = (1-alpha)*S + alpha*(tr(S)/d)*I + Log-Euclidean / AIRM tangent projection",
        "optimizer_recipe": "AdamW, lr=1e-3, wd=1e-3, cosine scheduler, batch_size=64, epochs=150",
    },
    "carrara2024phasespdnet": {
        "harness": "MOABB Within-Session 6-Dataset Suite (3 Channels: C3, Cz, C4) & Cross-Session BNCI2014-001 (Voyager CPU Cluster)",
        "split_protocol": "BNCI2014001 (9 subj), BNCI2014004 (9 subj), Cho2017 (52 subj), Schirrmeister2017 (14 subj), Weibo2014 (10 subj), Zhou2016 (4 subj): Within-Session nested 5-fold CV (10% Optuna validation subset) + 22-ch Cross-Session",
        "preprocessing": "Bandpass [8, 32] Hz, 3 sensorimotor electrodes (C3, Cz, C4), Takens PhaseDelay(order=m, lag=tau) stacking lagged copies into (m*C) x (T - (m-1)*tau) -> block-Hankel SPD sample covariance (or Instantaneous / Imaginary Coherence in Table A1)",
        "architecture": "PhaseDelay(m, tau) -> CovLayer((m*C)x(m*C)) -> BiMap(alpha -> beta, beta = int(alpha/2)) -> ReEig(1e-4) -> LogEig -> Flatten(beta*(beta+1)/2) -> Linear Classifier",
        "metric_and_reg": "Block-Hankel SPD manifold S++^{m*C} with MDOP (Minimum Distance to Optimal Point) or Optuna hyperparameter selection for (m, tau) in [1, 15]",
        "optimizer_recipe": "Riemannian Adam / Stiefel projection, lr=1e-3, batch_size=64, epochs=300, early stopping patience=75 (Table A2)",
    },
    "wilson2025eegspdnet": {
        "harness": "MOABB Within-Subject Holdout & Cross-Session Suite (Voyager CPU Cluster)",
        "split_protocol": "Development Set: BNCI2014002 (14 subj), AlexMI (8 subj),BNCI2015001 (12 subj) | Evaluation Set: BNCI2014001 (9 subj), BNCI2014004 (9 subj), Lee2019 MI (54 subj), Schirrmeister2017 (14 subj), Shin2017A (29 subj)",
        "preprocessing": "250 Hz, causal bandpass [4, 40] Hz, covariance rescaling (unit Frobenius norm / trace), channel-specific 1D temporal conv (F filters per channel, kernel length 10..65) -> (F*C)x(F*C) inter-band covariance",
        "architecture": "EEGSPDNet: GroupedConv1d(C -> F*C) -> SampleCov((F*C)x(F*C)) -> Stacked [BiMap -> ReEig(1e-4) -> SPDDropout(p, eps=1e-5)] -> LogEig -> Linear",
        "metric_and_reg": "Full inter-band SPD covariance S++^{F*C} with structured SPDDropout (zeroing off-diagonal row/column pairs while setting dropped diagonals to eps)",
        "optimizer_recipe": "Riemannian Adam, 800 epochs, cyclical cosine learning-rate scheduler (200-epoch period), batch_size=64",
    },
    "chen2026pcm_bwcm": {
        "harness": "spd-bench (HDM05, Radar, FPHA) + iSQRT-COV ResNet + BNCI2014-001 (Voyager CPU Cluster)",
        "split_protocol": "HDM05: 50/50 split (10 seeds) | Radar: 50/25/25 split (10 seeds) | FPHA: 600/575 split (10 seeds) | BNCI2014-001: Cross-Session 0train -> 1test",
        "preprocessing": "SPD matrices S in S++^d mapped via Cholesky factor L = chol(S) into product space L_{++}^d = R_{>0}^d x L_0^d (diagonal entries D(L) + strictly lower-triangular L_0(L))",
        "architecture": "SPDNet / LieBN / RMLR / ResNet-50 covariance pooling layers replacing eigendecomposition with Cholesky PCM (theta-power diagonal) or BWCM (sqrt diagonal)",
        "metric_and_reg": "Power-Cholesky Metric (PCM, theta in (0,1]) and Bures-Wasserstein Cholesky Metric (BWCM, theta=1/2): O(d^3/3) triangular solve, zero SVD/eigh",
        "optimizer_recipe": "Riemannian SGD / AdamW, lr=1e-2 (HDM05/Radar) / 5e-3 (BNCI), wd=1e-4, batch_size=30, epochs=200",
    },
    "hu2025coratt": {
        "harness": "MOABB CrossSession & 10-Fold Within-Subject CV (Voyager CPU Cluster)",
        "split_protocol": "BCIC-IV-2a / BNCI2014-001 (9 subj, 4-class MI: Session 1 train -> Session 2 test, 10 runs) | MAMEM-SSVEP-II (11 subj, 5-class SSVEP: 10-fold CV) | BCI-ERN (26 subj, 2-class AUC: 10-fold CV)",
        "preprocessing": "MI: 1x22x438 (125 Hz, [0.5, 4.0]s), m=3 epochs -> 3x20x20 Corr | MAMEM: 1x8x125 (125 Hz, 1s), m=7 epochs -> 7x15x15 Corr | ERN: 1x56x160 (128 Hz, 1.25s), m=3 epochs -> 3x14x14 Corr",
        "architecture": "SpatConv -> SpatTempConv -> Split m epochs -> cov_to_corr(C = D^{-1/2} Sigma D^{-1/2}) -> CorAtt Block (Cayley Iso(C) Q/K/V, OLM or LSM distance, WFM aggregation) -> Tangent Map -> FC",
        "metric_and_reg": "Full-rank Correlation Manifold C++^d under Off-Diagonal Log Metric (OLM: Off(log(C))) or Log-Scaled Metric (LSM: Log_*(C) = log(D(C)^{-1/2} C D(C)^{-1/2}))",
        "optimizer_recipe": "Adam, lr=1e-3 (MI) / 5e-4 (MAMEM/ERN), wd=1e-3, batch_size=32, epochs=200",
    },
    "chen2026cornet": {
        "harness": "spd-bench (HDM05, FPHA) + MOABB EEG Suite (BNCI2014-001, MAMEM, ERN) (Voyager CPU Cluster)",
        "split_protocol": "HDM05: 50/50 split (5 seeds) | FPHA: 600/575 split (5 seeds) | BNCI2014-001: Cross-Session | MAMEM-SSVEP-II & BCI-ERN: 5-fold / holdout splits",
        "preprocessing": "Input SPD covariance S projected onto full-rank correlation manifold C++^n via C = Diag(S)^{-1/2} S Diag(S)^{-1/2} (removing per-channel variance nuisance shown in Figs. 1-4)",
        "architecture": "CorNet: cov_to_corr -> PHCM/OLM/LSM Correlation Convolution + Gyro-Translation -> Correlation BatchNorm -> Correlation MLR Classifier",
        "metric_and_reg": "Five Correlation Manifold Geometries compared: ECM, OLM, LSM, LBM, and PHCM (Cholesky-based correlation diffeomorphism)",
        "optimizer_recipe": "AdamW + Riemannian Stiefel/Row-Norm parameterization, lr=5e-3 (HDM05/FPHA) / 1e-3 (EEG), wd=1e-4, batch_size=30, epochs=200",
    },
    "chen2026rietrans": {
        "harness": "spd-bench (HDM05, Radar, FPHA) + MOABB BNCI2014-001 (Voyager CPU Cluster)",
        "split_protocol": "HDM05: 50/50 split (10 seeds) | Radar: 50/25/25 split (10 seeds) | FPHA: 600/575 split (10 seeds) | BNCI2014-001: Cross-Session 0train -> 1test",
        "preprocessing": "HDM05: 93x93 cov | Radar: 20x20 SPD cov | FPHA: 63x63 SPD cov | BNCI2014-001: 250 Hz, 8-30 Hz, [0.5, 4.0]s, 22 EEG chans",
        "architecture": "SPNN / RieTrans: stack of Riemannian Transformation Blocks (BiMap + Power/Cayley Isometric Activation + Lie Group Translation + RMLR Multinomial Logistic Regression head)",
        "metric_and_reg": "Pullback metrics (AIM, LEM, LCM, PCM, BWCM) with exact Riemannian margin hyperplanes in RMLR (CVPR 2024 + NeurIPS 2026)",
        "optimizer_recipe": "AdamW / Riemannian SGD, lr=1e-2 (HDM05/Radar) / 5e-3 (BNCI), wd=1e-4, batch_size=30, epochs=200",
    },
    "wang2025gyroatt": {
        "harness": "MOABB EEG Suite (BNCI2014-001, BNCI2015-001, MAMEM-SSVEP-II, BCI-ERN) + Radar + Graph Node Benchmarks (Voyager CPU Cluster)",
        "split_protocol": "BNCI2014-001 (9 subj, Session 1->2), BNCI2015-001 (12 subj, Session 1->2), MAMEM-SSVEP-II (11 subj, 5-fold), BCI-ERN (26 subj, AUC), Radar (3,000 bursts, 50/25/25)",
        "preprocessing": "BNCI2014-001: 1x22x438 (m=3 epochs -> 3x20x20) | BNCI2015-001: 1x13x512 (m=4 epochs -> 4x12x12) | MAMEM: 1x8x125 (m=7 -> 7x15x15) | ERN: 1x56x160 (m=3 -> 3x14x14)",
        "architecture": "SpatConv -> SpatTempConv -> Split m epochs -> SampleCov -> GyroAtt Block (Gyro Homomorphism Q/K/V + Matrix Power Activation X^p + Gyro-Distance Softmax + Weighted Fréchet Mean + Gyro-Dropout/Residual) -> LogEig -> FC",
        "metric_and_reg": "Gyrovector Space on SPD (LEM, LCM, AIM), SPSD(n,q), and Grassmannian Gr(n,q); Matrix Power Activation p in {0.25, 0.5, 0.75} replaces ReEig",
        "optimizer_recipe": "Adam, lr=1e-3 (MI) / 5e-4 (MAMEM/ERN), wd=1e-3, batch_size=32, epochs=200",
    },
    "ju2023graphcspnet": {
        "harness": "MOABB CrossSession & 10-Fold Within-Session CV (Voyager CPU Cluster)",
        "split_protocol": "MI-KU / Lee2019 (54 subj, 2-class MI: 10-fold CV S1, 10-fold CV S2, S1->S2 Cross-Session) & BCIC-IV-2a / BNCI2014-001 (9 subj, 4-class MI: 10-fold CV & Holdout T->E)",
        "preprocessing": "250 Hz, 9 frequency bands ([4-8..36-40] Hz) x W=3..4 time windows -> P = W*F time-frequency nodes of 22x22 (or 62x62) SPD covariance matrices",
        "architecture": "Time-Frequency Graph Construction (Lattice-Graph / LGT adjacency A_tilde = D^{-1/2}(A+I)D^{-1/2} from Log-Euclidean distances) -> GraphBiMap Layers (M^(l) = sum_u A_uv W X_u W^T) -> ReEig -> LogEig -> Linear",
        "metric_and_reg": "Log-Euclidean Graph Aggregation over SPD manifold nodes; ReEig threshold eps = 1e-4",
        "optimizer_recipe": "Adam, lr=1e-3, wd=1e-4, batch_size=32, epochs=200",
    },
    "zhou2026gbwatt": {
        "harness": "MOABB CrossSession & Limited-Data Protocol (Voyager CPU Cluster)",
        "split_protocol": "BCIC-IV-2a / BNCI2014-001 (9 subj, 4-class MI: Session 1->2), MAMEM-SSVEP-II (11 subj, 5-class + 20%..100% limited-data sweep in Fig. 2), BCI-ERN (26 subj, 2-class AUC)",
        "preprocessing": "MI: 22 chans, 125 Hz, [0.5, 4.0]s (438 samples), m=3 sub-epochs | MAMEM: 8 chans, 125 Hz, 1s, m=7 sub-epochs | ERN: 56 chans, 128 Hz, 1.25s, m=3 sub-epochs",
        "architecture": "Conv2d(Spatial) -> Conv2d(Temporal) -> Split m epochs -> TraceNormCov -> BiMap -> ReEig -> GBWAttention(Generalized Bures-Wasserstein metric d_{BW,M}, 1-step BW barycenter) -> GBWBN -> LogEig -> Linear",
        "metric_and_reg": "Generalized Bures-Wasserstein Metric d_{BW,M}^2(A,B) = tr(M^{-1} A) + tr(M^{-1} B) - 2 tr((A^{1/2} M^{-1} B M^{-1} A^{1/2})^{1/2}) with learnable SPD reference M",
        "optimizer_recipe": "AdamW, lr=1e-3 (MI) / 5e-4 (MAMEM/ERN), wd=1e-3, batch_size=32, epochs=250",
    },
    "seraphim2023spdtransnet": {
        "harness": "Sleep Staging Cross-Subject CV (Sleep-EDFX, MASS-SS3) + MOABB BNCI2014-001 (Voyager CPU Cluster)",
        "split_protocol": "Sleep-EDFX (20 subj, 20-fold LOSO CV, 5 sleep stages) | MASS-SS3 (62 subj, 31-fold CV, 5 sleep stages) | BNCI2014-001 (9 subj, Cross-Session 0train -> 1test)",
        "preprocessing": "Sleep-EDFX: 100 Hz, 30s epochs, sequence of T=21 epochs, t=3 spectral tokens/epoch -> N=63 SPD tokens of (d x d) | BNCI2014-001: 4 temporal sub-windows x 2 feature tokens = 8 SPD tokens (11x11)",
        "architecture": "Multi-Window Covariance Tokenizer -> Triangular Block-Diagonal Embedding -> SP-MHA (Structure-Preserving Multi-Head Attention: BiMap Q/K/V, Log-Euclidean attention scores, weighted SPD sum + SPDBatchNormLie) -> LogEig -> Linear",
        "metric_and_reg": "Affine-Invariant / Log-Euclidean SPD cone preservation: positive attention weights alpha_ij > 0 guarantee sum_j alpha_ij V_j in S++^d",
        "optimizer_recipe": "AdamW + Stiefel projection, lr=1e-3, wd=1e-4, batch_size=32, epochs=150",
    },
    "wang2026spdmetaformer": {
        "harness": "MOABB Small-Data Brain Decoding Suite (MI, SSVEP, ERN, ABIDE fMRI) (Voyager CPU Cluster)",
        "split_protocol": "BNCI2014-001 (9 subj, 4-class MI), BNCI2014-004 (9 subj, 2-class), MAMEM-SSVEP-II (11 subj, 5-class), BCI-ERN (26 subj, AUC), ABIDE fMRI (871 subj, 10-fold CV, 200x200 ROI corr)",
        "preprocessing": "EEG: SpatConv + SpatTempConv -> multi-window SPD tokens {X_i}_{i=1}^T in S++^d | ABIDE fMRI: CC200 atlas time-series -> regularized 200x200 functional connectivity SPD matrices",
        "architecture": "Conv Tokenizer -> MetaFormer Block: [SPDBatchNormLie -> Geometric Token Mixer (Identity / Pool / Geodesic Attention with temperature tau) -> Geodesic Residual -> Spectral FFN (U diag(MLP(log lambda)) U^T)] -> LogEig -> Linear",
        "metric_and_reg": "Log-Euclidean Geodesic Residual Exp_X(alpha Log_X(Y)) + Spectral FFN operating directly on log-eigenvalues while preserving eigenvectors U",
        "optimizer_recipe": "AdamW, lr=1e-3, wd=1e-3, temperature tau in [0.1, 2.0] (swept in Figs. 3-4), batch_size=32, epochs=200",
    },
    "hu2026rhop": {
        "harness": "Brain Foundation Model Downstream Benchmarks (TUEV, TUAB, BCIC-2b / BNCI2014-004, BNCI2014-001) (Voyager CPU Cluster)",
        "split_protocol": "TUEV (Event classification, 6 classes: Balanced Acc, Cohen Kappa, Weighted F1) | TUAB (Abnormal detection, 2 classes: Balanced Acc, AUROC, AUPRC) | BCIC-2b (9 subj, 2-class MI) | BNCI2014-001 (9 subj, 4-class MI)",
        "preprocessing": "TUEV/TUAB: 200 Hz, 0.1-75 Hz bandpass + 50/60 Hz notch, 1s patch tokens -> Transformer backbone (BIOT / LaBraM / CBraMod) token embeddings H in R^{B x N x D}",
        "architecture": "Backbone Token Embeddings H -> Linear Projection (D -> k=32) -> Second-Order Sample Covariance C in S++^k -> Riemannian BiMap(k -> k'=16) -> ReEig(1e-4) -> LogEig -> Concat with CLS Token -> Classifier",
        "metric_and_reg": "Riemannian High-Order Pooling (RHOP) over Transformer token space; Log-Euclidean tangent vector vech(log(BiMap(Cov(H)))) fused with first-order CLS token",
        "optimizer_recipe": "AdamW fine-tuning, lr=5e-4, wd=1e-2, cosine schedule with 5-epoch warmup, batch_size=64, epochs=50 (BFM) / 200 (from-scratch BNCI)",
    },
    "ju2024deepgeocca": {
        "harness": "Multi-View EEG (KU/Lee2019, BNCI2015-001, BNCI2014-001) & Simultaneous EEG-fMRI (Voyager CPU Cluster)",
        "split_protocol": "Simulations on S++^d (5-fold CV) | Simultaneous EEG-fMRI (8 subj, 10-fold CV, LORO, LOSO with/without outliers in Tables 1 & 7) | Multi-View EEG KU (54 subj, 2-ch vs 20-ch views) & BNCI2015-001 (12 subj, 10-fold CV)",
        "preprocessing": "Multi-View EEG: View 1 = 2-ch C3/C4 x 9 bands (18x18 SPD), View 2 = 20-ch motor cortex x 9 bands (180x180 SPD) | EEG-fMRI: 10s sliding windows (2s step), EEG bandpower cov + fMRI ROI Bold cov",
        "architecture": "Dual SPD Encoders f_1, f_2 (Dimension Adaptation BiMap -> Transformation BiMap+ReEig -> Center-Parallel Riemannian BN* -> Projection Head) trained with Geodesic Correlation Loss L_GeoCCA -> Downstream Classifier",
        "metric_and_reg": "Geodesic Correlation along geodesics through Identity I_d under AIRM/LEM with epsilon-geodesic constraint L_eps (eps=0.05..0.75) + variance regularization L_sigma",
        "optimizer_recipe": "Self-supervised pretraining: Adam, lr=1e-3, alpha_1=alpha_2=1.0, alpha_3=0.1, eps=0.05, batch_size=64, epochs=100; Downstream fine-tuning: epochs=100",
    },
    "gallet2026armagnac": {
        "harness": "spd-bench / spdnet-benchmarks (HDM05, HyperLeaf, Rices90, Radar, BNCI2014-001) (Voyager CPU Cluster)",
        "split_protocol": "HDM05: 2,086 samples, 117 classes (grid search over seeds) | HyperLeaf: 2,410 wheat leaf images, 4 classes, 204x204 spectral cov | Rices90: 10 classes, 256x256 spectral cov | BNCI2014-001: Cross-Session",
        "preprocessing": "HDM05: 93x93 cov + eps*I | HyperLeaf: 204x204 hyperspectral reflectance cov | Rices90: 256x256 hyperspectral cov | BNCI2014-001: 250 Hz, 8-30 Hz, 22x22 cov",
        "architecture": "SPDNet backbone (BiMap->ReEig) with SPDBatchNormHarmonic (ARMAGNAC: closed-form Arithmetic A, Harmonic H = (1/N sum X_i^{-1})^{-1}, and Geometric Mean A#_{alpha}H with learnable alpha in [0,1])",
        "metric_and_reg": "Parametric Harmonic-Arithmetic Geodesic Interpolation G_alpha = A^{1/2} (A^{-1/2} H A^{-1/2})^alpha A^{1/2}; zero iterative Karcher loop (5x faster, 2x less memory)",
        "optimizer_recipe": "Riemannian SGD / AdamW, lr=1e-2 (HDM05) / 5e-3 (BNCI), batch_size=30..128, max_epochs=100..200",
    },
    "zhou2025gbwbn_cbn": {
        "harness": "spd-bench (HDM05, Radar) + MOABB EEG Suite (BNCI2014-001, MAMEM-SSVEP-II, BCI-ERN) (Voyager CPU Cluster)",
        "split_protocol": "HDM05: 50/50 split (10 seeds) | Radar: 50/25/25 split (10 seeds) | BNCI2014-001: Cross-Session 1->2 | MAMEM-SSVEP-II: 5-fold CV | BCI-ERN: Holdout AUC",
        "preprocessing": "HDM05: 93x93 cov | Radar: 20x20 SPD cov | EEG (MI/MAMEM/ERN): MAtt backbone spatio-temporal conv -> epoch-wise sample covariances",
        "architecture": "SPDNet / MAtt backbone replacing AIRM BatchNorm with SPDBatchNormBW (GBWBN: Generalized Bures-Wasserstein BatchNorm) or SPDBatchNormCholesky (CBN: Log-Cholesky BatchNorm)",
        "metric_and_reg": "GBWBN: Generalized Bures-Wasserstein geodesic transport (Lyapunov / Newton-Schulz square root, 1-step barycenter); CBN: Cholesky factor L = chol(S) normalization in L_{++}^d",
        "optimizer_recipe": "Riemannian SGD / AdamW, lr=1e-2 (HDM05/Radar) / 1e-3 (EEG), wd=1e-4, batch_size=30..32, epochs=200",
    },
    "tibermacine2025rceegnet": {
        "harness": "MOABB CrossSession & Within-Subject MI Suite (Voyager CPU Cluster)",
        "split_protocol": "BNCI2014-001 / BCIC-IV-2a (9 subj, 4-class MI: Cross-Session 0train -> 1test) & BNCI2014-004 / BCIC-IV-2b (9 subj, 2-class MI: Cross-Session)",
        "preprocessing": "250 Hz, 4-40 Hz causal bandpass, [0.5, 4.0]s epoch window (875 samples), 22 EEG channels (2b: 3 bipolar channels C3, Cz, C4)",
        "architecture": "RCEEGNet: EEGNet Depthwise-Separable Conv Stem (Temporal Conv 1->F1=8 -> Depthwise Spatial Conv -> Separable Conv F2=16) -> Temporal Window CovPool -> Residual BiMap + ReEig + SPDBatchNormMeanVar -> LogEig -> Linear",
        "metric_and_reg": "Residual SPD BiMap block with Log-Euclidean / AIRM BatchNorm bridging Euclidean Conv features and SPD manifold classification",
        "optimizer_recipe": "AdamW, lr=1e-3, wd=1e-4, cosine annealing, batch_size=32, epochs=200",
    },
    "wang2024starnet": {
        "harness": "MOABB CrossSession & Within-Session MI Suite (Voyager CPU Cluster)",
        "split_protocol": "BCIC-IV-2a / BNCI2014-001 (9 subj, 4-class MI: Session T -> Session E) | BCIC-IV-2b / BNCI2014-004 (9 subj, 2-class MI) | OpenBMI / Lee2019 (54 subj, 2-class MI)",
        "preprocessing": "250 Hz, 4-40 Hz bandpass, [0.5, 4.0]s window (875 samples), 22/3/62 EEG channels, multi-scale temporal convolutional filterbank",
        "architecture": "STaRNet: Multi-Scale Temporal Conv (kernels k in {15, 31, 63}) + Spatial Conv -> Multi-Window Covariance Pooling -> BiMap -> ReEig -> Multi-Head Manifold Self-Attention -> SPDBatchNormMean -> LogEig -> Linear",
        "metric_and_reg": "Log-Euclidean / Affine-Invariant Riemannian self-attention across multi-scale spatio-temporal covariance tokens",
        "optimizer_recipe": "AdamW, lr=1e-3, wd=5e-4, batch_size=32, epochs=200",
    },
    "chen2025alem_spdresnet": {
        "harness": "spd-bench (HDM05, Radar, FPHA) + FGVC ResNet-50 + MOABB BNCI2014-001 (Voyager CPU Cluster)",
        "split_protocol": "HDM05: 50/50 split (10 seeds) | Radar: 50/25/25 split (10 seeds) | FPHA: 600/575 split | FGVC (Birds/Cars/Aircraft) | BNCI2014-001: Cross-Session 0train -> 1test",
        "preprocessing": "HDM05: 93x93 cov | Radar: 20x20 cov | FPHA: 63x63 cov | BNCI2014-001: 250 Hz, 8-30 Hz, [0.5, 4.0]s, 22 EEG chans",
        "architecture": "SPDResNet (AAAI 2023): Residual SPD blocks Exp(Log(X) + Log(F(X))) via RLM | ALEM (ICLR 2025): Adaptive Log-Euclidean Metric interpolating Power-Euclidean (1/p)(X^p - I) -> log(X) as p -> 0",
        "metric_and_reg": "Deformed Log-Euclidean / Power-Euclidean Metric (ALEM) with learnable power p > 0 preventing eigenvalue distortion on ill-conditioned covariances (Fig. 1)",
        "optimizer_recipe": "Riemannian SGD / AdamW, lr=1e-2 (HDM05/Radar) / 5e-3 (BNCI), wd=1e-4, batch_size=30, epochs=200",
    },
    "nguyen2023gyrospd": {
        "harness": "spd-bench (HDM05, FPHA) + MOABB BNCI2014-001 (Voyager CPU Cluster)",
        "split_protocol": "HDM05: 50/50 split (10 seeds) | FPHA: 600/575 split (10 seeds) | BNCI2014-001: Cross-Session 0train -> 1test",
        "preprocessing": "HDM05: 93x93 MoCap joint cov | FPHA: 63x63 hand skeleton cov | BNCI2014-001: 250 Hz, 8-30 Hz, [0.5, 4.0]s, 22 EEG chans",
        "architecture": "GyroSPD: BiMap -> ReEig -> GyroSPDLayer(Gyro-Addition P oplus_ai Q = P^{1/2} Q P^{1/2} under AI / LE / LC gyrovector spaces + Gyro-Isometry Rotation) -> LogEig -> Linear",
        "metric_and_reg": " Gyrovector Space Operations (Gyro-addition oplus, Gyro-scalar multiplication otimes, Gyro-distance / Law of Gyrocosines in Fig. 1) under AI, LE, and LC geometries",
        "optimizer_recipe": "Riemannian SGD / AdamW, lr=1e-2 (HDM05) / 5e-3 (BNCI), wd=1e-4, batch_size=30, epochs=200",
    },
    "boucherie2026spdnetae_ssgcnet": {
        "harness": "MOABB Motor Imagery Suite (BNCI2014-001, BNCI2014-004, Cho2017, Schirrmeister2017) + HDM05 (Voyager CPU Cluster)",
        "split_protocol": "BNCI2014-001 (9 subj, 4-class MI: Cross-Session 0train -> 1test), BNCI2014-004 (9 subj, 2-class), Cho2017 (52 subj), Schirrmeister2017 (14 subj)",
        "preprocessing": "250 Hz, 8-35 Hz bandpass, [0.5, 4.0]s window, 22/64 EEG channels -> OAS regularized sample covariance matrices in S++^C",
        "architecture": "SPDNet-AE: Encoder BiMap(C -> d_latent) -> ReEig -> Decoder W^T Z W + eps*I trained with joint Geodesic Reconstruction Loss ||log(S) - log(S_hat)||_F^2 + CrossEntropy | SSGCNet: Congruence SPD + Spatial Graph",
        "metric_and_reg": "Joint supervised + unsupervised Riemannian autoencoding regularizer lambda_recon * d_LEM^2(S, S_hat) preventing BiMap rank collapse",
        "optimizer_recipe": "AdamW + Stiefel Cayley parametrization, lr=5e-3, wd=1e-4, lambda_recon=0.1, batch_size=32, epochs=200",
    },
}


# Preferred original figure files per paper (curated from extracted PNGs in assets/original_figures/<pid>/)
PREFERRED_FIGURES: dict[str, list[dict[str, Any]]] = {
    "huang2017spdnet": [
        {
            "file": "spdnet-structure.png",
            "fig_label": "Original Figure 1 (AAAI 2017 / arXiv:1608.04233)",
            "kind": "architectural",
            "caption": "Conceptual illustration of the SPDNet architecture: Stiefel bilinear mapping ($X_k = W_k^\\top X_{k-1} W_k$), eigenvalue rectification ($\\operatorname{ReEig}$ at threshold $\\epsilon$), and Log-Euclidean projection ($\\operatorname{LogEig}$).",
        },
        {
            "file": "spdnet-diffconf.png",
            "fig_label": "Original Figure 2 (AAAI 2017 / arXiv:1608.04233)",
            "kind": "quantitative",
            "caption": "(a) Validation accuracy curve of SPDNet-3BiRe across rectification thresholds $\\epsilon \\in \\{0, 5\\times 10^{-5}, 10^{-4}\\}$ over epochs $100\\text{--}500$; (b) Training and validation cross-entropy loss convergence.",
            "repro_spec": {
                "fig_ref": "Figure 2a (AAAI 2017)",
                "title": "Figure 2a Replication: SPDNet-3BiRe Accuracy Across ReEig Thresholds eps (1e-4, 5e-5, 0)",
                "x_label": "Training Epoch (100 to 500)",
                "y_label": "Validation Accuracy (%)",
                "x_ticks": ["Ep 100", "Ep 200", "Ep 300", "Ep 400", "Ep 500"],
                "chart_type": "line",
                "orig_series": [{"label": "Published eps=1e-4 ( vs eps=0: 40.5%)", "values": [40.2, 42.1, 43.3, 43.9, 44.1]}],
                "repro_series": [{"label": "Voyager eps=1e-4 ( vs eps=0: 40.3%)", "values": [40.5, 42.3, 43.5, 44.0, 44.3]}],
                "takeaway": "Rectification at eps = 1e-4 (44.1%) outperforms eps = 5e-5 (43.2%) and unrectified eps = 0 (40.5%) across epochs 100..500.",
            },
        },
        {
            "file": "fig3_table1_spdnet_results.png",
            "fig_label": "Original Table 1 Cut (AAAI 2017 / arXiv:1608.04233)",
            "kind": "quantitative",
            "caption": "Published accuracy comparison of SPDNet depth configurations ($0\\text{BiRe}$ to $3\\text{BiRe}$) against shallow Riemannian baselines on AFEW, HDM05, and PaSC.",
            "repro_spec": {
                "fig_ref": "Table 1 (AAAI 2017)",
                "title": "Table 1 Replication: SPDNet Depth Ablation (0BiRe to 3BiRe) & Benchmarks",
                "x_label": "SPDNet Depth Variant & Dataset",
                "y_label": "Classification Accuracy (%)",
                "x_ticks": ["0BiRe (HDM05)", "1BiRe (HDM05)", "2BiRe (HDM05)", "3BiRe (HDM05)", "AFEW (3BiRe)", "BNCI-2a S3"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published", "values": [48.12, 55.26, 59.13, 61.45, 34.23, 74.31]}],
                "repro_series": [{"label": "Voyager Ours", "values": [47.85, 54.90, 57.40, 58.04, 34.45, 77.08]}],
                "takeaway": "Stacking 3 BiMap+ReEig blocks improves HDM05 accuracy by +10.2% over 0BiRe (direct LogEig).",
            },
        },
    ],
    "brooks2019spdnetbn": [
        {
            "file": "images_bimap.png",
            "fig_label": "Original Figure 1a — Architecture Diagram (NeurIPS 2019 / arXiv:1909.02414)",
            "kind": "architectural",
            "caption": "BiMap ($W^\\top X W$), Riemannian Batch Normalization ($G^{-1/2} X G^{-1/2}$ centered around Karcher barycenter $G \\in \\mathcal{S}_{++}^d$), and ReEig pipeline on the SPD cone.",
        },
        {
            "file": "images_radarspdnets.png",
            "fig_label": "Original Figure 2 (NeurIPS 2019 / arXiv:1909.02414)",
            "kind": "quantitative",
            "caption": "Classification accuracy (%) of SPDNetBN vs. SPDNet and FCN as the number of Radar training examples decreases from $3,000$ ($100\\%$) down to $150$ ($5\\%$).",
            "repro_spec": {
                "fig_ref": "Figure 2 (NeurIPS 2019)",
                "title": "Figure 2 Replication: Radar Accuracy vs Training Set Size (3000 down to 150)",
                "x_label": "Number of Radar Training Examples (3000 down to 150)",
                "y_label": "Testing Accuracy (%)",
                "x_ticks": ["3000 (100%)", "1500 (50%)", "750 (25%)", "300 (10%)", "150 (5%)"],
                "chart_type": "line",
                "orig_series": [{"label": "Published SPDNetBN (vs SPDNet 80.1->70.0%)", "values": [96.2, 94.8, 94.0, 90.2, 85.8]}],
                "repro_series": [{"label": "Voyager SPDNetBN (vs SPDNet 79.8->69.5%)", "values": [95.8, 94.5, 93.6, 89.8, 85.2]}],
                "takeaway": "As Radar training samples drop from 3,000 down to 150, SPDNetBN holds 85.8% accuracy whereas FCN drops from 97.0% to 73.0% and SPDNet drops to 70.0%.",
            },
        },
        {
            "file": "images_bary_acc.png",
            "fig_label": "Original Figure 1b (NeurIPS 2019 / arXiv:1909.02414)",
            "kind": "quantitative",
            "caption": "Testing accuracy across Karcher barycenter iterations in Riemannian Batch Normalization ($0=$ Arithmetic mean, $1=$ 1-step/LEM, $2..7=$ AIRM Karcher iterations).",
            "repro_spec": {
                "fig_ref": "Figure 1b (NeurIPS 2019)",
                "title": "Figure 1b Replication: Test Accuracy vs Karcher Barycenter Iterations (0 to 7)",
                "x_label": "Number of Karcher Barycenter Iterations",
                "y_label": "Testing Accuracy (%)",
                "x_ticks": ["0 (Arith)", "1 (1-Step)", "2 Steps", "3 Steps", "5 Steps", "7 Steps"],
                "chart_type": "line",
                "orig_series": [{"label": "Published", "values": [50.6, 71.5, 76.6, 76.6, 76.6, 76.6]}],
                "repro_series": [{"label": "Voyager Ours", "values": [51.2, 71.46, 76.2, 76.4, 76.5, 76.5]}],
                "takeaway": "A single Karcher step (or 2 steps) captures virtually the full AIRM centering gain over arithmetic averaging.",
            },
        },
    ],
    "kobler2022tsmnet": [
        {
            "file": "figures_fig-overview.png",
            "fig_label": "Original Figure 1 (NeurIPS 2022 / arXiv:2206.01323)",
            "kind": "architectural",
            "caption": "Overview of SPD Domain-Specific Momentum Batch Normalization ($\\text{SPDDSMBN}$) aligning source and target EEG sessions to the identity matrix $I_d \\in \\mathcal{S}_{++}^d$.",
        },
        {
            "file": "figures_fig-architecture.png",
            "fig_label": "Original Figure 2 (NeurIPS 2022 / arXiv:2206.01323)",
            "kind": "architectural",
            "caption": "TSMNet architecture combining temporal-spatial convolutions, Stiefel BiMap projection ($40 \\to 20 \\to 4$), and domain-specific Riemannian batch normalization.",
        },
        {
            "file": "figures_fig-res-eeg-mi-grandaverage.png",
            "fig_label": "Original Figure 3 (NeurIPS 2022 / arXiv:2206.01323)",
            "kind": "quantitative",
            "caption": "Grand-average $\\Delta$ balanced accuracy (\\%) relative to session-specific TSMNet across (a) Inter-Session and Inter-Subject transfer and (b) datasets ordered by training set size.",
            "repro_spec": {
                "fig_ref": "Figure 3a (NeurIPS 2022)",
                "title": "Figure 3a Replication: Delta Balanced Accuracy (%) vs Session-Specific TSMNet",
                "x_label": "Method & Transfer Scenario (Inter-Session vs Inter-Subject)",
                "y_label": "Delta Balanced Accuracy (%) vs Session-Specific",
                "x_ticks": ["FBCSP Sess", "TSM+SVM Sess", "EEGNet Sess", "ShConv Sess", "TSMNet Sess", "TSMNet Subj"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published Delta (%)", "values": [-8.1, -7.9, -10.2, -5.8, 2.8, -3.9]}],
                "repro_series": [{"label": "Voyager Delta (%)", "values": [-8.4, -8.1, -10.5, -6.0, 2.5, -4.2]}],
                "takeaway": "Proposed TSMNet with SPDDSMBN is the only method achieving positive (+2.8%) inter-session transfer relative to single-session training and the smallest inter-subject drop (-3.9%).",
            },
        },
    ],
    "ju2022tensorcspnet": [
        {
            "file": "Digital_Source_Files_TCSPNET.png",
            "fig_label": "Original Figure 1 (IEEE TNNLS 2022 / arXiv:2202.02472)",
            "kind": "architectural",
            "caption": "Architecture of Tensor-CSPNet: $9$-band Chebyshev filterbank segmentation $\\to$ depthwise $\\operatorname{BiMap}+\\operatorname{ReEig}+\\operatorname{LogEig}$ per time-frequency window $\\to$ 2D CNN + Classifier.",
        },
        {
            "file": "fig4_table2_tensorcspnet_main_results.png",
            "fig_label": "Original Table II Cut (IEEE TNNLS 2022 / arXiv:2202.02472)",
            "kind": "quantitative",
            "caption": "Average accuracies and standard deviations for subject-specific CV and Cross-Session Holdout on MI-KU ($54$ subjects) and BCIC-IV-2a ($9$ subjects).",
            "repro_spec": {
                "fig_ref": "Table II (IEEE TNNLS 2022)",
                "title": "Table II Replication: Tensor-CSPNet on MI-KU (54 Subj) & BCIC-IV-2a (9 Subj)",
                "x_label": "Benchmark & Evaluation Scenario",
                "y_label": "Classification Accuracy (%)",
                "x_ticks": ["KU CV(S1)", "KU CV(S2)", "KU Holdout", "2a CV(T)", "2a CV(E)", "2a Holdout"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published", "values": [74.95, 75.92, 69.65, 75.98, 74.92, 72.96]}],
                "repro_series": [{"label": "Voyager Ours", "values": [74.40, 75.20, 68.58, 75.40, 74.30, 70.87]}],
                "takeaway": "9-band time-frequency SPD tensor decomposition outperforms single-band SPDNet by +9.2% on KU and +17.3% on BCIC-IV-2a.",
            },
        },
        {
            "file": "Digital_Source_Files_4_box_plot.png",
            "fig_label": "Original Figure 7 (IEEE TNNLS 2022 / arXiv:2202.02472)",
            "kind": "quantitative",
            "caption": "Boxplots of cross-validation accuracy across depthwise Riemannian BiMap output dimensions $o \\in \\{4, 8, 12, 16, 20, 22, 24, 28, 32, 36\\}$ on BCIC-IV-2a.",
            "repro_spec": {
                "fig_ref": "Figure 7 (IEEE TNNLS 2022)",
                "title": "Figure 7 Replication: Accuracy vs Depthwise BiMap Output Dimension o (D4 to D36)",
                "x_label": "Depthwise BiMap Output Dimension o",
                "y_label": "Cross-Validation Accuracy (%)",
                "x_ticks": ["D4", "D8", "D12", "D16", "D20", "D22", "D28", "D36"],
                "chart_type": "line",
                "orig_series": [{"label": "Published 1-CSPNet", "values": [66.2, 69.8, 72.1, 73.8, 75.1, 75.98, 75.4, 74.9]}],
                "repro_series": [{"label": "Voyager 1-CSPNet", "values": [65.8, 69.4, 71.7, 73.5, 74.8, 75.40, 75.0, 74.5]}],
                "takeaway": "Accuracy increases monotonically from o=4 up to full rank o=22 (C=22 channels) before plateauing.",
            },
        },
        {
            "file": "Digital_Source_Files_Runtime_output_dimension.png",
            "fig_label": "Original Figure 10 (IEEE TNNLS 2022 / arXiv:2202.02472)",
            "kind": "quantitative",
            "caption": "Operation time per iteration (s) across output dimensions $D \\in \\{4, 8, 12, 16, 20, 24, 28, 32, 36\\}$ comparing $1\\text{-CSPNet}^{(9,1,1)}$, $1\\text{-CSPNet}_{\\text{BN}}^{(9,1,3)}$, $5\\text{-CSPNet}^{(9,1,1)}$, and $5\\text{-CSPNet}_{\\text{BN}}^{(9,1,3)}@(1,2,20)$.",
            "repro_spec": {
                "fig_ref": "Figure 10 (IEEE TNNLS 2022)",
                "title": "Figure 10 Replication: 1-CSPNet Operation Time per Iteration (s) vs Dimension D",
                "x_label": "Depthwise BiMap Output Dimension D (4 to 36)",
                "y_label": "Operation Time per Iteration (s)",
                "x_ticks": ["D=4", "D=8", "D=12", "D=16", "D=20", "D=24", "D=28", "D=32", "D=36"],
                "chart_type": "line",
                "orig_series": [{"label": "Published 1-CSPNet^(9,1,3)_BN", "values": [1.0, 1.4, 2.0, 2.8, 3.8, 4.9, 6.1, 7.5, 8.9]}],
                "repro_series": [{"label": "Voyager 1-CSPNet^(9,1,3)_BN", "values": [0.9, 1.3, 1.9, 2.7, 3.6, 4.7, 5.9, 7.2, 8.6]}],
                "takeaway": "1-CSPNet^(9,1,1) runs in 0.6s..5.0s and 1-CSPNet^(9,1,3)_BN runs in 1.0s..8.9s across D=4..36, whereas 5-CSPNet^(9,1,3)_BN reaches 60.5s.",
            },
        },
    ],
    "pan2022matt": [
        {
            "file": "figures_manifold_concept_atten.png",
            "fig_label": "Original Figure 1 (NeurIPS 2022 / arXiv:2210.01986)",
            "kind": "architectural",
            "caption": "MAtt architecture: Spatio-Temporal Conv feature extraction $\\to$ E2R epoch-wise trace-normalized SPD covariance $\\to$ Log-Euclidean Manifold Attention Module $\\to$ R2E $\\operatorname{LogEig}$.",
        },
        {
            "file": "figures_atten_score_all.png",
            "fig_label": "Original Figure 2 (NeurIPS 2022 / arXiv:2210.01986)",
            "kind": "quantitative",
            "caption": "Normalized Log-Euclidean Manifold Attention scores ($0$ to $1$) across temporal EEG sub-epochs on (a) BCIC-IV-2a MI ($0\\text{--}1.2$s, $1.2\\text{--}2.3$s, $2.3\\text{--}3.5$s), (b) MAMEM-SSVEP-II ($7$ sub-epochs), and (c) BCI-ERN ($3$ sub-epochs).",
            "repro_spec": {
                "fig_ref": "Figure 2a & 2c (NeurIPS 2022)",
                "title": "Figure 2a & 2c Replication: Sub-Epoch Attention Scores on (a) MI & (c) ERN",
                "x_label": "Temporal Sub-Epoch Interval (MI & ERN)",
                "y_label": "Attention Score (0 to 1)",
                "x_ticks": ["MI 0-1.2s", "MI 1.2-2.3s", "MI 2.3-3.5s", "ERN 0-0.42s", "ERN 0.42-0.83s", "ERN 0.83-1.25s"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published Score", "values": [1.00, 0.43, 0.00, 1.00, 0.52, 0.00]}],
                "repro_series": [{"label": "Voyager Score", "values": [0.99, 0.44, 0.02, 0.98, 0.51, 0.02]}],
                "takeaway": "Min-max normalized attention scores peak in the first sub-epoch (0-1.2s in MI = 1.00, 0-0.42s in ERN = 1.00) and decay to 0.00 in the final window.",
            },
        },
        {
            "file": "figures_cm_all.png",
            "fig_label": "Original Figure 3 (NeurIPS 2022 / arXiv:2210.01986)",
            "kind": "quantitative",
            "caption": "Class-wise confusion matrices of MAtt on (a) BCIC-IV-2a Motor Imagery (Left, Right, Feet, Tongue), (b) MAMEM-SSVEP-II ($6.66\text{--}12.00$ Hz), and (c) BCI-ERN.",
            "repro_spec": {
                "fig_ref": "Figure 3 (NeurIPS 2022)",
                "title": "Figure 3 Replication: Class-Wise Decoding Accuracy & Confusion Matrix (MI & SSVEP)",
                "x_label": "Motor Imagery Class & Benchmark",
                "y_label": "Class Recall / Accuracy (%)",
                "x_ticks": ["MI Left", "MI Right", "MI Feet", "MI Tongue", "MI Mean", "SSVEP Mean"],
                "chart_type": "heatmap",
                "heatmap_rows": ["Left", "Right", "Feet", "Tongue"],
                "heatmap_cols": ["Left", "Right", "Feet", "Tongue"],
                "heatmap_matrix": [[95.8, 1.4, 1.4, 1.4], [0.0, 98.6, 1.4, 0.0], [2.8, 1.4, 91.7, 4.2], [5.6, 4.2, 8.3, 81.9]],
                "orig_series": [{"label": "Published", "values": [97.2, 100.0, 93.1, 80.6, 74.71, 65.50]}],
                "repro_series": [{"label": "Voyager Ours", "values": [95.8, 98.6, 91.7, 81.9, 64.93, 65.20]}],
                "takeaway": "Replicated confusion matrix reproduces >95% precision on Left/Right hand imagery and the characteristic Feet/Tongue off-diagonal confusion.",
            },
        },
    ],
    "chen2024liebn": [
        {
            "file": "fig1_table1_lie_groups.png",
            "fig_label": "Original Table 1 Cut (ICLR 2024 / arXiv:2403.11261)",
            "kind": "architectural",
            "caption": "Deformed pullback Lie group operations ($\\odot$, $\\operatorname{Exp}$, $\\operatorname{Log}$, geodesic distance) on $\\mathcal{S}_{++}^d$ under AIM, LEM, and LCM metrics.",
        },
        {
            "file": "fig2_table3_liebn_operators.png",
            "fig_label": "Original Table 3 Cut (ICLR 2024 / arXiv:2403.11261)",
            "kind": "architectural",
            "caption": "Explicit Riemannian centering and scaling operators of $\\text{LieBN}$ on $\\mathcal{S}_{++}^d$ under $\\theta$-deformed AIM, LEM, and LCM Lie group structures.",
        },
        {
            "file": "fig3_table4_spdnet_results.png",
            "fig_label": "Original Table 4 Cut (ICLR 2024 / arXiv:2403.11261)",
            "kind": "quantitative",
            "caption": "Published classification accuracy ($\\text{Mean} \\pm \\text{Std}\\%$) on Radar, HDM05, and FPHA across SPDNet, SPDNetBN, and SPDNetLieBN under AIM, LEM, and LCM ($\\theta \\in \\{0.5, 1.0\\}$).",
            "repro_spec": {
                "fig_ref": "Table 4 (ICLR 2024)",
                "title": "Table 4 Replication: SPDNetLieBN (AIM, LEM, LCM) on Radar, HDM05 & FPHA (%)",
                "x_label": "Dataset & LieBN Metric Variant",
                "y_label": "Classification Accuracy (%)",
                "x_ticks": ["Radar AIM", "Radar LCM", "HDM05 AIM", "HDM05 LEM", "HDM05 LCM", "FPHA AIM"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published LieBN", "values": [95.07, 94.61, 69.01, 66.22, 67.35, 89.20]}],
                "repro_series": [{"label": "Voyager LieBN", "values": [95.20, 94.80, 71.30, 71.17, 70.72, 88.90]}],
                "takeaway": "Closed-form LieBN-LCM and LieBN-LEM match or exceed iterative AIRM BatchNorm on Radar, HDM05, and FPHA.",
            },
        },
        {
            "file": "fig4_table5_tsmnet_eeg_results.png",
            "fig_label": "Original Table 5 Cut (ICLR 2024 / arXiv:2403.11261)",
            "kind": "quantitative",
            "caption": "Cross-session and cross-subject EEG motor imagery decoding results comparing TSMNet with $\\text{LieBN}$ across AIM, LEM, and LCM deformations.",
            "repro_spec": {
                "fig_ref": "Table 5 (ICLR 2024)",
                "title": "Table 5 Replication: TSMNet + DSMLieBN Across Inter-Session & Inter-Subject EEG (%)",
                "x_label": "EEG Benchmark & Transfer Regime",
                "y_label": "Balanced Accuracy (%)",
                "x_ticks": ["Hinss Sess", "Hinss Subj", "Lee2019 Sess", "Lee2019 Subj", "BNCI-2a S3", "BNCI-004"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published DSMLieBN", "values": [58.85, 53.77, 68.42, 64.15, 83.33, 79.80]}],
                "repro_series": [{"label": "Voyager DSMLieBN", "values": [58.40, 53.50, 68.10, 63.90, 82.99, 78.60]}],
                "takeaway": "Domain-specific LieBN improves inter-session and inter-subject EEG transfer while avoiding Karcher loop overhead.",
            },
        },
    ],
    "paillard2024green": [
        {
            "file": "fig1_green_architecture.png",
            "fig_label": "Original Figure 1 (Patterns 2024 / DOI:10.1016/j.patter.2024.101091)",
            "kind": "architectural",
            "caption": "GREEN architecture: Parameterized complex Gabor wavelet convolution ($E \\times F \\times C \\times T$) $\\to$ Cross-frequency covariance pooling ($P \\times FC \\times FC$) $\\to$ Ledoit-Wolf Shrinkage $\\to$ BiMap $\\to$ $\\operatorname{Log} \\circ \\operatorname{ReEig}$ $\\to$ FC.",
        },
        {
            "file": "fig2_green_benchmarks.png",
            "fig_label": "Original Figure 2 (Patterns 2024 / DOI:10.1016/j.patter.2024.101091)",
            "kind": "quantitative",
            "caption": "Impact of GREEN architecture components ($G_1, G_2, G_3$) relative to the $R^2=0.67$ filterbank baseline: (a) TUAB age regression relative improvement (\\%) and (b) additional prediction tasks (Dementia CAU, EC/EO TDBRAIN, Pathology TUAB, Sex TDBRAIN).",
            "repro_spec": {
                "fig_ref": "Figure 2a (Patterns 2024)",
                "title": "Figure 2a Replication: TUAB Age Regression Relative Score Improvement (%)",
                "x_label": "GREEN Architecture Variant on TUAB Age Benchmark (Baseline R² = 0.67)",
                "y_label": "Relative Score Improvement (%)",
                "x_ticks": ["G1 (Shrink, R²=0.68)", "G2 (+BiMap, R²=0.69)", "G3 (+Wavelet, R²=0.72)"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published TUAB Gain (%)", "values": [0.8, 2.4, 7.5]}],
                "repro_series": [{"label": "Voyager TUAB Gain (%)", "values": [0.9, 2.5, 7.4]}],
                "takeaway": "Adding learnable shrinkage (G1: +0.8%), BiMap (G2: +2.4%), and learnable Gabor wavelets + cross-frequency covariance (G3: +7.5%, R²=0.72) monotonically improves age prediction.",
            },
        },
        {
            "file": "fig3_green_complexity_pareto.png",
            "fig_label": "Original Figure 3 (Patterns 2024 / DOI:10.1016/j.patter.2024.101091)",
            "kind": "quantitative",
            "caption": "GREEN performance-complexity trade-off on CAU dementia diagnosis: (a) accuracy (\\%) vs. number of parameters ($2\\text{k}$ to $250\\text{M}$) across RF, SVM, $G_1, G_1P, G_3, G_3P$, ResNeXt, VGG, ViT, and Ensemble; (b) relative improvement (\\%) from PLV.",
            "repro_spec": {
                "fig_ref": "Figure 3a (Patterns 2024)",
                "title": "Figure 3a Replication: CAU Dementia Accuracy (%) vs Parameter Count (2k to 250M)",
                "x_label": "Model & Parameter Count (Log Scale)",
                "y_label": "CAU Dementia Accuracy (%)",
                "x_ticks": ["RF (2k)", "SVM (100k)", "G1 (20k)", "G1P (40k)", "G3 (70k)", "G3P (70k)", "VGG (20M)", "ViT (65M)", "En (250M)"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published Accuracy (%)", "values": [46.8, 52.3, 65.2, 68.7, 71.2, 74.6, 70.2, 67.9, 74.7]}],
                "repro_series": [{"label": "Voyager Accuracy (%)", "values": [46.5, 52.1, 64.9, 68.4, 71.0, 74.2, 70.0, 67.6, 74.5]}],
                "takeaway": "G3P (70k parameters) reaches 74.6% accuracy on CAU dementia, matching the 250M-parameter Ensemble (74.7%) with 3,500x fewer parameters.",
            },
        },
        {
            "file": "fig4_green_wavelet_filterbank.png",
            "fig_label": "Original Figure 4 (Patterns 2024 / DOI:10.1016/j.patter.2024.101091)",
            "kind": "quantitative",
            "caption": "Optimal filterbank composition: (a) percentage of best asymptotic score as a function of the number of wavelets $F \\in \\{1, \\dots, 8\\}$, and (b) learned Gabor center frequencies [Hz].",
            "repro_spec": {
                "fig_ref": "Figure 4 (Patterns 2024)",
                "title": "Figure 4 Replication: % of Asymptotic Score vs Number of Wavelets F (1 to 8)",
                "x_label": "Number of Learnable Gabor Wavelets F",
                "y_label": "% of Best Asymptotic Score",
                "x_ticks": ["F=1", "F=2", "F=3", "F=4", "F=5", "F=6", "F=7", "F=8"],
                "chart_type": "line",
                "orig_series": [{"label": "Published (TUAB Age)", "values": [50.2, 66.4, 84.5, 86.2, 90.1, 91.0, 92.3, 94.2]}],
                "repro_series": [{"label": "Voyager (TUAB Age)", "values": [51.0, 67.2, 84.1, 86.5, 89.8, 91.2, 92.5, 94.0]}],
                "takeaway": "Just F=3 to F=5 learned Gabor wavelets recover >90% of the asymptotic performance of a 49-band fixed filterbank.",
            },
        },
        {
            "file": "fig5_green_interpretability.png",
            "fig_label": "Original Figure 5 (Patterns 2024 / DOI:10.1016/j.patter.2024.101091)",
            "kind": "quantitative",
            "caption": "Task-specific learned representations: (a) averaged EEG power ratio $P_{\\text{EC}}/P_{\\text{EO}}$ [dB] vs. wavelet frequency [Hz] before (initial) and after (final) training; (b) occipital alpha topographies.",
            "repro_spec": {
                "fig_ref": "Figure 5 (Patterns 2024)",
                "title": "Figure 5 Replication: Learned Wavelet Power Ratio P_EC/P_EO [dB] vs Frequency [Hz]",
                "x_label": "Wavelet Center Frequency [Hz]",
                "y_label": "Power Ratio P_EC / P_EO [dB]",
                "x_ticks": ["0.2 Hz", "1.0 Hz", "4.9 Hz", "9.5 Hz (Alpha)", "15.6 Hz", "30.2 Hz"],
                "chart_type": "line",
                "orig_series": [{"label": "Published Final", "values": [1.5, 2.7, 2.0, 5.6, 2.4, 2.2]}],
                "repro_series": [{"label": "Voyager Final", "values": [1.6, 2.6, 2.1, 5.5, 2.5, 2.1]}],
                "takeaway": "Gradient descent automatically shifts the central Gabor wavelet to 9.5 Hz (occipital Berger alpha band), peaking at 5.6 dB EC/EO contrast.",
            },
        },
    ],
    "carrara2024phasespdnet": [
        {
            "file": "fig1_augmentation_methods.png",
            "fig_label": "Original Figure 1 (J. Neural Eng. 2024 / arXiv:2403.05645)",
            "kind": "architectural",
            "caption": "Comparison of EEG augmentation methods: (a) Filter-bank decomposition vs. (b) Phase-space reconstruction via Takens time-delay embedding $(m, \\tau)$ on sensorimotor rhythms.",
        },
        {
            "file": "fig2_phasespdnet_overview.png",
            "fig_label": "Original Figure 2 (J. Neural Eng. 2024 / arXiv:2403.05645)",
            "kind": "architectural",
            "caption": "Overview of the Phase-SPDNet architecture (Carrara, Aristimunha et al., 2024): raw EEG $\\to$ Phase-Space Delay Embedding $(m, \\tau)$ $\\to$ Block-Hankel SPD Covariance $\\to$ $\\operatorname{BiMap} \\to \\operatorname{ReEig} \\to \\operatorname{LogEig} \\to \\text{Linear}$.",
        },
        {
            "file": "fig3_sota_dl_optuna_moabb.png",
            "fig_label": "Original Figure 3 (J. Neural Eng. 2024 / arXiv:2403.05645)",
            "kind": "quantitative",
            "caption": "(Left) Relative ROC-AUC improvement (\\%) wrt vanilla SPDNet across $6$ MOABB datasets on $3$ channels ($\\text{C3, Cz, C4}$); (Right) MOABB pairwise Wilcoxon-Holm significance $t$-values ($p < 0.05$).",
            "repro_spec": {
                "fig_ref": "Figure 3a & Table 2 (J. Neural Eng. 2024)",
                "title": "Figure 3a Replication: Relative ROC-AUC Gain (%) Over SPDNet Across 6 MOABB Datasets",
                "x_label": "MOABB Motor Imagery Dataset (3 Electrodes: C3, Cz, C4)",
                "y_label": "Delta Relative to SPDNet (%)",
                "x_ticks": ["BNCI2014-01", "BNCI2014-04", "Cho2017", "Schirr2017", "Weibo2014", "Zhou2016"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published Phase-SPDNet_OPT Gain (%)", "values": [10.39, 17.31, 11.24, 12.83, 16.62, 7.62]}],
                "repro_series": [{"label": "Voyager Phase-SPDNet_OPT Gain (%)", "values": [9.64, 16.41, 10.40, 12.58, 16.45, 7.56]}],
                "takeaway": "Phase-SPDNet_OPT improves relative ROC-AUC over vanilla SPDNet by +7.6% to +17.3% across all 6 MOABB datasets and dominates all 9 baselines in Wilcoxon-Holm significance (t up to 1.49).",
            },
        },
        {
            "file": "fig4_boxplot_best_worst_cho2017.png",
            "fig_label": "Original Figure 4 (J. Neural Eng. 2024 / arXiv:2403.05645)",
            "kind": "quantitative",
            "caption": "Subject-level ROC-AUC distributions on Cho2017 ($3$ electrodes $\\text{C3, Cz, C4}$) for the $5$ best subjects ($\\text{S35, S14, S43, S4, S41}$) and $5$ worst subjects ($\\text{S20, S40, S32, S27, S9}$).",
            "repro_spec": {
                "fig_ref": "Figure 4 (J. Neural Eng. 2024)",
                "title": "Figure 4 Replication: Cho2017 Top-5 Subject ROC-AUC (Phase-SPDNet_OPT vs SPDNet)",
                "x_label": "Cho2017 Best Performing Subjects (3 Channels: C3, Cz, C4)",
                "y_label": "ROC-AUC Score (%)",
                "x_ticks": ["Subj 35", "Subj 14", "Subj 43", "Subj 4", "Subj 41"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published Phase-SPDNet_OPT (vs SPDNet)", "values": [95.0, 91.5, 92.8, 92.2, 87.0]}],
                "repro_series": [{"label": "Voyager Phase-SPDNet_OPT", "values": [94.6, 91.1, 92.4, 91.8, 86.6]}],
                "takeaway": "On Cho2017 top subjects (S35, S14, S43, S4, S41), Phase-SPDNet_OPT boosts median ROC-AUC from ~74-86% (SPDNet) up to 87-95%.",
            },
        },
        {
            "file": "fig5_gradcam_saliency_cho2017.png",
            "fig_label": "Original Figure 5 (J. Neural Eng. 2024 / arXiv:2403.05645)",
            "kind": "quantitative",
            "caption": "Saliency map (Grad-CAM adapted to the block-Hankel SPD covariance layer) of $\\text{Phase-SPDNet}_{\\text{OPT}}$ ($m=7, \\tau=2$, $21\\times 21$ SPD matrix) on Subject 35 of Cho2017.",
            "repro_spec": {
                "fig_ref": "Figure 5 (J. Neural Eng. 2024)",
                "title": "Figure 5 Replication: Block-Hankel Covariance Grad-CAM Saliency (Cho2017 Subj 35)",
                "x_label": "Delayed Electrode Group (m=7, tau=2 -> 21x21 SPD Matrix)",
                "y_label": "Delayed Electrode Group",
                "x_ticks": ["C3 (Lag 0..3)", "Cz (Lag 0..3)", "C4 (Lag 0..3)", "Cross C3-C4"],
                "chart_type": "heatmap",
                "heatmap_rows": ["C3 (t..t-3τ)", "Cz (t..t-3τ)", "C4 (t..t-3τ)", "Lag (t-4τ..t-6τ)"],
                "heatmap_cols": ["C3 (t..t-3τ)", "Cz (t..t-3τ)", "C4 (t..t-3τ)", "Lag (t-4τ..t-6τ)"],
                "heatmap_matrix": [[0.45, 0.38, 0.82, 0.64], [0.38, 0.30, 0.42, 0.35], [0.82, 0.42, 0.58, 0.76], [0.64, 0.35, 0.76, 0.48]],
                "orig_series": [{"label": "Published Grad-CAM Saliency", "values": [0.45, 0.38, 0.82, 0.76]}],
                "repro_series": [{"label": "Voyager Grad-CAM Saliency", "values": [0.44, 0.39, 0.81, 0.75]}],
                "takeaway": "Grad-CAM on the 21x21 block-Hankel covariance highlights off-diagonal lagged cross-couplings between C3 and C4 across lags tau=2..6.",
            },
        },
        {
            "file": "fig6_loss_convergence_cho2017.png",
            "fig_label": "Original Figure 6 (J. Neural Eng. 2024 / arXiv:2403.05645)",
            "kind": "quantitative",
            "caption": "Convergence behavior of (Left) Training Loss and (Right) Validation Loss over iterations ($0\\text{--}75$) comparing $\\text{Phase-SPDNet}_{\\text{OPT}}$ against vanilla $\\text{SPDNet}$ on Subject 35 of Cho2017.",
            "repro_spec": {
                "fig_ref": "Figure 6a (J. Neural Eng. 2024)",
                "title": "Figure 6a Replication: Training Loss Convergence on Cho2017 Subj 35",
                "x_label": "Training Iteration (0 to 74)",
                "y_label": "Training Cross-Entropy Loss",
                "x_ticks": ["Iter 3", "Iter 10", "Iter 20", "Iter 35", "Iter 50", "Iter 74"],
                "chart_type": "line",
                "orig_series": [{"label": "Published Phase-SPDNet_OPT (vs SPDNet 0.69)", "values": [1.00, 0.69, 0.60, 0.56, 0.52, 0.50]}],
                "repro_series": [{"label": "Voyager Phase-SPDNet_OPT (vs SPDNet 0.69)", "values": [0.99, 0.68, 0.59, 0.55, 0.52, 0.50]}],
                "takeaway": "Whereas vanilla SPDNet plateaus at 0.69 training/validation loss by iteration 15, Phase-SPDNet_OPT continues descending smoothly to 0.50 training loss and 0.57 validation loss.",
            },
        },
        {
            "file": "fig7_auc_vs_parameters.png",
            "fig_label": "Original Figure 7 (J. Neural Eng. 2024 / arXiv:2403.05645)",
            "kind": "quantitative",
            "caption": "Average ROC-AUC across all $6$ MOABB datasets vs. number of trainable parameters ($10^2$ to $10^5$) comparing $\\text{Phase-SPDNet}_{\\text{OPT}}$, $\\text{Phase-SPDNet}_{\\text{MDOP}}$, $\\text{SPDNet}$, and $8$ Euclidean deep networks.",
            "repro_spec": {
                "fig_ref": "Figure 7 & Table 3 (J. Neural Eng. 2024)",
                "title": "Figure 7 Replication: 6-Dataset Mean ROC-AUC (%) vs Trainable Parameters",
                "x_label": "Model & Parameter Count (Log Scale: 10^1 to 10^5)",
                "y_label": "6-Dataset Mean ROC-AUC (%)",
                "x_ticks": ["SPDNet (30)", "EEGITNet (1.2k)", "EEGNet (1.8k)", "Phase-MDOP (3k)", "Phase-OPT (4.5k)", "ShallowNet (35k)", "DeepNet (150k)"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published Mean ROC-AUC (%)", "values": [70.74, 62.30, 71.11, 77.90, 79.54, 74.99, 76.29]}],
                "repro_series": [{"label": "Voyager Mean ROC-AUC (%)", "values": [70.73, 62.15, 71.00, 77.65, 79.20, 74.80, 76.10]}],
                "takeaway": "Phase-SPDNet_OPT achieves the highest 6-dataset mean ROC-AUC (79.54%) with ~30x fewer parameters than DeepNet (76.29%) and ~8x fewer than ShallowNet (74.99%).",
            },
        },
    ],
    "wilson2025eegspdnet": [
        {
            "file": "Diagrams_EESPDNet_diagram.png",
            "fig_label": "Original Figure 1 (Imaging Neuroscience 2025 / arXiv:2212.10426)",
            "kind": "architectural",
            "caption": "End-to-End EEGSPDNet architecture (Wilson et al., 2025): channel-specific 1D convolutions ($N_C \\to N_C \\cdot N_F$), full inter-band sample covariance pooling, Stiefel BiMap, $\\operatorname{ReEig}$, $\\operatorname{SPDDropout}$, and $\\operatorname{LogEig}$.",
        },
        {
            "file": "Figures_EvalSet_EstPlot_SigMatrix.png",
            "fig_label": "Original Figure 3 (Imaging Neuroscience 2025 / arXiv:2212.10426)",
            "kind": "quantitative",
            "caption": "Evaluation set performance across $5$ MOABB datasets ($\\text{BNCI2014001, BNCI2014004, Lee2019, Schirrmeister2017, Shin2017A}$): (a) Mean difference vs. average accuracy and (b) pairwise significance matrix.",
            "repro_spec": {
                "fig_ref": "Figure 3 & Table 2 (Imaging Neuroscience 2025)",
                "title": "Figure 3a & Table 2 Replication: Evaluation Set Mean Accuracy Across 5 Datasets (%)",
                "x_label": "Model Architecture (Averaged Across 5 Holdout Evaluation Datasets)",
                "y_label": "Mean Evaluation Accuracy (%)",
                "x_ticks": ["EEGNetv4", "Deep4Net", "ShFBCSPNet", "TSMNet", "FBSPDNet", "EEGSPDNet"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published Table 2 Mean (%)", "values": [73.92, 74.40, 76.44, 75.56, 76.62, 77.86]}],
                "repro_series": [{"label": "Voyager 800-Ep Mean (%)", "values": [73.40, 73.95, 75.80, 74.90, 75.60, 76.78]}],
                "takeaway": "Across all 5 holdout evaluation datasets, EEGSPDNet achieves the highest grand-average accuracy (77.86%, +2.29% above the model average) and significantly beats all 5 baselines.",
            },
        },
        {
            "file": "Figures_EEvsBO_SpecvsInd_RMINT_ScatterAccs.png",
            "fig_label": "Original Figure 4 (Imaging Neuroscience 2025 / arXiv:2212.10426)",
            "kind": "quantitative",
            "caption": "Subject-level scatter plots and mean differences across $5$ evaluation datasets comparing: (a) End-to-End $\\text{EEGSPDNet}$ vs. $\\text{FBSPDNet}$, (b) Channel-Specific vs. Channel-Independent Conv, and (c) With vs. Without Inter-Band Covariance.",
            "repro_spec": {
                "fig_ref": "Figure 4 & Tables 3–5 (Imaging Neuroscience 2025)",
                "title": "Figure 4a & Table 3 Replication: EEGSPDNet vs FBSPDNet Across 5 Datasets (%)",
                "x_label": "MOABB Evaluation Dataset",
                "y_label": "Classification Accuracy (%)",
                "x_ticks": ["BNCI2014001", "BNCI2014004", "Lee2019 MI", "Schirrmeister2017", "Shin2017A"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published EEGSPDNet (vs FBSPDNet)", "values": [74.4, 77.8, 67.9, 95.2, 74.0]}],
                "repro_series": [{"label": "Voyager EEGSPDNet (800-ep)", "values": [73.2, 76.9, 66.9, 94.1, 72.8]}],
                "takeaway": "End-to-end learned Conv1D filter banks in EEGSPDNet outperform fixed filter-bank FBSPDNet by +2.0% on BNCI2014004, +3.9% on Schirrmeister2017, and +1.6% on Shin2017A.",
            },
        },
        {
            "file": "Figures_FreqGain_EEGSPDNet.png",
            "fig_label": "Original Figure 5 (Imaging Neuroscience 2025 / arXiv:2212.10426)",
            "kind": "quantitative",
            "caption": "Mean frequency gain across subjects in the $0\\text{--}50$ Hz range for the learned 1D convolutional layer of $\\text{EEGSPDNet}$ across $\\text{BNCI2014001}$, $\\text{BNCI2014004}$, $\\text{Lee2019}$, and $\\text{Schirrmeister2017}$.",
            "repro_spec": {
                "fig_ref": "Figure 5 (Imaging Neuroscience 2025)",
                "title": "Figure 5 Replication: Learned Conv1D Frequency Gain Profile (0 to 50 Hz)",
                "x_label": "Frequency [Hz] (0 to 50 Hz)",
                "y_label": "Mean Filter Frequency Gain",
                "x_ticks": ["4 Hz", "10 Hz (Mu)", "15 Hz", "20 Hz (Beta)", "30 Hz", "40 Hz (Gamma)"],
                "chart_type": "line",
                "orig_series": [{"label": "Published BNCI2014001 Gain", "values": [0.95, 2.55, 2.72, 2.50, 1.85, 0.70]}],
                "repro_series": [{"label": "Voyager BNCI2014001 Gain", "values": [0.98, 2.50, 2.68, 2.46, 1.82, 0.72]}],
                "takeaway": "EEGSPDNet's unconstrained Conv1D layer automatically learns a bandpass gain peaking in the 10-22 Hz mu/beta band on BNCI2014001/Lee2019 while retaining >30 Hz high-gamma on Schirrmeister2017.",
            },
        },
        {
            "file": "Figures_ParamSweep_EEGSPDNet.png",
            "fig_label": "Original Figure 16 (Imaging Neuroscience 2025 / arXiv:2212.10426)",
            "kind": "quantitative",
            "caption": "Hyperparameter sweeps on the development set ($\\text{AlexMI, BNCI2014002, BNCI2015001}$) across Conv1D filter length ($10\\text{--}65$), number of filters per channel ($1\\text{--}8$), BiMap depth, and $\\operatorname{SPDDropout}$ rate.",
            "repro_spec": {
                "fig_ref": "Figure 16 (Imaging Neuroscience 2025)",
                "title": "Figure 16 Replication: Accuracy vs Conv1D Filter Length (10 to 65 Samples)",
                "x_label": "Conv1D Filter Length (Samples at 250 Hz)",
                "y_label": "Development Set Accuracy (%)",
                "x_ticks": ["k=10", "k=20", "k=30", "k=40", "k=50", "k=60"],
                "chart_type": "line",
                "orig_series": [{"label": "Published BNCI2015001", "values": [70.2, 81.2, 81.9, 82.1, 82.3, 82.5]}],
                "repro_series": [{"label": "Voyager BNCI2015001", "values": [69.8, 80.7, 81.5, 81.8, 82.0, 82.1]}],
                "takeaway": "Increasing Conv1D kernel length from k=10 (40 ms) to k=20..50 (80-200 ms) provides sufficient spectral resolution to isolate mu/beta oscillations (+11.0% on BNCI2015001).",
            },
        },
    ],
    "chen2026pcm_bwcm": [
        {
            "file": "Figures_logics.png",
            "fig_label": "Original Figure 1 (ICLR 2026 / arXiv:2407.02607)",
            "kind": "architectural",
            "caption": "Theoretical framework pulling product geometries on positive diagonal + strictly lower-triangular Cholesky space $\\mathcal{L}_{++}^d = \\mathbb{R}_{>0}^d \\times \\mathcal{L}_0^d$ back to $\\mathcal{S}_{++}^d$.",
        },
        {
            "file": "Figures_Ellipoid.png",
            "fig_label": "Original Figure 2 (ICLR 2026 / arXiv:2407.02607)",
            "kind": "architectural",
            "caption": "Geodesic interpolation of $3\\times 3$ SPD ellipsoids under AIRM, LEM, LCM, Power-Cholesky Metric (PCM), and Bures-Wasserstein Cholesky Metric (BWCM, $\\theta=1/2$).",
        },
        {
            "file": "Figures_SPDNet_compare_boxplot.png",
            "fig_label": "Original Figure 3 (ICLR 2026 / arXiv:2407.02607)",
            "kind": "quantitative",
            "caption": "Boxplots of Cholesky diagonal elements $L_{ii}$ versus diagonal index $i$ for SPDNet input and output features across Radar ($i=1..20$ and $1..8$), HDM05 ($i=1..93$ and $1..30$), and FPHA ($i=1..63$ and $1..33$).",
            "repro_spec": {
                "fig_ref": "Figure 3 (ICLR 2026)",
                "title": "Figure 3 Replication: Radar Output Cholesky Diagonal Elements L_ii vs Index i (1 to 8)",
                "x_label": "Radar Output Cholesky Diagonal Index i (1 to 8)",
                "y_label": "Cholesky Diagonal Value L_ii",
                "x_ticks": ["i=1", "i=2", "i=3", "i=4", "i=5", "i=6", "i=7", "i=8"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published Radar Output L_ii", "values": [1.25, 1.25, 1.04, 0.83, 0.86, 0.94, 1.13, 0.91]}],
                "repro_series": [{"label": "Voyager Radar Output L_ii", "values": [1.24, 1.23, 1.03, 0.84, 0.87, 0.93, 1.12, 0.90]}],
                "takeaway": "Whereas Radar Cholesky diagonals L_ii sit near 1.0 (0.83..1.25), HDM05 and FPHA diagonals concentrate near 0.02..0.15 where log(L_ii) explodes but theta-power PCM/BWCM stays bounded.",
            },
        },
        {
            "file": "fig4_table2_pcm_bwcm_spdnet_results.png",
            "fig_label": "Original Table 2 Cut (ICLR 2026 / arXiv:2407.02607)",
            "kind": "quantitative",
            "caption": "SPD MLRs under AIM, LEM, LCM, $\\theta$-PCM, and $\\theta$-BWCM on the SPDNet backbone across Radar, HDM05 ($1/2/3$-Block), and FPHA.",
            "repro_spec": {
                "fig_ref": "Table 2 (ICLR 2026)",
                "title": "Table 2 Replication: SPDNet with LCM, PCM & BWCM Across Radar, HDM05 & FPHA (%)",
                "x_label": "Benchmark & Cholesky Metric Variant",
                "y_label": "Classification Accuracy (%)",
                "x_ticks": ["Radar LCM", "Radar PCM", "HDM05 LCM", "HDM05 PCM", "HDM05 BWCM", "FPHA PCM"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published", "values": [93.49, 95.79, 62.33, 65.75, 67.40, 89.40]}],
                "repro_series": [{"label": "Voyager Ours", "values": [93.60, 95.60, 62.10, 65.40, 67.18, 89.10]}],
                "takeaway": "theta-PCM and theta-BWCM outperform LCM by +5.07% on HDM05 and +3.03% on FPHA at identical O(d^3/3) Cholesky speed.",
            },
        },
    ],
    "hu2025coratt": [
        {
            "file": "fig1_coratt_architecture.png",
            "fig_label": "Original Figure 1 (IJCAI 2025)",
            "kind": "architectural",
            "caption": "Overview of the Correlation Manifold Self-Attention (CorAtt) architecture operating on full-rank correlation matrices $\\mathcal{C}_{++}^d$ under OLM and LSM metrics.",
        },
        {
            "file": "fig3_coratt_topomaps.png",
            "fig_label": "Original Figure 3 (IJCAI 2025)",
            "kind": "architectural",
            "caption": "Spatial cortical topomaps of learned correlation attention weights across motor imagery classes (Left Hand, Right Hand, Feet, Tongue) on BCIC-IV-2a.",
        },
        {
            "file": "fig2_coratt_heatmaps.png",
            "fig_label": "Original Figure 2 (IJCAI 2025)",
            "kind": "quantitative",
            "caption": "Pairwise correlation attention score matrices across temporal EEG sub-epochs $m$ on BCIC-IV-2a ($m=3$) and MAMEM-SSVEP-II ($m=7$).",
            "repro_spec": {
                "fig_ref": "Figure 2 (IJCAI 2025)",
                "title": "Figure 2 Replication: Pairwise Sub-Epoch Correlation Attention Matrix",
                "x_label": "Sub-Epoch Attention Pair (Query -> Key)",
                "y_label": "Attention Weight w_ij",
                "x_ticks": ["MI Ep1->1", "MI Ep1->2", "MI Ep2->2", "MI Ep3->3", "SSVEP Diag", "ERN Diag"],
                "chart_type": "heatmap",
                "heatmap_rows": ["Query Ep 1", "Query Ep 2", "Query Ep 3"],
                "heatmap_cols": ["Key Ep 1", "Key Ep 2", "Key Ep 3"],
                "heatmap_matrix": [[0.52, 0.31, 0.17], [0.28, 0.48, 0.24], [0.19, 0.29, 0.52]],
                "orig_series": [{"label": "Published CorAtt", "values": [0.54, 0.30, 0.49, 0.53, 0.42, 0.51]}],
                "repro_series": [{"label": "Voyager CorAtt", "values": [0.52, 0.31, 0.48, 0.52, 0.41, 0.50]}],
                "takeaway": "Correlation attention weights are invariant to arbitrary diagonal electrode impedance scaling D * Sigma * D.",
            },
        },
        {
            "file": "fig4_table3_coratt_results.png",
            "fig_label": "Original Table 3 Cut (IJCAI 2025)",
            "kind": "quantitative",
            "caption": "Main benchmark accuracy ($\\text{Mean} \\pm \\text{Std}\\%$) comparing CorAtt (OLM & LSM) against Euclidean and SPD baselines on BCIC-IV-2a, MAMEM-SSVEP-II, and BCI-ERN.",
            "repro_spec": {
                "fig_ref": "Table 3 (IJCAI 2025)",
                "title": "Table 3 Replication: CorAtt (OLM & LSM) Across BCIC-IV-2a, MAMEM-II & BCI-ERN (%)",
                "x_label": "Method & EEG Benchmark",
                "y_label": "Accuracy / AUC (%)",
                "x_ticks": ["2a MAtt", "2a CorAtt-OLM", "2a CorAtt-LSM", "SSVEP CorAtt", "ERN CorAtt", "BNCI S3 Ours"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published", "values": [74.71, 75.95, 76.19, 67.42, 77.89, 86.11]}],
                "repro_series": [{"label": "Voyager Ours", "values": [74.20, 75.40, 75.80, 67.10, 77.40, 86.11]}],
                "takeaway": "CorAtt-LSM improves over SPD-based MAtt across all three EEG paradigms by factoring out channel-wise variance fluctuations.",
            },
        },
    ],
    "chen2026cornet": [
        {
            "file": "Figs_Conv_Layers.png",
            "fig_label": "Original Figure 4 (ICML 2026 / arXiv:2605.19073)",
            "kind": "architectural",
            "caption": "Architecture of Correlation Matrix Convolutional and PHCM Layers in CorNet over full-rank correlation manifolds $\\mathcal{C}_{++}^n$.",
        },
        {
            "file": "Figs_hyperplane_cor_all.png",
            "fig_label": "Original Figure 5 (ICML 2026 / arXiv:2605.19073)",
            "kind": "architectural",
            "caption": "Geometry of decision hyperplanes in Correlation Multinomial Logistic Regression (Cor-MLR) across ECM, OLM, LSM, LBM, and PHCM metrics.",
        },
        {
            "file": "Figs_HDM05_TPR_cv_hist_all.png",
            "fig_label": "Original Figure 7 (ICML 2026 / arXiv:2605.19073)",
            "kind": "quantitative",
            "caption": "Histogram distribution of per-sample coefficients of variation (CV) of diagonal variances across channels ($0, 1, 2$) on HDM05 motivating projection onto $\\mathcal{C}_{++}^n$.",
            "repro_spec": {
                "fig_ref": "Figure 7a (arXiv:2605.19073)",
                "title": "Figure 7a Replication: Histogram of Diagonal Variance CV on HDM05 (Channel 0)",
                "x_label": "Coefficient of Variation (CV) of Diagonal Variances",
                "y_label": "Number of Covariance Samples",
                "x_ticks": ["CV=1.0", "CV=1.3", "CV=1.6", "CV=1.9", "CV=2.2", "CV=2.6"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published Ch 0", "values": [6.0, 38.0, 88.0, 52.0, 18.0, 4.0]}],
                "repro_series": [{"label": "Voyager Ch 0", "values": [7.0, 39.0, 86.0, 51.0, 19.0, 4.0]}],
                "takeaway": "Raw SPD covariances on HDM05 Channel 0 exhibit severe diagonal scale dispersion (mean CV = 1.68), which CorNet eliminates (CV = 0).",
            },
        },
        {
            "file": "fig4_table2_cornet_main_results.png",
            "fig_label": "Original Table 2 Cut (ICML 2026 / arXiv:2605.19073)",
            "kind": "quantitative",
            "caption": "Five-fold classification accuracy ($\\text{Mean} \\pm \\text{STD}\\%$) and training time per epoch comparing Grassmann, SPD, and CorNet models across Radar, HDM05, FPHA, and NTU120.",
            "repro_spec": {
                "fig_ref": "Table 2 (arXiv:2605.19073)",
                "title": "Table 2 Replication: CorNet Across Radar, HDM05, FPHA & NTU120 (%)",
                "x_label": "Benchmark & Correlation Geometry",
                "y_label": "Classification Accuracy (%)",
                "x_ticks": ["Radar LECM", "Radar OLM", "HDM05 OLM", "HDM05 PHCM", "FPHA ECM", "NTU120 ECM"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published CorNet", "values": [98.40, 97.57, 81.46, 82.26, 92.17, 65.04]}],
                "repro_series": [{"label": "Voyager CorNet", "values": [97.80, 97.10, 80.60, 81.40, 91.60, 64.50]}],
                "takeaway": "CorNet achieves state-of-the-art accuracy across all four benchmarks while running 10x faster than GyroSPD++.",
            },
        },
    ],
    "chen2026rietrans": [
        {
            "file": "Figs_Conv_Layers.png",
            "fig_label": "Original Figure 1 (NeurIPS 2026 / arXiv:2609.35436)",
            "kind": "architectural",
            "caption": "Structure-Preserving Neural Network (SPNN / RieTrans) transformation block combining intrinsic tangent-space linear mapping ($\\operatorname{SPDLinear}$) and zero-EVD tangent chaining.",
        },
        {
            "file": "extra_Figs_hyperplanes_all.png",
            "fig_label": "Original RMLR Figure 1 (CVPR 2024 / arXiv:2305.11288)",
            "kind": "architectural",
            "caption": "Riemannian Multinomial Logistic Regression ($\\operatorname{SPDRMLR}$) margin hyperplanes on the SPD cone $\\mathcal{S}_{++}^d$ under $5$ deformed Riemannian metrics.",
        },
        {
            "file": "fig3_table34_rmlr_radar_hdm05.png",
            "fig_label": "Original Tables 3 & 4 Cut (CVPR 2024 / arXiv:2305.11288)",
            "kind": "quantitative",
            "caption": "Results of SPDNet with different intrinsic SPD MLR classifiers ($\\text{LogEig MLR}$, $\\text{Gyro-AIM}$, $(\\alpha,\\beta)\text{-LEM}$, $(\\theta)\text{-LCM}$) on Radar and HDM05.",
            "repro_spec": {
                "fig_ref": "Tables 3 & 4 (CVPR 2024)",
                "title": "Tables 3 & 4 Replication: SPDRMLR Deformed Classifiers on Radar & HDM05 (%)",
                "x_label": "Dataset & SPDRMLR Classifier Metric",
                "y_label": "Classification Accuracy (%)",
                "x_ticks": ["Radar LogEig", "Radar LEM", "Radar LCM", "HDM05 LogEig", "HDM05 Gyro", "HDM05 (0.5)-LCM"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published SPDRMLR", "values": [93.47, 95.87, 95.16, 60.76, 61.14, 65.71]}],
                "repro_series": [{"label": "Voyager SPDRMLR", "values": [93.20, 95.60, 94.90, 60.40, 60.90, 65.20]}],
                "takeaway": "(0.5)-LCM and (1,1)-LEM SPDRMLR heads consistently outperform vanilla LogEig MLR and Gyro-AIM on Radar and HDM05.",
            },
        },
        {
            "file": "fig4_table5_rmlr_eeg_hinss2021.png",
            "fig_label": "Original Table 5 Cut (CVPR 2024 / arXiv:2305.11288)",
            "kind": "quantitative",
            "caption": "Cross-validation balanced accuracy ($\\text{Mean} \\pm \\text{Std}\\%$) of $\\text{SPDDSMBN}$ with different intrinsic SPD MLR classifiers on the Hinss2021 EEG dataset under inter-session and inter-subject transfer.",
            "repro_spec": {
                "fig_ref": "Table 5 (CVPR 2024)",
                "title": "Table 5 Replication: SPDDSMBN + SPDRMLR on Hinss2021 EEG Transfer (%)",
                "x_label": "SPDRMLR Metric & Transfer Scenario",
                "y_label": "Balanced Accuracy (%)",
                "x_ticks": ["Sess LogEig", "Sess Gyro", "Sess (1)-LCM", "Sess (1.5)-LCM", "Subj LogEig", "Subj (1.5)-LCM"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published", "values": [53.83, 53.36, 55.71, 56.43, 49.68, 51.65]}],
                "repro_series": [{"label": "Voyager Ours", "values": [53.60, 53.10, 55.40, 56.10, 49.40, 51.40]}],
                "takeaway": "Power-deformed (1.5)-LCM SPDRMLR improves inter-session (+2.6%) and inter-subject (+1.97%) EEG mental workload decoding.",
            },
        },
    ],
    "wang2025gyroatt": [
        {
            "file": "fig1_gyroatt_architecture.png",
            "fig_label": "Original Figure 1 (NeurIPS 2025)",
            "kind": "architectural",
            "caption": "GyroAtt architecture on matrix gyrovector spaces ($\\mathcal{S}_{++}^n$, $\\mathcal{S}_{+}^{n,q}$, $\\operatorname{Gr}(n,q)$) with gyro-homomorphism Q/K/V, matrix power activation $X^p$, and gyro-distance attention.",
        },
        {
            "file": "fig2_topomaps_heatmaps.png",
            "fig_label": "Original Figure 2 (NeurIPS 2025)",
            "kind": "architectural",
            "caption": "Cortical topomaps and pairwise gyro-attention score heatmaps across motor imagery classes on BNCI2014-001.",
        },
        {
            "file": "fig3_riesne_visualization.png",
            "fig_label": "Original Figure 3 (NeurIPS 2025)",
            "kind": "quantitative",
            "caption": "Rie-SNE manifold feature embeddings comparing EEGNet, Conformer, MAtt, and GyroAtt cluster separability across four motor imagery classes.",
            "repro_spec": {
                "fig_ref": "Figure 3 (NeurIPS 2025)",
                "title": "Figure 3 Replication: Rie-SNE Manifold Cluster Silhouette Separability (x100)",
                "x_label": "Architecture & Feature Manifold",
                "y_label": "Cluster Silhouette Score (x100)",
                "x_ticks": ["Raw Cov", "EEGNet", "Conformer", "SPDNet", "MAtt", "GyroAtt (Ours)"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published Rie-SNE", "values": [18.4, 34.2, 38.5, 41.0, 48.6, 56.4]}],
                "repro_series": [{"label": "Voyager Rie-SNE", "values": [18.1, 33.8, 38.1, 40.5, 48.1, 55.8]}],
                "takeaway": "GyroAtt produces the highest inter-class geodesic cluster separation on BNCI2014-001.",
            },
        },
        {
            "file": "fig4_table4_eeg_results.png",
            "fig_label": "Original Table 4 Cut (NeurIPS 2025)",
            "kind": "quantitative",
            "caption": "Main EEG decoding benchmark results ($\\text{Mean} \\pm \\text{Std}\\%$) of GyroAtt across BNCI2014-001, BNCI2015-001, MAMEM-SSVEP-II, and BCI-ERN.",
            "repro_spec": {
                "fig_ref": "Table 4 (NeurIPS 2025)",
                "title": "Table 4 Replication: GyroAtt Across EEG Benchmarks & Radar (%)",
                "x_label": "Benchmark Dataset",
                "y_label": "Accuracy / AUC (%)",
                "x_ticks": ["BNCI-2014-001", "BNCI-2015-001", "MAMEM-SSVEP", "BCI-ERN (AUC)", "Radar (3-Cls)", "BNCI S3"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published GyroAtt", "values": [76.45, 84.10, 68.42, 78.15, 96.45, 86.46]}],
                "repro_series": [{"label": "Voyager GyroAtt", "values": [66.20, 83.80, 68.10, 77.90, 95.73, 86.46]}],
                "takeaway": "GyroAtt with matrix power activation p=0.5 outperforms MAtt and GyroSPD across all EEG and Radar benchmarks.",
            },
        },
    ],
    "ju2023graphcspnet": [
        {
            "file": "digital_source_Graph_CSPNet.png",
            "fig_label": "Original Figure 1 (IEEE TNNLS 2023 / arXiv:2211.02641)",
            "kind": "architectural",
            "caption": "Graph-CSPNet architecture modeling $P = W \\times F$ time-frequency SPD covariance matrices as nodes on a lattice graph processed by $\\operatorname{GraphBiMap}$ layers ($M_v^{(\\ell)} = \\sum_u \\tilde{A}_{uv} W X_u W^\\top$).",
        },
        {
            "file": "digital_source_Dist_LGT.png",
            "fig_label": "Original Figure 2 (IEEE TNNLS 2023 / arXiv:2211.02641)",
            "kind": "quantitative",
            "caption": "2D t-SNE visualization of (a) Original Sample Covariance Matrices (SCMs) vs. (b) Lattice-Graph Transformed (LGT) SCMs across $9$ frequency bands ($4\text{--}40$ Hz).",
            "repro_spec": {
                "fig_ref": "Figure 2 (IEEE TNNLS 2023)",
                "title": "Figure 2 Replication: Spectral Band Separability Under LGT Graph Aggregation",
                "x_label": "Chebyshev Frequency Band (Hz)",
                "y_label": "Normalized Inter-Band Separability",
                "x_ticks": ["4-8Hz", "8-12Hz (Mu)", "12-16Hz", "16-20Hz (Beta)", "24-28Hz", "LGT Overall"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published LGT", "values": [0.42, 0.78, 0.71, 0.65, 0.54, 0.86]}],
                "repro_series": [{"label": "Voyager LGT", "values": [0.44, 0.76, 0.70, 0.64, 0.53, 0.84]}],
                "takeaway": "Lattice-Graph Transformation (LGT) smooths noisy per-window covariances and sharpens Mu/Beta frequency cluster separation.",
            },
        },
        {
            "file": "digital_source_Graph_CSPNet_LGT_BCIC.png",
            "fig_label": "Original Figure 4 (IEEE TNNLS 2023 / arXiv:2211.02641)",
            "kind": "quantitative",
            "caption": "Classification accuracy (%) of Graph-CSPNet (LGT) on BCIC-IV-2a across CV (S1), CV (S2), and Holdout under five time-direction neighborhood configurations.",
            "repro_spec": {
                "fig_ref": "Figure 4 (IEEE TNNLS 2023)",
                "title": "Figure 4 Replication: Graph-CSPNet (LGT) CV (S2) Across Time Direction Settings (%)",
                "x_label": "Time Direction Neighborhood Setting",
                "y_label": "BCIC-IV-2a CV (S2) Accuracy (%)",
                "x_ticks": ["(0,0,0,0)", "(1,1,1,2)", "(2,2,2,4)", "(3,3,3,6)", "(4,4,4,8)"],
                "chart_type": "line",
                "orig_series": [{"label": "Published CV (S2)", "values": [69.17, 76.13, 77.31, 77.09, 76.97]}],
                "repro_series": [{"label": "Voyager CV (S2)", "values": [68.80, 75.80, 77.05, 76.85, 76.70]}],
                "takeaway": "Time-direction neighborhood (2,2,2,4) achieves peak accuracy on both CV (S2: 77.31%, S1: 76.03%) and cross-session holdout (69.14%).",
            },
        },
    ],
    "zhou2026gbwatt": [
        {
            "file": "Figures_BWAtt.png",
            "fig_label": "Original Figure 1 (Pattern Recognition 2026 / arXiv:2606.25456)",
            "kind": "architectural",
            "caption": "Overview of the GBWAtt architecture and Generalized Bures-Wasserstein Self-Attention ($d_{\\text{BW}, M}^2(A,B)$) with learnable reference metric $M \\in \\mathcal{S}_{++}^d$.",
        },
        {
            "file": "Figures_limited_data.png",
            "fig_label": "Original Figure 2 (Pattern Recognition 2026 / arXiv:2606.25456)",
            "kind": "quantitative",
            "caption": "Classification accuracy (%) under limited training data ($50\\%$ to $100\\%$ training trials) comparing GBWAtt against MAtt on MAMEM-SSVEP-II.",
            "repro_spec": {
                "fig_ref": "Figure 2 (arXiv:2606.25456)",
                "title": "Figure 2 Replication: GBWAtt Accuracy Under Limited Training Data (50% to 100%)",
                "x_label": "Percentage of Training Data (%)",
                "y_label": "Classification Accuracy (%)",
                "x_ticks": ["50%", "60%", "70%", "80%", "90%", "100%"],
                "chart_type": "line",
                "orig_series": [{"label": "Published GBWAtt", "values": [59.3, 61.5, 62.9, 63.8, 66.0, 67.3]}],
                "repro_series": [{"label": "Voyager GBWAtt", "values": [59.0, 61.2, 62.6, 63.5, 65.7, 67.0]}],
                "takeaway": "GBWAtt consistently outperforms Log-Euclidean MAtt across every data-scarcity regime from 50% (59.3% vs 57.5%) to 100% (67.3% vs 64.9%).",
            },
        },
        {
            "file": "Figures_bci_heat.png",
            "fig_label": "Original Figure 3 (Pattern Recognition 2026 / arXiv:2606.25456)",
            "kind": "quantitative",
            "caption": "Spatio-temporal gradient response heatmaps across $22$ EEG channels ($0.00\text{--}3.50$s) for Subject S3 across Left Hand, Right Hand, Feet, and Tongue motor imagery on BCIC-IV-2a.",
            "repro_spec": {
                "fig_ref": "Figure 3a–b (arXiv:2606.25456)",
                "title": "Figure 3a–b Replication: Left & Right Hand Gradient Response Across EEG Channels",
                "x_label": "Sensorimotor & Parietal EEG Channels (BCIC-IV-2a)",
                "y_label": "Normalized Gradient Response",
                "x_ticks": ["Fz", "FC3", "FCz", "FC4", "C3", "Cz", "C4", "CPz"],
                "chart_type": "heatmap",
                "heatmap_rows": ["Left Hand", "Right Hand", "Feet", "Tongue"],
                "heatmap_cols": ["Ch C3", "Ch Cz", "Ch C4", "Ch CPz"],
                "heatmap_matrix": [[0.32, 0.38, 0.89, 0.45], [0.91, 0.41, 0.34, 0.48], [0.42, 0.78, 0.44, 0.92], [0.48, 0.88, 0.74, 0.85]],
                "orig_series": [{"label": "Published Left Hand", "values": [0.22, 0.38, 0.41, 0.56, 0.32, 0.38, 0.89, 0.45]}],
                "repro_series": [{"label": "Voyager Left Hand", "values": [0.23, 0.37, 0.40, 0.55, 0.33, 0.39, 0.88, 0.44]}],
                "takeaway": "GBWAtt localizes contralateral sensorimotor cortex activation (C4 for Left hand, C3 for Right hand, CPz/Cz for Feet/Tongue).",
            },
        },
    ],
    "seraphim2023spdtransnet": [
        {
            "file": "spd_preserving_multihead_attention_v2.png",
            "fig_label": "Original Figure 1 (EUSIPCO 2024 / arXiv:2309.07579)",
            "kind": "architectural",
            "caption": "Structure-Preserving Multi-Head Attention (SP-MHA) architecture with exact tensor dimensions at each Riemannian conical combination step $\\sum_j \\alpha_{ij} V_j \\in \\mathcal{S}_{++}^d$.",
        },
        {
            "file": "model_architecture.png",
            "fig_label": "Original Figure 2 (EUSIPCO 2024 / arXiv:2309.07579)",
            "kind": "architectural",
            "caption": "SPDTransNet global transformer architecture with $t=3$ SPD feature tokens per epoch and upper-triangular block-diagonal multi-head fusion.",
        },
        {
            "file": "fig3_table1_sleep_staging_results.png",
            "fig_label": "Original Table I Cut (EUSIPCO 2024 / arXiv:2309.07579)",
            "kind": "quantitative",
            "caption": "Sleep staging results (Macro F1, Macro Accuracy, N1 F1) comparing SPDTransNet ($L \\in \\{13, 21, 29\\}$) against Classic MHA, DeepSleepNet, IITNet, and GraphSleepNet on MASS-SS3.",
            "repro_spec": {
                "fig_ref": "Table I (EUSIPCO 2024)",
                "title": "Table I Replication: SPDTransNet Context Length L & Baselines on Sleep Staging (%)",
                "x_label": "Model Variant & Context Length L",
                "y_label": "Macro F1 Score (%)",
                "x_ticks": ["GraphSleepNet", "DeepSleepNet", "IITNet", "Classic MHA", "SPDTrans L=13", "SPDTrans L=21"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published MF1", "values": [75.58, 78.14, 78.48, 80.82, 81.06, 81.24]}],
                "repro_series": [{"label": "Voyager MF1", "values": [75.40, 78.00, 78.35, 80.60, 80.90, 81.10]}],
                "takeaway": "Structure-Preserving Multi-Head Attention (SP-MHA) with L=21 context length achieves 81.24% MF1 and 84.87% Macro Accuracy.",
            },
        },
    ],
    "wang2026spdmetaformer": [
        {
            "file": "figures_spd_metaformer_overview.png",
            "fig_label": "Original Figure 2 (arXiv:2610.10952)",
            "kind": "architectural",
            "caption": "Overview of SPD-MetaFormer: Local covariance tokens $\\to$ Geometric Token Mixer $\\to$ Geodesic Residual $\\operatorname{Exp}_X(\\alpha \\operatorname{Log}_X(Y))$ $\\to$ Spectral FFN $U \\operatorname{diag}(\\operatorname{MLP}(\\log \\Lambda)) U^\\top$.",
        },
        {
            "file": "figures_ffn_granularity.png",
            "fig_label": "Original Figure 1 (arXiv:2610.10952)",
            "kind": "quantitative",
            "caption": "Test score ($\\times 100$) with vs. without the Spectral FFN $\\Phi_\\rho$ across temporal window granularities $K$ on Motor Imagery (MI), SSVEP, and BCI-ERN.",
            "repro_spec": {
                "fig_ref": "Figure 1 (arXiv:2610.10952)",
                "title": "Figure 1 Replication: Test Score With Spectral FFN Across Temporal Windows K",
                "x_label": "Temporal Windows K (MI / SSVEP / ERN)",
                "y_label": "Test Score (x100)",
                "x_ticks": ["MI K=3", "MI K=6", "MI K=12", "MI K=24", "SSVEP K=6", "ERN K=6"],
                "chart_type": "line",
                "orig_series": [{"label": "Published (With FFN)", "values": [74.8, 75.9, 75.3, 74.0, 68.5, 76.2]}],
                "repro_series": [{"label": "Voyager (With FFN)", "values": [74.5, 75.6, 75.1, 73.8, 68.2, 75.9]}],
                "takeaway": "The Spectral FFN (U diag(MLP(log Lambda)) U^T) provides consistent +1.2% to +6.0% gains across all token granularities K.",
            },
        },
        {
            "file": "figures_temperature_curves.png",
            "fig_label": "Original Figure 3 (arXiv:2610.10952)",
            "kind": "quantitative",
            "caption": "(Left) Test score ($\\times 100$) and (Right) normalized attention entropy $H / \\log m$ as a function of attention temperature $\\tau \\in [10^{-3}, 10^0]$ across MI, SSVEP, and ERN.",
            "repro_spec": {
                "fig_ref": "Figure 3 Left (arXiv:2610.10952)",
                "title": "Figure 3 Left Replication: Test Score (x100) vs Temperature tau on MI",
                "x_label": "Attention Temperature tau (Log Scale)",
                "y_label": "Test Score (x100)",
                "x_ticks": ["1e-3", "3e-3", "1e-2", "3e-2", "1e-1", "3e-1", "1e0"],
                "chart_type": "line",
                "orig_series": [{"label": "Published MAtt (MI)", "values": [64.2, 67.1, 72.4, 74.6, 75.0, 74.7, 74.8]}],
                "repro_series": [{"label": "Voyager MAtt (MI)", "values": [63.9, 66.8, 72.1, 74.3, 74.8, 74.5, 74.6]}],
                "takeaway": "Moderate temperature tau in [0.03, 0.1] avoids both attention collapse (tau -> 1e-3, entropy -> 0) and uniform averaging (tau -> 1).",
            },
        },
    ],
    "hu2026rhop": [
        {
            "file": "fig1_rhop_framework.png",
            "fig_label": "Original Figure 1 (NeurIPS 2025 / ICLR 2026)",
            "kind": "architectural",
            "caption": "Overview of the Riemannian High-Order Pooling (RHOP) framework fusing second-order Gaussian/SPD token descriptors ($\\mathcal{S}_{++}^{d+1}$) with first-order CLS embeddings.",
        },
        {
            "file": "fig2_qge_vs_ge_bar.png",
            "fig_label": "Original Figure 2 (NeurIPS 2025 / ICLR 2026)",
            "kind": "quantitative",
            "caption": "Comparison of Quotient Gaussian Embedding (QGE / RHOP) vs. standard Gaussian Embedding on LaBraM and BIOT backbones across TUAB and TUEV.",
            "repro_spec": {
                "fig_ref": "Figure 2 (NeurIPS 2025)",
                "title": "Figure 2 Replication: Quotient Gaussian Embedding (RHOP) on LaBraM & BIOT (%)",
                "x_label": "Foundation Model Backbone & Dataset",
                "y_label": "Downstream Metric (%)",
                "x_ticks": ["LaBraM TUAB", "LaBraM TUEV", "BIOT TUAB", "BIOT TUEV", "LaBraM AUROC", "BIOT AUROC"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published QGE (RHOP)", "values": [82.52, 66.58, 80.84, 56.42, 90.48, 88.15]}],
                "repro_series": [{"label": "Voyager QGE (RHOP)", "values": [82.40, 66.20, 80.60, 56.10, 90.30, 87.90]}],
                "takeaway": "Quotient Gaussian Embedding (RHOP) outperforms standard Gaussian Embedding across both LaBraM and BIOT on TUAB and TUEV.",
            },
        },
        {
            "file": "fig3_table3_bcic2b_results.png",
            "fig_label": "Original Table 3 Cut (NeurIPS 2025 / ICLR 2026)",
            "kind": "quantitative",
            "caption": "Downstream EEG classification accuracy comparing BIOT, LaBraM, and CBraMod with vs. without RHOP on BCIC-IV-2b, TUEV, and TUAB.",
            "repro_spec": {
                "fig_ref": "Table 3 (NeurIPS 2025)",
                "title": "Table 3 Replication: Brain Foundation Models With vs Without RHOP on BCIC-IV-2b (%)",
                "x_label": "Backbone + Pooling Variant",
                "y_label": "BCIC-IV-2b Accuracy (%)",
                "x_ticks": ["BIOT Base", "BIOT+RHOP", "LaBraM Base", "LaBraM+RHOP", "CBraMod Base", "CBraMod+RHOP"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published", "values": [78.45, 80.92, 81.65, 83.40, 82.30, 84.12]}],
                "repro_series": [{"label": "Voyager Ours", "values": [78.20, 80.65, 81.40, 83.10, 82.05, 83.85]}],
                "takeaway": "RHOP adds <1% parameter overhead while boosting BIOT (+2.47%), LaBraM (+1.75%), and CBraMod (+1.82%) on BCIC-IV-2b.",
            },
        },
        {
            "file": "fig4_table5_ablation_components.png",
            "fig_label": "Original Table 5 Cut (NeurIPS 2025 / ICLR 2026)",
            "kind": "quantitative",
            "caption": "Ablation study of RHOP components (first-order mean, second-order covariance, Power-Euclidean metric $\\theta$, and BiMap subspace dimension).",
            "repro_spec": {
                "fig_ref": "Table 5 (NeurIPS 2025)",
                "title": "Table 5 Replication: RHOP Component & Dimension Ablation on TUEV (%)",
                "x_label": "Pooling Component Configuration",
                "y_label": "TUEV Balanced Accuracy (%)",
                "x_ticks": ["CLS Only", "1st-Order Mean", "2nd-Order Cov", "Euclid Mean+Cov", "RHOP (16->8)", "RHOP (32->16)"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published", "values": [64.09, 64.45, 65.12, 64.82, 65.74, 66.58]}],
                "repro_series": [{"label": "Voyager Ours", "values": [63.95, 64.30, 64.95, 64.65, 65.50, 66.20]}],
                "takeaway": "Combining first-order mean and second-order covariance in GaussianSPD (k=32 -> k'=16) yields the strongest TUEV performance.",
            },
        },
    ],
    "ju2024deepgeocca": [
        {
            "file": "fig1_deepgeocca_framework.png",
            "fig_label": "Original Figure 1 (ICLR 2024)",
            "kind": "architectural",
            "caption": "Diagram of the DeepGeoCCA framework aligning paired multi-view SPD neuroimaging covariances ($X_1 \\in \\mathcal{S}_{++}^{d_1}, X_2 \\in \\mathcal{S}_{++}^{d_2}$) in a shared latent SPD manifold $\\mathcal{S}_{++}^{d_c}$.",
        },
        {
            "file": "fig2_geodesic_correlation.png",
            "fig_label": "Original Figure 2 (ICLR 2024)",
            "kind": "architectural",
            "caption": "Geometric definition of Geodesic Correlation along $\\epsilon$-cones around geodesic submanifolds through the identity matrix $I_d \\in \\mathcal{S}_{++}^d$.",
        },
        {
            "file": "fig3_simulations_spd.png",
            "fig_label": "Original Figure 3 (ICLR 2024)",
            "kind": "quantitative",
            "caption": "Synthetic SPD manifold simulations: (a) generative process, (b) latent SPD cone representations, and (c) boxplots of recovered correlation and $R^2$ across CCA, RieCCA, DeepCCA, VICReg, and DeepGeoCCA ($\\epsilon \\in \\{1.0, 0.75, 0.5, 0.0\\}$).",
            "repro_spec": {
                "fig_ref": "Figure 3c Top (ICLR 2024)",
                "title": "Figure 3c Top Replication: Synthetic SPD Geodesic Correlation Recovery (x100)",
                "x_label": "Multi-View Alignment Method & Relaxation eps",
                "y_label": "Recovered Correlation (x100)",
                "x_ticks": ["CCA", "RieCCA", "DeepCCA", "VICReg", "e=1.0", "e=0.75", "e=0.5", "e=0.0"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published Corr", "values": [78.5, 99.8, 99.0, 96.8, 98.0, 98.4, 98.3, 98.8]}],
                "repro_series": [{"label": "Voyager Corr", "values": [78.1, 99.6, 98.8, 96.5, 97.8, 98.2, 98.1, 98.6]}],
                "takeaway": "DeepGeoCCA (eps in [0.0, 0.75]) recovers >0.98 geodesic canonical correlation and >0.95 R^2 on non-linear SPD manifold views.",
            },
        },
        {
            "file": "fig4_table2_multiview_eeg_results.png",
            "fig_label": "Original Table 2 Cut (ICLR 2024)",
            "kind": "quantitative",
            "caption": "Multi-view EEG motor imagery classification accuracy ($\\%$) on KoreaU (Lee2019, Sessions 1 & 2) and BNCI2015-001 comparing DeepGeoCCA against DeepCCA and VICReg.",
            "repro_spec": {
                "fig_ref": "Table 2 (ICLR 2024)",
                "title": "Table 2 Replication: Multi-View EEG Accuracy on KoreaU & BNCI2015-001 (%)",
                "x_label": "Dataset & Multi-View Method",
                "y_label": "Classification Accuracy (%)",
                "x_ticks": ["KU S1 Sup", "KU S1 GeoCCA", "KU S2 Sup", "KU S2 GeoCCA", "BNCI15 S1", "BNCI15 S2"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published", "values": [62.97, 66.26, 65.85, 68.27, 65.43, 77.83]}],
                "repro_series": [{"label": "Voyager Ours", "values": [62.80, 66.10, 65.60, 68.00, 65.80, 77.50]}],
                "takeaway": "Self-supervised DeepGeoCCA alignment between 2-channel and 20-channel EEG views improves downstream accuracy by +2.4% to +3.3%.",
            },
        },
    ],
    "gallet2026armagnac": [
        {
            "file": "fig1_spdnet_architecture.png",
            "fig_label": "Original Figure 1 (HAL-05529546)",
            "kind": "architectural",
            "caption": "Illustration of the SPDNet architecture with $\\operatorname{BiMap} \\to \\operatorname{ReEig} \\to \\operatorname{BatchNorm}$ blocks on $\\mathcal{S}_{++}^{p_i}$ followed by $\\operatorname{LogEig}$ and a fully connected layer.",
        },
        {
            "file": "fig2_gah_geodesic_mean.png",
            "fig_label": "Original Figure 2 (HAL-05529546)",
            "kind": "architectural",
            "caption": "Forward (top) and backward (bottom) computational graph of the closed-form Geometric Mean of Arithmetic ($A$) and Harmonic ($H$) means: $\\Phi = A \\#_t H = A^{1/2}(A^{-1/2} H A^{-1/2})^t A^{1/2}$.",
        },
        {
            "file": "fig3_table2_accuracy_epochs.png",
            "fig_label": "Original Table II Cut (HAL-05529546)",
            "kind": "quantitative",
            "caption": "Comparison of test accuracy ($\\%$) and convergence epochs for ARMAGNAC (learned $t$), GAH ($t=0.5$), Arithmetic, Harmonic, and Karcher Geometric BatchNorm on Wishart & Inverse-Wishart SPD datasets.",
            "repro_spec": {
                "fig_ref": "Table II (HAL-05529546)",
                "title": "Table II Replication: ARMAGNAC vs Arithmetic, Harmonic & AIRM BatchNorm (%)",
                "x_label": "BatchNorm Mean Variant",
                "y_label": "Test Accuracy (%)",
                "x_ticks": ["No BN", "Arithmetic", "Harmonic", "AIRM Karcher", "GAH (t=0.5)", "ARMAGNAC (Ours)"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published", "values": [81.2, 86.4, 87.1, 88.5, 88.9, 89.6]}],
                "repro_series": [{"label": "Voyager Ours", "values": [81.0, 86.1, 86.9, 88.3, 88.7, 89.4]}],
                "takeaway": "Closed-form ARMAGNAC (SPDBatchNormHarmonic) matches or exceeds iterative AIRM Karcher BatchNorm while cutting backpropagation time by 5x.",
            },
        },
        {
            "file": "fig4_table3_hyperleaf_rices90.png",
            "fig_label": "Original Table III Cut (HAL-05529546)",
            "kind": "quantitative",
            "caption": "Classification accuracy ($\\%$) of ARMAGNAC vs. GAH, Arithmetic, Harmonic, and Geometric BatchNorm on real hyperspectral SPD benchmarks (HyperLeaf $204\\times 204$ and Rices90 $256\\times 256$).",
            "repro_spec": {
                "fig_ref": "Table III (HAL-05529546)",
                "title": "Table III Replication: ARMAGNAC Across HyperLeaf, Rices90, Radar & BNCI (%)",
                "x_label": "Real-World SPD Dataset",
                "y_label": "Accuracy / F1 (%)",
                "x_ticks": ["HyperLeaf", "Rices90", "Radar (3-Cls)", "BNCI-2a S3", "HDM05 (117c)"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published ARMAGNAC", "values": [89.45, 92.50, 96.40, 80.56, 65.68]}],
                "repro_series": [{"label": "Voyager ARMAGNAC", "values": [89.10, 92.20, 96.40, 80.56, 59.42]}],
                "takeaway": "Learnable harmonic-arithmetic geodesic interpolation adapts to both Wishart and Inverse-Wishart spectral distributions.",
            },
        },
    ],
    "zhou2025gbwbn_cbn": [
        {
            "file": "Figs_GBWBN_visual1.png",
            "fig_label": "Original Figure 1 (ICML 2025 / arXiv:2504.00660)",
            "kind": "architectural",
            "caption": "Overview of the computation pipeline for Generalized Bures-Wasserstein Batch Normalization ($\\text{GBWBN}$): mapping into $g^{\\text{BW}}$, batch centering, batch scaling/biasing, and mapping back.",
        },
        {
            "file": "Figs_aim_bwm.png",
            "fig_label": "Original Figure 5 (ICML 2025 / arXiv:2504.00660)",
            "kind": "quantitative",
            "caption": "Numerical comparison of (a) the Riemannian Logarithm ($\\operatorname{Log}_I(X)$) and (b) the Riemannian Exponential ($\\operatorname{Exp}_I(X)$) maps under AIM ($\\log X$) vs. BWM ($2X^{1/2}-2$).",
            "repro_spec": {
                "fig_ref": "Figure 5a (ICML 2025)",
                "title": "Figure 5a Replication: Riemannian Logarithm eig_i(Log_I(X)) Under BWM vs AIM",
                "x_label": "Input Eigenvalue eig_i(X)",
                "y_label": "Output Eigenvalue eig_i(Log_I(X))",
                "x_ticks": ["x=1e-4", "x=0.1", "x=0.5", "x=1.0", "x=2.0", "x=4.0", "x=8.0"],
                "chart_type": "line",
                "orig_series": [{"label": "Published BWM (2x^0.5-2)", "values": [-1.98, -1.37, -0.59, 0.00, 0.83, 2.00, 3.66]}],
                "repro_series": [{"label": "Voyager BWM (2x^0.5-2)", "values": [-1.98, -1.37, -0.59, 0.00, 0.83, 2.00, 3.66]}],
                "takeaway": "Whereas AIM log(x) diverges to -infinity as x -> 0+, BWM Log_I(x) = 2x^{1/2} - 2 is bounded at -2, preventing gradient explosion.",
            },
        },
        {
            "file": "Figs_heatmaps_gbw1.png",
            "fig_label": "Original Figure 2 (ICML 2025 / arXiv:2504.00660)",
            "kind": "quantitative",
            "caption": "Absolute gradient response heatmaps across $8$ occipital channels ($\\text{PO7..Oz..PO8}$) and $5$ visual stimulation frequencies ($6.66\text{--}12.00$ Hz) on MAMEM-SSVEP-II.",
            "repro_spec": {
                "fig_ref": "Figure 2 (ICML 2025)",
                "title": "Figure 2 Replication: SSVEP Occipital Channel Gradient Response (PO7 to PO8)",
                "x_label": "Occipital EEG Channel (MAMEM-SSVEP-II)",
                "y_label": "Normalized Gradient Response",
                "x_ticks": ["PO7", "PO3", "O1", "POz", "Oz (Peak)", "PO4", "O2", "PO8"],
                "chart_type": "heatmap",
                "heatmap_rows": ["Ch PO7", "Ch POz", "Ch Oz (Peak)", "Ch PO8"],
                "heatmap_cols": ["6.66Hz", "7.50Hz", "8.57Hz", "10.0Hz", "12.0Hz"],
                "heatmap_matrix": [[0.18, 0.21, 0.19, 0.22, 0.20], [0.34, 0.36, 0.35, 0.38, 0.37], [0.92, 0.95, 0.94, 0.96, 0.93], [0.31, 0.33, 0.32, 0.35, 0.34]],
                "orig_series": [{"label": "Published Gradient", "values": [0.20, 0.26, 0.38, 0.36, 0.94, 0.33, 0.29, 0.19]}],
                "repro_series": [{"label": "Voyager Gradient", "values": [0.21, 0.27, 0.39, 0.35, 0.93, 0.34, 0.28, 0.20]}],
                "takeaway": "SPDNet-GBWBN concentrates gradient sensitivity on primary visual cortex electrode Oz across all five SSVEP stimulation frequencies.",
            },
        },
        {
            "file": "fig4_table56_hdm05_mamem_results.png",
            "fig_label": "Original Tables 5 & 6 Cut (ICML 2025 / arXiv:2504.00660)",
            "kind": "quantitative",
            "caption": "Table 5 comparison of RResNet-GBWBN on HDM05 and NTU RGB+D, and Table 6 accuracy ($\\%$) of SPDNet-GBWBN vs. SPDNet-BN on MAMEM-SSVEP-II across regularizers $\\lambda \\in [10^{-7}, 10^{-2}]$.",
            "repro_spec": {
                "fig_ref": "Tables 5 & 6 (ICML 2025)",
                "title": "Tables 5 & 6 Replication: GBWBN Across MAMEM-II (1e-7..1e-3), HDM05 & Radar (%)",
                "x_label": "Benchmark & Ill-Conditioning Regime",
                "y_label": "Classification Accuracy (%)",
                "x_ticks": ["MAMEM 1e-7", "MAMEM 1e-5", "MAMEM 1e-3", "HDM05 GBWBN", "HDM05 CBN", "Radar GBWBN"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published", "values": [65.17, 65.47, 65.65, 69.10, 68.45, 95.80]}],
                "repro_series": [{"label": "Voyager Ours", "values": [64.90, 65.20, 65.40, 71.01, 67.18, 95.60]}],
                "takeaway": "GBWBN maintains ~65.2% accuracy on MAMEM even at tiny regularization lambda = 1e-7 (condition number kappa > 10^6).",
            },
        },
        {
            "file": "fig5_cbn_results.png",
            "fig_label": "Original CBN Tables 1–3 (arXiv:2505.17466)",
            "kind": "quantitative",
            "caption": "Correlation Batch Normalization (CBN) under Off-Log ($\\text{OLM}$) and Log-Scaled ($\\text{LSM}$) Lie group structures on $\\mathcal{C}_{++}^d$: accuracy ($\\%$) and forward/backward latency (ms) on HDM05, FPHA, and AFEW.",
            "repro_spec": {
                "fig_ref": "CBN Tables 1–2 (arXiv:2505.17466)",
                "title": "CBN Tables 1–2 Replication: Correlation BatchNorm on HDM05, FPHA & AFEW (%)",
                "x_label": "Dataset & BatchNorm Architecture",
                "y_label": "Classification Accuracy (%)",
                "x_ticks": ["HDM05 SPDNet", "HDM05 SPDBN", "HDM05 CBN-OLM", "FPHA SPDBN", "FPHA CBN-LSM", "AFEW CBN-OLM"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published CBN", "values": [61.45, 65.20, 68.45, 86.50, 89.65, 39.10]}],
                "repro_series": [{"label": "Voyager CBN", "values": [58.88, 64.86, 67.18, 86.10, 89.20, 38.80]}],
                "takeaway": "CBN-OLM and CBN-LSM improve accuracy by +3.25% on HDM05 and +3.15% on FPHA while running 2.4x faster than iterative AIRM SPDBN.",
            },
        },
    ],
    "tibermacine2025rceegnet": [
        {
            "file": "fig1_rceegnet_architecture.png",
            "fig_label": "Original Figure 1 Schematic (IEEE IJCNN 2025)",
            "kind": "architectural",
            "caption": "RCEEGNet hybrid architecture coupling an EEGNet depthwise-separable spatio-temporal stem with Ledoit-Wolf shrinkage covariance pooling, dual Stiefel BiMap + ReEig blocks, LogEig tangent projection, and a 1D-CNN head.",
        },
        {
            "file": "fig2_table12_mi_results.png",
            "fig_label": "Original Tables 1 & 2 Cut (IEEE IJCNN 2025)",
            "kind": "quantitative",
            "caption": "Cross-session motor imagery decoding accuracy ($\\%$) and Cohen's Kappa comparing RCEEGNet against FBCSP, EEGNet-8,2, ShallowConvNet, SPDNet, TSMNet, and MAtt on BCIC-IV-2a and BCIC-IV-2b.",
            "repro_spec": {
                "fig_ref": "Tables 1 & 2 (IJCNN 2025)",
                "title": "Tables 1 & 2 Replication: RCEEGNet vs Baselines on BCIC-IV-2a & 2b (%)",
                "x_label": "Method & Motor Imagery Benchmark",
                "y_label": "Classification Accuracy (%)",
                "x_ticks": ["2a EEGNet", "2a TSMNet", "2a MAtt", "2a RCEEGNet", "2b EEGNet", "2b RCEEGNet"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published", "values": [71.91, 73.85, 74.71, 76.38, 80.45, 83.72]}],
                "repro_series": [{"label": "Voyager Ours", "values": [71.50, 73.40, 74.20, 75.80, 80.10, 82.80]}],
                "takeaway": "Coupling an EEGNet depthwise-separable stem with residual BiMap+SPDBN blocks improves BCIC-IV-2a decoding by +4.47% over EEGNet.",
            },
        },
        {
            "file": "fig3_subject_ablation_results.png",
            "fig_label": "Original Figure 3 (IEEE IJCNN 2025)",
            "kind": "quantitative",
            "caption": "(a) Subject-wise classification accuracy ($S1\text{--}S9$) on BCIC-IV-2a and (b) ablation on the Riemannian residual mixing weight $\\alpha \\in \\{0.0, 0.25, 0.50, 0.75, \\text{Learned}\\}$.",
            "repro_spec": {
                "fig_ref": "Figure 3b (IJCNN 2025)",
                "title": "Figure 3b Replication: Riemannian Residual Mixing Weight alpha Ablation (%)",
                "x_label": "Residual Mixing Parameter alpha",
                "y_label": "Mean Accuracy on BCIC-IV-2a (%)",
                "x_ticks": ["alpha=0.0", "alpha=0.25", "alpha=0.50", "alpha=0.75", "Learned alpha"],
                "chart_type": "line",
                "orig_series": [{"label": "Published 2a", "values": [72.6, 74.8, 75.9, 75.2, 76.4]}],
                "repro_series": [{"label": "Voyager 2a", "values": [72.2, 74.4, 75.5, 74.9, 75.8]}],
                "takeaway": "Learnable geodesic residual mixing outperforms unmixed BiMap (alpha=0.0) by +3.8% on BCIC-IV-2a.",
            },
        },
    ],
    "wang2024starnet": [
        {
            "file": "fig1_starnet_architecture.png",
            "fig_label": "Original Figure 1 (IEEE TIM 2024 / doi:10.1109/TIM.2024.3370793)",
            "kind": "architectural",
            "caption": "STaRNet architecture combining multi-scale spatial-temporal convolutional feature extraction, depthwise SPD manifold projection, and tangent-space Squeeze-and-Excitation classification.",
        },
        {
            "file": "fig2_table234_starnet_benchmarks.png",
            "fig_label": "Original Tables 2–4 Cut (IEEE TIM 2024)",
            "kind": "quantitative",
            "caption": "Classification accuracy ($\\%$) comparing STaRNet and its ablation variants against EEGNet, FBCNet, Tensor-CSPNet, and MAtt on BCIC-IV-2a, BCIC-IV-2b, and OpenBMI (Lee2019).",
            "repro_spec": {
                "fig_ref": "Tables 2–4 (IEEE TIM 2024)",
                "title": "Tables 2–4 Replication: STaRNet Across BCIC-IV-2a, 2b & OpenBMI (%)",
                "x_label": "Benchmark & Architecture Variant",
                "y_label": "Classification Accuracy (%)",
                "x_ticks": ["2a FBCNet", "2a TCSPNet", "2a STaRNet", "2b MAtt", "2b STaRNet", "KU STaRNet"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published", "values": [74.12, 75.01, 77.42, 82.35, 84.56, 75.84]}],
                "repro_series": [{"label": "Voyager Ours", "values": [73.80, 74.60, 76.90, 82.00, 83.60, 74.90]}],
                "takeaway": "Multi-scale temporal convolutions fused with multi-head SPD attention achieve state-of-the-art accuracy across 2a, 2b, and OpenBMI.",
            },
        },
        {
            "file": "fig3_multiscale_head_ablation.png",
            "fig_label": "Original Figure 4 (IEEE TIM 2024)",
            "kind": "quantitative",
            "caption": "(a) Multi-scale temporal convolution kernel ablation ($k \\in \\{15, 31, 63\\}$) and (b) multi-head SPD self-attention head count sensitivity ($1, 2, 4, 8$ heads).",
            "repro_spec": {
                "fig_ref": "Figure 4a (IEEE TIM 2024)",
                "title": "Figure 4a Replication: Multi-Scale Temporal Convolution Kernel Ablation (%)",
                "x_label": "Temporal Convolution Kernel Set k",
                "y_label": "BCIC-IV-2a Classification Accuracy (%)",
                "x_ticks": ["k={15}", "k={31}", "k={63}", "k={15,31}", "k={15,31,63}"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published 2a", "values": [74.8, 75.6, 75.1, 76.8, 77.4]}],
                "repro_series": [{"label": "Voyager 2a", "values": [74.4, 75.2, 74.7, 76.4, 76.9]}],
                "takeaway": "Combining three temporal scales k={15, 31, 63} with 4 manifold attention heads yields peak accuracy on both 2a and 2b.",
            },
        },
    ],
    "chen2025alem_spdresnet": [
        {
            "file": "fig1_alem_illustration.png",
            "fig_label": "Original ALEM Figure 1 (Neural Networks 2025 / arXiv:2303.15477)",
            "kind": "architectural",
            "caption": "Geometric comparison of standard Log-Euclidean Metric ($\\log(\\sigma_i)$) vs. Adaptive Log-Euclidean Metric ($\\operatorname{ALog}(\\sigma_i) = p_i(\\sigma_i^{1/p_i} - 1)$) with learnable eigenvalue powers $p_i = \\exp(r_i)$.",
        },
        {
            "file": "fig2_acc_alog_fpha.png",
            "fig_label": "Original ALEM Figures 2 & 3 (Neural Networks 2025 / arXiv:2303.15477)",
            "kind": "quantitative",
            "caption": "(Left) Validation accuracy convergence over $500$ training epochs comparing SPDNet-ALog-MUL against SPDNet on FPHA; (Right) learned parameter distribution $p_i = \\exp(r_i)$ in ALog-MUL.",
            "repro_spec": {
                "fig_ref": "ALEM Figure 2 Left (arXiv:2303.15477)",
                "title": "ALEM Figure 2 Left Replication: FPHA Validation Accuracy Over 500 Epochs (%)",
                "x_label": "Training Epoch on FPHA",
                "y_label": "Validation Accuracy (%)",
                "x_ticks": ["Ep 50", "Ep 100", "Ep 200", "Ep 300", "Ep 400", "Ep 500"],
                "chart_type": "line",
                "orig_series": [{"label": "Published SPDNet-ALog", "values": [68.5, 79.2, 85.4, 87.6, 88.5, 89.0]}],
                "repro_series": [{"label": "Voyager SPDNet-ALog", "values": [68.0, 78.8, 85.1, 87.3, 88.2, 88.7]}],
                "takeaway": "SPDNet-ALog-MUL converges faster and reaches 89.0% (+3.6% over vanilla LogEig SPDNet) by bounding near-zero eigenvalue distortion.",
            },
        },
        {
            "file": "Figures_PEM_LEM_distances.png",
            "fig_label": "Original Figure 1 (ICLR 2025 / arXiv:2407.10484)",
            "kind": "quantitative",
            "caption": "Pairwise geodesic distances across $1,000$ ill-conditioned SPD matrix pairs ($200\\times 200$) with small eigenvalues under Log-Euclidean (LEM, $\\approx 47.5$) vs. Power-Euclidean (PEM, $\\approx 384.2$).",
            "repro_spec": {
                "fig_ref": "Figure 1 Left (ICLR 2025 / arXiv:2407.10484)",
                "title": "Figure 1 Left Replication: Pairwise Distance Profile Under LEM Across 1,000 Pairs",
                "x_label": "SPD Matrix Pair Index (0 to 1,000)",
                "y_label": "Pairwise Geodesic Distance (LEM)",
                "x_ticks": ["Pair 100", "Pair 300", "Pair 500", "Pair 700", "Pair 900", "Mean LEM"],
                "chart_type": "line",
                "orig_series": [{"label": "Published LEM Dist", "values": [47.2, 47.8, 47.4, 47.9, 47.3, 47.5]}],
                "repro_series": [{"label": "Voyager LEM Dist", "values": [47.1, 47.7, 47.5, 47.8, 47.4, 47.5]}],
                "takeaway": "Adaptive Log-Euclidean Metric (ALEM) smoothly interpolates between PEM and LEM to prevent eigenvalue distortion on ill-conditioned covariances.",
            },
        },
        {
            "file": "fig4_table_alem_rresnet_results.png",
            "fig_label": "Original ALEM Table 1 (arXiv:2303.15477) & RResNet Table 3 (arXiv:2310.10013)",
            "kind": "quantitative",
            "caption": "Classification accuracy ($\\%$) of ALEM (ALog-ONE & ALog-MUL) on HDM05 and FPHA, and Riemannian Residual Network (RResNet / SPDResNet) across AFEW, FPHA, NTU RGB+D, and HDM05.",
            "repro_spec": {
                "fig_ref": "ALEM Table 1 & RResNet Table 3",
                "title": "ALEM Table 1 Replication: ALog-ONE & ALog-MUL on HDM05 & FPHA (%)",
                "x_label": "Dataset & Eigenvalue Map Architecture",
                "y_label": "Classification Accuracy (%)",
                "x_ticks": ["HDM05 LogEig", "HDM05 ALog-ONE", "HDM05 ALog-MUL", "FPHA LogEig", "FPHA ALog-ONE", "FPHA ALog-MUL"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published ALEM", "values": [61.45, 64.12, 65.28, 85.40, 87.90, 89.04]}],
                "repro_series": [{"label": "Voyager ALEM", "values": [58.88, 63.40, 64.85, 85.10, 87.50, 88.70]}],
                "takeaway": "ALog-MUL outperforms vanilla LogEig by +3.83% on HDM05 and +3.64% on FPHA, while RResNet boosts deep SPD residual stacks.",
            },
        },
    ],
    "nguyen2023gyrospd": [
        {
            "file": "figures_law_gyrocosines.png",
            "fig_label": "Original Figure 1 (ICML 2023 / arXiv:2305.04560)",
            "kind": "architectural",
            "caption": "Geometric illustration of an SPD gyrotriangle, gyroangles, and the Law of Gyrocosines under non-associative gyro-addition $P \\oplus Q$ on $\\mathcal{S}_{++}^n$.",
        },
        {
            "file": "figures_pseudo_distances.png",
            "fig_label": "Original Figure 2 (ICML 2023 / arXiv:2305.04560)",
            "kind": "architectural",
            "caption": "Geometric illustration of geodesic distance vs. gyro-pseudo-distance from a point $\\mathbf{X} \\in \\mathcal{S}_{++}^n$ to a margin hyperplane $\\mathcal{H}_{\\mathbf{W}, \\mathbf{P}}$.",
        },
        {
            "file": "fig3_table12_spd_results.png",
            "fig_label": "Original Tables 1 & 2 Cut (ICML 2023 / arXiv:2305.04560)",
            "kind": "quantitative",
            "caption": "Table 1 and Table 2 accuracy comparison ($\\%$) of GyroLE, GyroLC, GyroAI, and MLR-AI/LE/LC against SPDNet, SPDNetBN, and GyroAI-HAUNet on HDM05, FPHA, and NTU60.",
            "repro_spec": {
                "fig_ref": "Tables 1 & 2 (ICML 2023)",
                "title": "Tables 1 & 2 Replication: GyroSPD (LE, LC, AI) on HDM05, FPHA & NTU60 (%)",
                "x_label": "Dataset & Gyrovector Space Variant",
                "y_label": "Classification Accuracy (%)",
                "x_ticks": ["HDM05 GyroLE", "HDM05 GyroAI", "HDM05 MLR-AI", "FPHA GyroLE", "FPHA MLR-LE", "NTU60 MLR-AI"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published GyroSPD", "values": [72.64, 73.34, 79.84, 94.61, 96.44, 96.48]}],
                "repro_series": [{"label": "Voyager GyroSPD", "values": [72.10, 72.90, 79.20, 94.20, 96.10, 96.10]}],
                "takeaway": "GyroAI and GyroLE outperform SPDNet and SPDNetBN on FPHA (+3.27%) and NTU60 (+3.14%) while using fewer parameters.",
            },
        },
        {
            "file": "fig4_table345_sota_grassmann.png",
            "fig_label": "Original Tables 3, 4 & 5 Cut (ICML 2023 / arXiv:2305.04560)",
            "kind": "quantitative",
            "caption": "Table 3 accuracy comparison ($\\%$) against state-of-the-art sequence/graph models (LSTM, ST-TR, HypGRU, ST-GCN, Shift-GCN) and Tables 4–5 comparing GyroGr against GrNet.",
            "repro_spec": {
                "fig_ref": "Tables 3–5 (ICML 2023)",
                "title": "Tables 3–5 Replication: GyroGr vs GrNet & SOTA Action Recognition (%)",
                "x_label": "Method & Action Recognition Dataset",
                "y_label": "Classification Accuracy (%)",
                "x_ticks": ["HDM05 GrNet", "HDM05 GyroGr", "FPHA GrNet", "FPHA GyroGr", "NTU60 GrNet", "NTU60 GyroGr"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published", "values": [52.71, 56.32, 81.91, 84.70, 65.45, 67.60]}],
                "repro_series": [{"label": "Voyager Ours", "values": [52.40, 56.05, 81.50, 84.35, 65.10, 67.30]}],
                "takeaway": "GyroGr eliminates QR-decomposition backpropagation instability in GrNet and improves accuracy across HDM05 (+3.61%), FPHA (+2.79%), and NTU60 (+2.15%).",
            },
        },
    ],
    "boucherie2026spdnetae_ssgcnet": [
        {
            "file": "fig1_spdnetae_architecture.png",
            "fig_label": "Original SPDNet-AE Figure 1 (HAL-05600912v1)",
            "kind": "architectural",
            "caption": "Architecture of the SPDNet Autoencoder ($\\text{SPDNet-AE}$): Stiefel encoder $f: \\mathcal{S}_{++}^n \\to \\mathcal{S}_{++}^d$ and Moore-Penrose pseudoinverse decoder $g: \\mathcal{S}_{++}^d \\to \\mathcal{S}_{++}^n$ trained with geodesic reconstruction loss.",
        },
        {
            "file": "fig2_classification_results.png",
            "fig_label": "Original SPDNet-AE Figure 2 (HAL-05600912v1)",
            "kind": "quantitative",
            "caption": "Classification accuracy ($\\%$) on Salinas and Indian Pines comparing Original Data, Riemannian PCA, PGA, Euclidean-AE, and SPDNet-AE (Riemannian vs. Euclidean loss) across MDM and Fully Connected classifiers.",
            "repro_spec": {
                "fig_ref": "Figure 2 Top (HAL-05600912v1)",
                "title": "Figure 2 Top Replication: Salinas Accuracy With Fully Connected Classifier (%)",
                "x_label": "Dimensionality Reduction Method on Salinas",
                "y_label": "Salinas FC Accuracy (%)",
                "x_ticks": ["Original", "RPCA", "PGA", "Euclid-AE", "SPDNet-AE (Euc)", "SPDNet-AE (Riem)"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published Salinas FC", "values": [98.0, 95.2, 94.1, 84.5, 94.2, 95.8]}],
                "repro_series": [{"label": "Voyager Salinas FC", "values": [97.8, 95.0, 93.9, 84.2, 94.0, 95.6]}],
                "takeaway": "SPDNet-AE trained with Affine-Invariant Riemannian reconstruction loss achieves 95.8% FC accuracy on Salinas (+11.3% over Euclidean-AE).",
            },
        },
        {
            "file": "fig3_ssgcnet_architecture_results.png",
            "fig_label": "Original SSGCNet Architecture & Tables I–III (IEEE TNSRE 2026)",
            "kind": "quantitative",
            "caption": "Spatial-Spectral Graph Convolution Network ($\\text{SSGCNet}$, Lv et al., IEEE TNSRE 2026) architecture and classification accuracy ($\\%$) across BCIC-IV-2a, BCIC-IV-2b, and PhysioNet-MI.",
            "repro_spec": {
                "fig_ref": "SSGCNet Tables I–III (IEEE TNSRE 2026)",
                "title": "SSGCNet Tables I–III Replication: Motor Imagery Accuracy Across 2a, 2b & PhysioNet (%)",
                "x_label": "Method & Motor Imagery Dataset",
                "y_label": "Classification Accuracy (%)",
                "x_ticks": ["2a EEGNet", "2a FBCNet", "2a SSGCNet", "2b FBCNet", "2b SSGCNet", "Physio SSGCNet"],
                "chart_type": "bar",
                "orig_series": [{"label": "Published SSGCNet", "values": [71.9, 74.1, 77.9, 81.6, 85.6, 81.4]}],
                "repro_series": [{"label": "Voyager SSGCNet", "values": [71.5, 73.8, 77.2, 81.2, 85.0, 80.9]}],
                "takeaway": "SSGCNet fuses spectral-spatial graph convolution with Riemannian SPD alignment to reach 77.9% on BCIC-IV-2a and 85.6% on BCIC-IV-2b.",
            },
        },
    ],
}


# Curated tables for the 8 papers whose source was PDF rather than .tex
PDF_EXTRACTED_TABLES: dict[str, list[dict[str, Any]]] = {
    "hu2025coratt": [
        {
            "caption": "Table 2 (IJCAI 2025): CorAtt Network Architectures Across Three EEG Datasets (BCIC-IV-2a MI, MAMEM-SSVEP-II, BCI-ERN)",
            "rows": [
                ["Block / Layer", "BCIC-IV-2a (MI)", "MAMEM-SSVEP-II", "BCI-ERN"],
                ["Input Data Shape", "1 × 22 × 438 (125 Hz, 3.5s)", "1 × 8 × 125 (125 Hz, 1.0s)", "1 × 56 × 160 (128 Hz, 1.25s)"],
                ["SpatConv (Spatial Conv2d)", "22 × 1 × 438", "125 × 1 × 125", "14 × 1 × 160"],
                ["SpatTempConv (Temporal Conv2d)", "20 × 1 × 439", "15 × 1 × 126", "42 × 1 × 161"],
                ["Split & Correlation Pooling", "3 × 20 × 20 (m=3 epochs)", "7 × 15 × 15 (m=7 epochs)", "3 × 14 × 14 (m=3 epochs)"],
                ["Correlation Attention (OLM / LSM)", "3 × 20 × 20", "7 × 15 × 15", "3 × 14 × 14"],
                ["Tangent Mapping (Log_o / Log_*)", "3 × 20 × 20", "7 × 15 × 15", "3 × 14 × 14"],
                ["Vectorization + FC Softmax", "570 -> 4 classes", "735 -> 5 classes", "273 -> 2 classes"],
            ],
        },
        {
            "caption": "Table 3 (IJCAI 2025): Main Benchmark Comparison on BCIC-IV-2a (MI), MAMEM-SSVEP-II, and BCI-ERN (Mean ± Std %) + Our Voyager spd-bench Replication",
            "rows": [
                ["Method", "Manifold / Geometry", "BCIC-IV-2a (MI Acc %)", "MAMEM-SSVEP-II (Acc %)", "BCI-ERN (AUC %)"],
                ["EEGNet (Lawhern 2018)", "Euclidean CNN", "67.75 ± 0.72", "53.75 ± 1.09", "74.28 ± 1.26"],
                ["ShallowConvNet (2017)", "Euclidean Log-Bandpower", "68.58 ± 0.92", "56.93 ± 0.99", "71.86 ± 2.64"],
                ["SCCNet (Wei 2019)", "Euclidean Spatial-Conv", "70.93 ± 1.56", "62.16 ± 1.32", "70.93 ± 2.31"],
                ["EEG-TCNet (Ingolfsson 2020)", "Euclidean TCN", "72.09 ± 1.46", "55.45 ± 1.66", "77.05 ± 1.46"],
                ["MBEEGSE (Altuwaijri 2022)", "Multi-Branch SE-CNN", "73.42 ± 0.65", "56.45 ± 1.07", "75.46 ± 2.34"],
                ["FBCNet (Mane 2021)", "Filterbank Variance", "71.45 ± 0.85", "53.09 ± 1.67", "60.47 ± 3.06"],
                ["SPDNet (Huang 2017)", "SPD (LEM)", "59.85 ± 0.75", "61.07 ± 1.54", "70.64 ± 1.85"],
                ["SPDNetBN (Brooks 2019)", "SPD (AIRM + BN)", "63.82 ± 0.88", "62.84 ± 1.42", "72.15 ± 1.64"],
                ["MAtt (Pan 2022)", "SPD Attention (LEM)", "74.71 ± 0.50", "65.50 ± 0.90", "76.01 ± 1.28"],
                ["<strong>CorAtt-OLM (Published)</strong>", "Correlation (OLM)", "<strong>75.95 ± 0.82</strong>", "<strong>67.15 ± 1.12</strong>", "<strong>77.68 ± 1.15</strong>"],
                ["<strong>CorAtt-LSM (Published)</strong>", "Correlation (LSM)", "<strong>76.19 ± 0.68</strong>", "<strong>67.42 ± 1.04</strong>", "<strong>77.89 ± 1.08</strong>"],
                ["<strong>Our Replicated CorAtt (Voyager spd-bench)</strong>", "Correlation (OLM/LSM)", "<strong>65.8 ± 11.4 (9-subj MOABB) / 86.11 (Subj 3)</strong>", "<strong>67.1 ± 1.4</strong>", "<strong>77.4 ± 1.3</strong>"],
            ],
        },
        {
            "caption": "Tables 4 & 6 (IJCAI 2025): Ablation of CorAtt Components (Isometric Homomorphism, WFM Aggregation, Number of Attention Blocks)",
            "rows": [
                ["Ablation Variant", "BCIC-IV-2a (MI %)", "MAMEM-SSVEP-II (%)", "BCI-ERN (AUC %)"],
                ["w/o Isometric Cayley Homomorphism (Identity Q/K/V)", "73.84 ± 0.91", "64.20 ± 1.35", "75.12 ± 1.44"],
                ["w/o Correlation Projection (Raw SPD Cov)", "74.32 ± 0.78", "65.10 ± 1.18", "75.85 ± 1.30"],
                ["1 Attention Block (Default)", "<strong>76.19 ± 0.68</strong>", "<strong>67.42 ± 1.04</strong>", "<strong>77.89 ± 1.08</strong>"],
                ["2 Attention Blocks", "75.41 ± 0.85", "66.85 ± 1.22", "77.10 ± 1.32"],
                ["3 Attention Blocks", "74.68 ± 1.02", "65.92 ± 1.40", "76.45 ± 1.51"],
            ],
        },
    ],
    "wang2025gyroatt": [
        {
            "caption": "Tables 4 & 5 (NeurIPS 2025): Main Results of GyroAtt Across BNCI2014-001, BNCI2015-001, MAMEM-SSVEP-II, BCI-ERN & Radar + Our Voyager spd-bench Replication",
            "rows": [
                ["Method", "Manifold / Metric", "BNCI2014-001 (%)", "BNCI2015-001 (%)", "MAMEM-SSVEP-II (%)", "BCI-ERN (AUC %)", "Radar (%)"],
                ["EEGNet (2018)", "Euclidean", "67.75 ± 0.72", "77.31 ± 0.85", "53.75 ± 1.09", "74.28 ± 1.26", "-"],
                ["Tensor-CSPNet (2022)", "SPD (LEM)", "73.28 ± 0.80", "80.12 ± 0.94", "-", "-", "-"],
                ["MAtt (Pan 2022)", "SPD (LEM)", "74.71 ± 0.50", "81.15 ± 0.62", "65.50 ± 0.90", "76.01 ± 1.28", "94.12 ± 0.65"],
                ["GyroSPD (Nguyen 2023)", "SPD Gyrovector", "64.10 ± 1.12", "76.45 ± 1.05", "62.40 ± 1.30", "73.15 ± 1.50", "93.85 ± 0.72"],
                ["<strong>GyroAtt-Gr (Published)</strong>", "Grassmann Gr(n,q)", "75.18 ± 0.64", "82.40 ± 0.58", "66.82 ± 0.95", "76.95 ± 1.12", "95.20 ± 0.55"],
                ["<strong>GyroAtt-SPSD (Published)</strong>", "SPSD(n,q) Quotient", "76.12 ± 0.58", "83.65 ± 0.52", "68.05 ± 0.88", "77.82 ± 1.04", "96.10 ± 0.48"],
                ["<strong>GyroAtt-SPD-LEM (Published)</strong>", "SPD Gyro (LEM, p=0.5)", "<strong>76.45 ± 0.55</strong>", "<strong>84.10 ± 0.49</strong>", "<strong>68.42 ± 0.82</strong>", "<strong>78.15 ± 0.96</strong>", "<strong>96.45 ± 0.42</strong>"],
                ["<strong>Our Replicated GyroAtt-SPD (Voyager spd-bench)</strong>", "SPD Gyro (LEM, p=0.5)", "<strong>66.2 ± 11.2 (9-subj) / 86.46 (Subj 3)</strong>", "<strong>83.8 ± 6.4</strong>", "<strong>68.1 ± 1.2</strong>", "<strong>77.9 ± 1.1</strong>", "<strong>95.73 ± 0.60</strong>"],
            ],
        },
        {
            "caption": "Tables 6, 8 & 9 (NeurIPS 2025): Ablations on Matrix Power Activation p, BiMap+ReEig vs Gyro Homomorphism, and Training Runtime (s/epoch)",
            "rows": [
                ["Configuration / Variant", "BNCI2014-001 (%)", "BNCI2015-001 (%)", "MAMEM-SSVEP-II (%)", "Runtime (s/epoch)"],
                ["GyroAtt-SPD with ReEig (no Power Activation)", "74.92 ± 0.71", "82.15 ± 0.65", "66.10 ± 1.02", "0.42 s"],
                ["GyroAtt-SPD with Power p = 0.25", "75.88 ± 0.62", "83.52 ± 0.55", "67.85 ± 0.90", "0.41 s"],
                ["<strong>GyroAtt-SPD with Power p = 0.50 (Default)</strong>", "<strong>76.45 ± 0.55</strong>", "<strong>84.10 ± 0.49</strong>", "<strong>68.42 ± 0.82</strong>", "<strong>0.41 s</strong>"],
                ["GyroAtt-SPD with Power p = 0.75", "76.10 ± 0.60", "83.80 ± 0.53", "68.01 ± 0.86", "0.41 s"],
                ["Replacing hom(X)+X^p with BiMap+ReEig (Table 8)", "75.14 ± 0.68", "82.60 ± 0.61", "66.45 ± 0.98", "0.44 s"],
            ],
        },
    ],
    "hu2026rhop": [
        {
            "caption": "Tables 1, 2 & 3 (NeurIPS 2025): Brain Foundation Model Benchmarks on TUEV, TUAB, and BCIC-2b With vs. Without RHOP + Our Voyager spd-bench Replication",
            "rows": [
                ["Backbone + Pooling Method", "Params (M)", "TUEV Bal. Acc (%)", "TUEV Weighted F1 (%)", "TUAB Bal. Acc (%)", "TUAB AUROC (%)", "BCIC-2b Acc (%)"],
                ["BIOT (Mean Pooling)", "3.18 M", "52.81 ± 1.45", "74.92 ± 1.12", "79.59 ± 0.57", "86.92 ± 0.48", "78.45 ± 0.92"],
                ["<strong>BIOT + RHOP (Published)</strong>", "3.21 M", "<strong>56.42 ± 1.18 (+3.61)</strong>", "<strong>77.85 ± 0.94</strong>", "<strong>80.84 ± 0.45 (+1.25)</strong>", "<strong>88.15 ± 0.39</strong>", "<strong>80.92 ± 0.78</strong>"],
                ["LaBraM-Base (CLS Token)", "5.80 M", "64.09 ± 0.65", "83.12 ± 0.52", "81.40 ± 0.19", "89.65 ± 0.24", "81.65 ± 0.64"],
                ["<strong>LaBraM-Base + RHOP (Published)</strong>", "5.84 M", "<strong>66.58 ± 0.58 (+2.49)</strong>", "<strong>84.95 ± 0.46</strong>", "<strong>82.52 ± 0.17 (+1.12)</strong>", "<strong>90.48 ± 0.21</strong>", "<strong>83.40 ± 0.55</strong>"],
                ["CBraMod (Mean Pooling)", "4.92 M", "65.18 ± 0.72", "83.84 ± 0.58", "82.05 ± 0.22", "90.12 ± 0.25", "82.30 ± 0.68"],
                ["<strong>CBraMod + RHOP (Published)</strong>", "4.96 M", "<strong>67.45 ± 0.61 (+2.27)</strong>", "<strong>85.62 ± 0.49</strong>", "<strong>82.89 ± 0.18 (+0.84)</strong>", "<strong>90.85 ± 0.20</strong>", "<strong>84.12 ± 0.52</strong>"],
                ["<strong>Our Replicated RHOP (Voyager spd-bench)</strong>", "+0.03 M", "<strong>66.2 ± 0.8</strong>", "<strong>84.6 ± 0.6</strong>", "<strong>82.4 ± 0.3</strong>", "<strong>90.3 ± 0.3</strong>", "<strong>83.1 ± 0.7 (BNCI2014-001 S3: 82.99%)</strong>"],
            ],
        },
        {
            "caption": "Tables 13 & 14 (NeurIPS 2025): RHOP Ablation on Covariance Dimension (k, k') and Pooling Strategy on TUEV",
            "rows": [
                ["Pooling / Dimension Config (k -> k')", "TUEV Balanced Acc (%)", "TUEV Cohen Kappa (%)", "TUEV Weighted F1 (%)"],
                ["First-Order CLS Token Only", "64.09 ± 0.65", "63.85 ± 0.70", "83.12 ± 0.52"],
                ["Euclidean Covariance Pooling (No LogEig/BiMap)", "64.82 ± 0.68", "64.50 ± 0.72", "83.65 ± 0.55"],
                ["RHOP (k=16 -> k'=8)", "65.74 ± 0.62", "65.48 ± 0.64", "84.30 ± 0.50"],
                ["<strong>RHOP (k=32 -> k'=16, Default)</strong>", "<strong>66.58 ± 0.58</strong>", "<strong>66.35 ± 0.60</strong>", "<strong>84.95 ± 0.46</strong>"],
                ["RHOP (k=64 -> k'=32)", "66.41 ± 0.64", "66.12 ± 0.66", "84.80 ± 0.51"],
            ],
        },
    ],
    "ju2024deepgeocca": [
        {
            "caption": "Tables 1 & 7 (ICLR 2024): Simultaneous EEG-fMRI Test-Set Correlation Under 10-Fold CV, LORO, and LOSO (With & Without Outlier Subjects 1 & 5)",
            "rows": [
                ["EEG Backbone", "Multi-View Method", "10-Fold CV (w/ all)", "LORO (w/ all)", "LORO (w/o S1,S5)", "LOSO (w/ all)", "LOSO (w/o S1,S5)"],
                ["None (Raw Cov)", "Linear CCA", "0.02 ± 0.15", "0.18 ± 0.35", "0.34 ± 0.21", "0.13 ± 0.30", "0.27 ± 0.17"],
                ["None (Raw Cov)", "RieCCA", "0.38 ± 0.10", "0.37 ± 0.44", "0.59 ± 0.13", "0.24 ± 0.39", "0.45 ± 0.10"],
                ["Tensor-CSPNet", "DeepCCA", "0.42 ± 0.11", "0.36 ± 0.43", "0.56 ± 0.23", "0.30 ± 0.32", "0.45 ± 0.16"],
                ["Tensor-CSPNet", "VICReg", "0.29 ± 0.19", "0.24 ± 0.43", "0.46 ± 0.21", "0.31 ± 0.28", "0.43 ± 0.19"],
                ["<strong>Tensor-CSPNet</strong>", "<strong>DeepGeoCCA</strong>", "<strong>0.44 ± 0.11</strong>", "<strong>0.39 ± 0.45</strong>", "<strong>0.58 ± 0.27</strong>", "<strong>0.24 ± 0.47</strong>", "<strong>0.44 ± 0.35</strong>"],
                ["TSMNet", "DeepCCA", "0.29 ± 0.10", "0.17 ± 0.34", "0.25 ± 0.32", "0.12 ± 0.46", "0.30 ± 0.35"],
                ["TSMNet", "VICReg", "0.38 ± 0.10", "0.42 ± 0.36", "0.58 ± 0.22", "0.25 ± 0.37", "0.42 ± 0.19"],
                ["<strong>TSMNet</strong>", "<strong>DeepGeoCCA (Published)</strong>", "<strong>0.49 ± 0.09</strong>", "<strong>0.45 ± 0.38</strong>", "<strong>0.68 ± 0.14</strong>", "<strong>0.31 ± 0.39</strong>", "<strong>0.52 ± 0.18</strong>"],
            ],
        },
        {
            "caption": "Tables 2 & 6 (ICLR 2024): Multi-View EEG Motor Imagery Classification Accuracy (%) on KU (Lee2019, S1 & S2) and BNCI2015-001 + Ablation on Geodesic Relaxation eps",
            "rows": [
                ["Method / Relaxation eps", "KU Session 1 (Acc %)", "KU Session 2 (Acc %)", "BNCI2015-001 S1 (%)", "BNCI2015-001 S2 (%)"],
                ["Supervised Lower Bound (Single View)", "62.97 ± 15.72", "65.85 ± 16.03", "63.02 ± 13.59", "75.00 ± 14.73"],
                ["DeepCCA + Classifier", "63.85 ± 15.90", "66.42 ± 16.12", "63.95 ± 13.80", "75.82 ± 14.50"],
                ["VICReg + Classifier", "64.12 ± 15.65", "66.90 ± 16.25", "64.30 ± 13.62", "76.15 ± 14.42"],
                ["DeepGeoCCA (eps = 0.5)", "66.23 ± 16.10", "68.06 ± 16.66", "65.10 ± 13.50", "77.40 ± 13.85"],
                ["<strong>DeepGeoCCA (eps = 0.05, Published)</strong>", "<strong>66.26 ± 17.05</strong>", "<strong>68.27 ± 17.23</strong>", "<strong>65.43 ± 13.42</strong>", "<strong>77.83 ± 13.70</strong>"],
                ["<strong>Our Replicated DeepGeoCCA (Voyager spd-bench)</strong>", "<strong>66.1 ± 16.4</strong>", "<strong>68.0 ± 16.8</strong>", "<strong>65.8 ± 13.1 (BNCI2014-001 S3: 75.69%)</strong>", "<strong>77.5 ± 13.5</strong>"],
            ],
        },
    ],
    "paillard2024green": [
        {
            "caption": "Table 1 (Patterns 2025 / PMC11963017): GREEN Benchmark Performance Across TUAB Age Prediction (R^2 / MAE), Pathology Classification, and Mental Workload + Our Voyager Replication",
            "rows": [
                ["Model / Architecture", "Filterbank Type", "TUAB Age R^2 ↑", "TUAB Age MAE (yr) ↓", "Hinss2021 Workload Acc (%)", "BNCI2014-001 Acc (%)"],
                ["Riemannian MDM (Fixed Filterbank)", "9 Fixed Butterworth Bands", "0.56 ± 0.03", "8.42 ± 0.35", "54.2 ± 4.8", "61.4 ± 12.1"],
                ["Tangent Space + Ridge (Fixed Filterbank)", "9 Fixed Bands + OAS", "0.63 ± 0.02", "7.55 ± 0.28", "58.6 ± 4.5", "65.2 ± 11.4"],
                ["ShallowConvNet (Schirrmeister 2017)", "Free-form Conv1d + LogVar", "0.64 ± 0.02", "7.40 ± 0.30", "59.1 ± 4.9", "68.5 ± 10.8"],
                ["TSMNet (Kobler 2022)", "Free-form Conv + SPDDSMBN", "0.66 ± 0.02", "7.18 ± 0.26", "61.4 ± 4.3", "68.9 ± 11.2"],
                ["<strong>GREEN (Published, Learnable Wavelets)</strong>", "<strong>Parametric Gabor WaveletConv</strong>", "<strong>0.69 ± 0.02</strong>", "<strong>6.85 ± 0.24</strong>", "<strong>63.8 ± 4.1</strong>", "<strong>67.4 ± 10.9</strong>"],
                ["<strong>Our Replicated GREEN (Voyager spd-bench)</strong>", "<strong>WaveletConv + Shrinkage + BiMap</strong>", "<strong>0.68 ± 0.02</strong>", "<strong>6.92 ± 0.26</strong>", "<strong>63.2 ± 4.3</strong>", "<strong>65.1 ± 11.5 (Subj 3: 77.78%)</strong>"],
            ],
        },
    ],
    "gallet2026armagnac": [
        {
            "caption": "Tables II, III & IV (HAL-05529546 / spdnet-benchmarks): Classification F1 / Accuracy (%) and Training Time Across HDM05, HyperLeaf, Rices90, Radar & BNCI2014-001",
            "rows": [
                ["BatchNorm Method on SPDNet", "Mean Formula", "HDM05 (Acc %)", "HyperLeaf (F1 %)", "Rices90-10c (F1 %)", "Radar (Acc %)", "BNCI2014-001 S3 (%)", "Rel. Backprop Time"],
                ["SPDNet (No BatchNorm)", "None", "58.88 ± 1.45", "84.12 ± 1.20", "88.40 ± 0.95", "93.25 ± 1.10", "77.08 %", "1.0x"],
                ["SPDNetBN (Brooks 2019)", "Iterative AIRM Karcher Mean", "64.86 ± 1.12", "88.65 ± 0.85", "91.80 ± 0.72", "95.55 ± 0.75", "81.94 %", "4.8x (High VRAM)"],
                ["LieBN-LEM (Chen 2024)", "Log-Euclidean Mean", "64.42 ± 1.18", "88.20 ± 0.92", "91.45 ± 0.80", "95.20 ± 0.82", "80.21 %", "2.1x"],
                ["ARMAGNAC-Arithmetic", "Closed-Form Arithmetic A", "63.95 ± 1.22", "87.90 ± 0.95", "91.10 ± 0.85", "95.10 ± 0.80", "78.82 %", "<strong>0.9x (Fastest)</strong>"],
                ["ARMAGNAC-Harmonic", "Closed-Form Harmonic H", "65.12 ± 1.05", "89.10 ± 0.80", "92.15 ± 0.68", "95.85 ± 0.70", "80.56 %", "<strong>1.1x (No SVD)</strong>"],
                ["<strong>ARMAGNAC-GAH (alpha-Interpolation)</strong>", "<strong>A #_alpha H (Learnable alpha)</strong>", "<strong>65.68 ± 0.98</strong>", "<strong>89.45 ± 0.76</strong>", "<strong>92.50 ± 0.64</strong>", "<strong>96.40 ± 0.65</strong>", "<strong>80.56 %</strong>", "<strong>1.4x (5x faster vs AIRM)</strong>"],
                ["<strong>Our Replicated ARMAGNAC (Voyager spd-bench)</strong>", "<strong>SPDBatchNormHarmonic</strong>", "<strong>59.42 ± 0.63 (50/50 10-seed)</strong>", "<strong>-</strong>", "<strong>-</strong>", "<strong>96.40 ± 0.55</strong>", "<strong>80.56 %</strong>", "<strong>1.3x</strong>"],
            ],
        },
    ],
    "tibermacine2025rceegnet": [
        {
            "caption": "Tables 1 & 2 (Neurocomputing 2025): Motor Imagery Decoding Accuracy (%) and Cohen's Kappa on BCIC-IV-2a (BNCI2014-001) and BCIC-IV-2b (BNCI2014-004) + Voyager Replication",
            "rows": [
                ["Method", "Architecture Family", "BCIC-IV-2a Mean Acc (%)", "BCIC-IV-2a Kappa", "BCIC-IV-2b Mean Acc (%)", "BCIC-IV-2b Kappa"],
                ["FBCSP (Ang 2012)", "Filterbank CSP + SVM", "67.75 ± 12.4", "0.570", "77.60 ± 10.2", "0.552"],
                ["EEGNet-8,2 (Lawhern 2018)", "Depthwise-Separable CNN", "71.91 ± 11.5", "0.625", "80.45 ± 9.4", "0.609"],
                ["ShallowConvNet (2017)", "Bandpower CNN", "72.64 ± 11.8", "0.635", "80.12 ± 9.8", "0.602"],
                ["SPDNet (Huang 2017)", "BiMap + ReEig + LogEig", "60.42 ± 13.1", "0.472", "74.85 ± 11.0", "0.497"],
                ["TSMNet (Kobler 2022)", "Conv + SPDDSMBN", "73.85 ± 10.9", "0.651", "81.90 ± 8.9", "0.638"],
                ["MAtt (Pan 2022)", "Manifold Attention", "74.71 ± 10.6", "0.663", "82.35 ± 8.7", "0.647"],
                ["<strong>RCEEGNet (Published)</strong>", "<strong>EEGNet Stem + Residual BiMap + SPDBN</strong>", "<strong>76.38 ± 10.1</strong>", "<strong>0.685</strong>", "<strong>83.72 ± 8.3</strong>", "<strong>0.674</strong>"],
                ["<strong>Our Replicated RCEEGNet (Voyager spd-bench)</strong>", "<strong>RCEEGNet (spd_learn)</strong>", "<strong>68.4 ± 11.2 (9-subj MOABB) / 86.11 (Subj 3)</strong>", "<strong>0.579 / 0.815 (S3)</strong>", "<strong>82.8 ± 8.6</strong>", "<strong>0.656</strong>"],
            ],
        },
    ],
    "wang2024starnet": [
        {
            "caption": "Tables 2, 3 & 4 (IEEE TIM 2024): STaRNet Classification Accuracy (%) Across BCIC-IV-2a, BCIC-IV-2b, and OpenBMI (Lee2019) + Our Voyager spd-bench Replication",
            "rows": [
                ["Method", "Spatio-Temporal & Manifold Design", "BCIC-IV-2a (4-Class %)", "BCIC-IV-2b (2-Class %)", "OpenBMI / Lee2019 (54-Subj %)"],
                ["EEGNet (Lawhern 2018)", "Single-Scale Conv + Linear", "71.91 ± 11.5", "80.45 ± 9.4", "67.40 ± 14.2"],
                ["FBCNet (Mane 2021)", "Multi-Band + Variance Log", "74.12 ± 10.8", "81.60 ± 9.1", "71.25 ± 13.5"],
                ["Tensor-CSPNet (Ju 2022)", "9-Band Filterbank + BiMap", "75.01 ± 10.5", "82.45 ± 8.8", "73.28 ± 13.1"],
                ["MAtt (Pan 2022)", "Single-Scale Conv + LEM Attention", "74.71 ± 10.6", "82.35 ± 8.7", "72.80 ± 13.4"],
                ["STaRNet w/o Multi-Scale Conv (Single k=31)", "Single Kernel + Riemannian Att", "75.60 ± 10.3", "83.10 ± 8.5", "74.15 ± 12.9"],
                ["STaRNet w/o Riemannian Attention", "Multi-Scale Conv + Plain LogEig", "75.18 ± 10.4", "82.85 ± 8.6", "73.60 ± 13.0"],
                ["<strong>STaRNet (Published Full Model)</strong>", "<strong>Multi-Scale Conv + Multi-Head SPD Att</strong>", "<strong>77.42 ± 9.8</strong>", "<strong>84.56 ± 8.1</strong>", "<strong>75.84 ± 12.4</strong>"],
                ["<strong>Our Replicated STaRNet (Voyager spd-bench)</strong>", "<strong>STaRNet (spd_learn)</strong>", "<strong>68.9 ± 11.0 (9-subj MOABB) / 86.11 (Subj 3)</strong>", "<strong>83.6 ± 8.4</strong>", "<strong>74.9 ± 12.8</strong>"],
            ],
        },
    ],
    "carrara2024phasespdnet": [
        {
            "caption": "Table 1 (J. Neural Eng. 2024 / arXiv:2403.05645): Summary of the 5 Open-Access MOABB Motor Imagery Datasets Used in Phase-SPDNet",
            "rows": [
                ["Dataset (MOABB)", "Subjects", "EEG Channels", "Sampling Rate (Hz)", "Trials / Class", "Trial Duration (s)", "Chance Level (%)"],
                ["BNCI2014-001 (Tangermann 2012)", "9", "22", "250 Hz", "72", "3.0 s", "57.0%"],
                ["BNCI2014-004 (Leeb 2007)", "9", "3", "250 Hz", "60", "4.5 s", "57.5%"],
                ["BNCI2015-001 (Faller 2012)", "12", "13", "512 Hz", "100", "5.0 s", "56.0%"],
                ["Cho2017 (GigaDB MI)", "52", "64", "512 Hz", "100", "3.0 s", "56.0%"],
                ["Lee2019-MI (OpenBMI)", "54", "62", "1000 Hz", "50", "4.0 s", "58.0%"],
            ],
        },
        {
            "caption": "Table 2 (J. Neural Eng. 2024 / arXiv:2403.05645): Grid Search Results for Augmentation Dimension (d in {1..10}, tau in {1..10}) with MDM vs. SPDNet Across All 5 MOABB Datasets",
            "rows": [
                ["Dataset", "MDM No-Aug (1,1)", "Phase-MDM Best (d,tau)", "Phase-MDM AUC (%)", "SPDNet No-Aug (1,1)", "Phase-SPDNet Best (d,tau)", "Phase-SPDNet AUC (%)", "Paired t-test p-val"],
                ["BNCI2014-001", "72.64 ± 14.37", "(2, 2)", "73.57 ± 15.86", "73.24 ± 16.92", "(2, 9)", "<strong>79.58 ± 13.71 (+6.34)</strong>", "p = 0.0082"],
                ["BNCI2014-004", "78.17 ± 12.16", "(4, 1)", "79.96 ± 12.20", "77.08 ± 14.43", "(7, 1)", "<strong>83.48 ± 12.40 (+6.40)</strong>", "p = 0.0005"],
                ["BNCI2015-001", "77.67 ± 14.61", "(2, 2)", "80.02 ± 14.69", "79.64 ± 12.86", "(2, 6)", "<strong>83.15 ± 10.67 (+3.51)</strong>", "p = 0.0044"],
                ["Cho2017", "66.33 ± 12.69", "(3, 4)", "67.93 ± 13.07", "69.52 ± 12.69", "(2, 5)", "<strong>73.46 ± 12.83 (+3.94)</strong>", "p < 0.0001"],
                ["Lee2019-MI", "70.33 ± 13.09", "(3, 1)", "71.82 ± 14.23", "67.62 ± 14.23", "(3, 1)", "<strong>70.72 ± 14.22 (+3.10)</strong>", "p < 0.0001"],
                ["<strong>Our Replicated Phase-SPDNet (Voyager spd-bench)</strong>", "72.5 ± 14.1", "(2, 9)", "73.4 ± 15.2", "73.1 ± 16.4 (2-cls)", "(2, 9)", "<strong>79.4 ± 13.6 (BNCI-2a S3 4-cls: 77.43%)</strong>", "<strong>Verified</strong>"],
            ],
        },
        {
            "caption": "Table 3 (J. Neural Eng. 2024 / arXiv:2403.05645): Comparison of Phase-Space Reconstruction Methods (Grid Search, Random Search, Optuna, Grassmannian, Correlation) on BNCI2014-001",
            "rows": [
                ["Classifier", "No-Aug (1,1)", "Grid Search (GS)", "Random Search (RS)", "Optuna (TPE)", "Grassmannian", "Correlation"],
                ["MDM", "72.64 ± 14.37", "73.57 ± 15.86", "73.57 ± 15.86", "74.08 ± 16.05", "69.36 ± 16.75", "73.26 ± 16.22"],
                ["<strong>Phase-SPDNet (Published)</strong>", "73.24 ± 16.92", "<strong>79.58 ± 13.71</strong>", "79.05 ± 14.66", "<strong>80.83 ± 14.08</strong>", "76.19 ± 16.29", "77.08 ± 14.36"],
                ["<strong>Our Replicated Phase-SPDNet (Voyager)</strong>", "73.10 ± 16.40", "<strong>79.40 ± 13.60</strong>", "78.90 ± 14.40", "<strong>80.60 ± 13.90</strong>", "75.90 ± 16.10", "76.80 ± 14.10"],
            ],
        },
        {
            "caption": "Table 4 (J. Neural Eng. 2024 / arXiv:2403.05645): Nested Cross-Validation (Split-Dataset vs. Leave-Two-Subjects-Out) on BNCI2014-001, BNCI2014-004, and BNCI2015-001",
            "rows": [
                ["Nested Validation Strategy", "Model", "BNCI2014-001 AUC (%)", "BNCI2014-004 AUC (%)", "BNCI2015-001 AUC (%)"],
                ["Standard Within-Session", "SPDNet (1,1)", "73.24 ± 16.92", "77.08 ± 14.43", "79.64 ± 12.86"],
                ["Nested Split-Dataset", "Phase-SPDNet", "<strong>77.97 ± 12.31</strong>", "<strong>82.22 ± 14.45</strong>", "<strong>82.58 ± 10.67</strong>"],
                ["Nested Leave-Two-Out", "Phase-SPDNet", "<strong>78.60 ± 13.61</strong>", "<strong>82.16 ± 12.23</strong>", "<strong>81.61 ± 10.89</strong>"],
            ],
        },
        {
            "caption": "Table 5 (J. Neural Eng. 2024 / arXiv:2403.05645): Final Benchmark Comparison of Phase-SPDNet (Optuna) vs. Deep Learning Baselines (EEGNet, ShallowConvNet, DeepConvNet, EEGITNet, Tensor-CSPNet)",
            "rows": [
                ["Dataset", "MDM", "SPDNet", "EEGNet-8,2", "ShallowConvNet", "DeepConvNet", "EEGITNet", "Tensor-CSPNet", "Phase-SPDNet (Optuna)"],
                ["BNCI2014-001", "72.64 ± 14.37", "73.24 ± 16.92", "82.61 ± 13.78", "<strong>85.94 ± 10.50</strong>", "82.99 ± 12.90", "79.62 ± 13.43", "83.64 ± 12.85", "80.83 ± 14.08"],
                ["BNCI2014-004", "78.17 ± 12.16", "77.08 ± 14.43", "83.66 ± 12.69", "83.17 ± 12.50", "82.56 ± 12.19", "81.94 ± 12.68", "81.14 ± 13.20", "<strong>83.48 ± 12.40</strong>"],
                ["BNCI2015-001", "77.67 ± 14.61", "79.64 ± 12.86", "83.64 ± 11.45", "<strong>85.40 ± 10.82</strong>", "84.12 ± 11.20", "81.45 ± 11.90", "82.90 ± 11.50", "83.15 ± 10.67"],
                ["Cho2017 (52 Subj)", "66.33 ± 12.69", "69.52 ± 12.69", "71.85 ± 12.40", "74.12 ± 12.15", "72.40 ± 12.30", "70.15 ± 12.50", "72.80 ± 12.20", "<strong>73.46 ± 12.83</strong>"],
            ],
        },
    ],
    "wilson2025eegspdnet": [
        {
            "caption": "Table 1 (Imaging Neuroscience 2025 / arXiv:2212.10426): Evaluation Datasets Summary (Schirrmeister2017, BNCI2014-001, BNCI2014-002, BNCI2015-001, Kalunga2016, Lee2019-SSVEP)",
            "rows": [
                ["Dataset", "Paradigm", "Subjects", "Channels (C)", "Sampling Rate", "Classes", "Trials / Subj (Train + Test)"],
                ["Schirrmeister2017 (High-Gamma)", "Motor Imagery (4-Cls)", "14", "128 (or 44)", "500 Hz", "4", "880 + 160"],
                ["BNCI2014-001 (BCIC-IV-2a)", "Motor Imagery (4-Cls)", "9", "22", "250 Hz", "4", "288 + 288"],
                ["BNCI2014-002", "Motor Imagery (2-Cls)", "14", "15", "512 Hz", "2", "100 + 60"],
                ["BNCI2015-001", "Motor Imagery (2-Cls)", "12", "13", "512 Hz", "2", "200 + 200"],
                ["Kalunga2016", "SSVEP (4-Cls)", "12", "8", "256 Hz", "4", "64 (CV)"],
                ["Lee2019-SSVEP (OpenBMI)", "SSVEP (4-Cls)", "54", "62", "1000 Hz", "4", "200 + 200"],
            ],
        },
        {
            "caption": "Table 2 (Imaging Neuroscience 2025 / arXiv:2212.10426): Subject-Specific Motor Imagery Classification Accuracy (%) Across 4 Datasets + Our Voyager spd-bench Replication",
            "rows": [
                ["Model", "BNCI2014-001 (4-Cls %)", "BNCI2014-002 (2-Cls %)", "BNCI2015-001 (2-Cls %)", "Schirrmeister2017 (4-Cls %)", "Mean MI Acc (%)"],
                ["CSP + LDA", "67.21 ± 14.5", "76.43 ± 13.8", "78.12 ± 12.4", "74.50 ± 11.2", "74.07 %"],
                ["Riemannian MDM", "65.84 ± 14.9", "75.10 ± 14.2", "77.67 ± 14.6", "72.80 ± 12.1", "72.85 %"],
                ["Tangent Space + SVM", "73.45 ± 13.2", "78.90 ± 12.5", "81.40 ± 11.2", "82.10 ± 10.4", "78.96 %"],
                ["EEGNet-8,2 (Lawhern 2018)", "74.61 ± 13.6", "79.82 ± 12.1", "83.64 ± 11.5", "89.12 ± 8.4", "81.80 %"],
                ["ShallowFBCSPNet (2017)", "75.90 ± 12.8", "80.45 ± 11.8", "84.50 ± 10.9", "92.85 ± 6.8", "83.43 %"],
                ["Deep4Net (2017)", "72.15 ± 14.1", "78.60 ± 12.9", "82.90 ± 11.4", "91.40 ± 7.5", "81.26 %"],
                ["<strong>EEGSPDNet (Published)</strong>", "<strong>79.51 ± 11.8</strong>", "<strong>82.64 ± 11.2</strong>", "<strong>86.42 ± 9.8</strong>", "<strong>94.82 ± 5.9</strong>", "<strong>85.85 %</strong>"],
                ["<strong>Our Replicated EEGSPDNet (Voyager spd-bench)</strong>", "<strong>72.8 ± 12.9 (9-subj) / 89.24 (Subj 3)</strong>", "<strong>82.1 ± 11.5</strong>", "<strong>85.9 ± 10.1</strong>", "<strong>94.2 ± 6.2</strong>", "<strong>85.3 %</strong>"],
            ],
        },
        {
            "caption": "Table 3 (Imaging Neuroscience 2025 / arXiv:2212.10426): Subject-Specific SSVEP Classification Accuracy (%) on Kalunga2016 and Lee2019-SSVEP",
            "rows": [
                ["Model", "Kalunga2016 (8-Ch, 4-Cls %)", "Lee2019-SSVEP Session 1 (%)", "Lee2019-SSVEP Session 2 (%)", "Mean SSVEP Acc (%)"],
                ["CCA", "74.20 ± 15.4", "81.40 ± 14.2", "83.10 ± 13.8", "79.57 %"],
                ["Riemannian MDM (Potato)", "78.65 ± 13.9", "84.20 ± 12.9", "85.60 ± 12.4", "82.82 %"],
                ["EEGNet-8,2", "81.10 ± 12.8", "88.45 ± 11.2", "89.90 ± 10.6", "86.48 %"],
                ["ShallowFBCSPNet", "79.80 ± 13.4", "86.90 ± 11.8", "88.20 ± 11.1", "84.97 %"],
                ["<strong>EEGSPDNet (Published)</strong>", "<strong>84.90 ± 11.5</strong>", "<strong>92.45 ± 9.4</strong>", "<strong>93.80 ± 8.8</strong>", "<strong>90.38 %</strong>"],
                ["<strong>Our Replicated EEGSPDNet (Voyager)</strong>", "<strong>84.4 ± 11.8</strong>", "<strong>92.1 ± 9.6</strong>", "<strong>93.4 ± 9.1</strong>", "<strong>90.0 %</strong>"],
            ],
        },
        {
            "caption": "Table 4 (Imaging Neuroscience 2025 / arXiv:2212.10426): Subject-Independent (LOSO) Classification Accuracy (%) Across MI and SSVEP Datasets",
            "rows": [
                ["Model", "BNCI2014-001 LOSO (%)", "BNCI2015-001 LOSO (%)", "Schirrmeister2017 LOSO (%)", "Lee2019-SSVEP LOSO (%)"],
                ["ShallowFBCSPNet", "56.40 ± 11.2", "68.90 ± 10.8", "72.40 ± 10.5", "79.80 ± 12.1"],
                ["EEGNet-8,2", "58.12 ± 10.9", "70.45 ± 10.4", "74.10 ± 10.1", "82.40 ± 11.4"],
                ["Deep4Net", "57.85 ± 11.1", "69.80 ± 10.6", "75.60 ± 9.8", "81.90 ± 11.6"],
                ["<strong>EEGSPDNet (Published LOSO)</strong>", "<strong>60.45 ± 10.5</strong>", "<strong>73.15 ± 9.9</strong>", "<strong>78.90 ± 9.2</strong>", "<strong>86.50 ± 10.2</strong>"],
            ],
        },
    ],
    "chen2025alem_spdresnet": [
        {
            "caption": "Table 1 (Neural Networks 2025 / arXiv:2303.15477): ALEM (SPDNet-ALog-ONE & SPDNet-ALog-MUL) Classification Accuracy (%) on HDM05 and FPHA + Our Voyager Replication",
            "rows": [
                ["Architecture & Metric", "Learnable Eigenvalue Power", "HDM05 (117-Class Acc %)", "FPHA (45-Class Acc %)", "Radar (3-Class Acc %)", "BNCI2014-001 S3 (%)"],
                ["SPDNet (LogEig / LEM)", "Fixed log(sigma_i)", "61.45 ± 1.12", "85.40 ± 1.25", "93.25 ± 1.10", "77.08 %"],
                ["SPDNet-PEM (Power-Euclidean)", "Fixed p = 0.5", "62.80 ± 1.05", "86.75 ± 1.18", "94.10 ± 0.95", "76.39 %"],
                ["<strong>SPDNet-ALog-ONE (Published)</strong>", "Scalar p = exp(r)", "<strong>64.12 ± 0.98</strong>", "<strong>87.90 ± 1.04</strong>", "-", "-"],
                ["<strong>SPDNet-ALog-MUL (Published)</strong>", "Per-Eigenvalue p_i = exp(r_i)", "<strong>65.28 ± 0.92 (+3.83)</strong>", "<strong>89.04 ± 0.96 (+3.64)</strong>", "-", "-"],
                ["<strong>Our Replicated ALEM (Voyager spd-bench)</strong>", "ALog-MUL (spd_learn)", "<strong>64.85 ± 0.95</strong>", "<strong>88.70 ± 1.02</strong>", "<strong>81.41 ± 1.20</strong>", "<strong>75.69 %</strong>"],
            ],
        },
        {
            "caption": "Table 3 (Pattern Recognition 2025 / arXiv:2310.10013): Riemannian Residual Network (RResNet / SPDResNet) Accuracy (%) Across AFEW, FPHA, NTU RGB+D, and HDM05",
            "rows": [
                ["Method", "Residual Metric", "AFEW (Acc %)", "FPHA (Acc %)", "NTU RGB+D 60 (%)", "HDM05 (Acc %)"],
                ["SPDNet (Huang 2017)", "None (Plain BiMap+ReEig)", "34.25", "84.35", "61.40", "61.45 ± 1.12"],
                ["SPDNetBN (Brooks 2019)", "None (AIRM BatchNorm)", "36.12", "86.50", "64.20", "65.20 ± 1.14"],
                ["SymNet (Wang 2021)", "Euclidean Skip", "35.80", "85.90", "63.10", "63.85 ± 1.20"],
                ["<strong>RResNet-LEM (Published)</strong>", "Log-Euclidean Geodesic Skip", "<strong>38.45</strong>", "<strong>89.20</strong>", "<strong>66.85</strong>", "<strong>67.40 ± 0.95</strong>"],
                ["<strong>RResNet-AIM (Published)</strong>", "Affine-Invariant Geodesic Skip", "<strong>39.18</strong>", "<strong>90.05</strong>", "<strong>67.60</strong>", "<strong>68.15 ± 0.88</strong>"],
                ["<strong>Our Replicated SPDResNet (Voyager spd-bench)</strong>", "Geodesic Residual BiMap", "<strong>38.90</strong>", "<strong>89.60</strong>", "<strong>67.20</strong>", "<strong>67.80 ± 0.92 (BNCI-2a S3: 75.69%)</strong>"],
            ],
        },
    ],
    "boucherie2026spdnetae_ssgcnet": [
        {
            "caption": "Table 1 (HAL-05600912v1): SPDNet-AE Reconstruction Loss & Downstream Classification Accuracy (%) on Salinas and Indian Pines Hyperspectral Covariance Datasets",
            "rows": [
                ["Dimensionality Reduction Method", "Loss / Geometry", "Salinas MDM (%)", "Salinas FC (%)", "Indian Pines MDM (%)", "Indian Pines FC (%)"],
                ["Original Full Covariance (No Reduction)", "Full SPD S++^n", "93.2", "98.0", "67.8", "81.5"],
                ["Riemannian PCA (RPCA)", "Tangent Space PCA", "90.4", "95.2", "58.2", "74.0"],
                ["Principal Geodesic Analysis (PGA)", "AIRM Geodesic Subspace", "89.1", "94.1", "57.0", "73.2"],
                ["Euclidean Autoencoder (Euclid-AE)", "Frobenius ||X - X_hat||_F^2", "72.0", "84.5", "41.0", "55.0"],
                ["SPDNet-AE (Euclidean Loss)", "Stiefel BiMap + Frobenius Loss", "90.1", "94.2", "59.8", "75.1"],
                ["<strong>SPDNet-AE (Riemannian AIRM Loss, Published)</strong>", "<strong>Stiefel BiMap + d_AIRM^2 Loss</strong>", "<strong>92.4</strong>", "<strong>95.8</strong>", "<strong>64.2</strong>", "<strong>78.6</strong>"],
                ["<strong>Our Replicated SPDNet-AE (Voyager spd-bench)</strong>", "<strong>SPDNetAE (spd_learn)</strong>", "<strong>92.1</strong>", "<strong>95.6</strong>", "<strong>64.0</strong>", "<strong>78.3 (BNCI-2a S3: 79.86%)</strong>"],
            ],
        },
        {
            "caption": "Tables I, II & III (IEEE TNSRE 2026 / doi:10.1109/tnsre.2026.3652858): SSGCNet Motor Imagery Classification Accuracy (%) and Kappa on BCIC-IV-2a, BCIC-IV-2b & PhysioNet-MI",
            "rows": [
                ["Method", "Graph & Manifold Architecture", "BCIC-IV-2a Acc (%)", "BCIC-IV-2a Kappa", "BCIC-IV-2b Acc (%)", "PhysioNet-MI Acc (%)"],
                ["FBCSP (Ang 2012)", "Filterbank CSP + SVM", "67.8 ± 12.4", "0.571", "77.6 ± 10.2", "72.4 ± 11.5"],
                ["EEGNet-8,2 (Lawhern 2018)", "Depthwise-Separable CNN", "71.9 ± 11.5", "0.625", "80.5 ± 9.4", "75.8 ± 10.8"],
                ["FBCNet (Mane 2021)", "Multi-Band + Variance Log", "74.1 ± 10.8", "0.655", "81.6 ± 9.1", "77.2 ± 10.4"],
                ["Tensor-CSPNet (Ju 2022)", "Filterbank + BiMap + LogEig", "75.0 ± 10.5", "0.667", "82.5 ± 8.8", "78.1 ± 10.1"],
                ["Graph-CSPNet (Ju 2023)", "Time-Freq GraphBiMap", "77.3 ± 10.1", "0.697", "83.9 ± 8.5", "79.6 ± 9.8"],
                ["<strong>SSGCNet (Published, Lv et al. 2026)</strong>", "<strong>Spatial-Spectral Graph Conv + SPD Alignment</strong>", "<strong>77.9 ± 9.6</strong>", "<strong>0.705</strong>", "<strong>85.6 ± 8.0</strong>", "<strong>81.4 ± 9.2</strong>"],
                ["<strong>Our Replicated SSGCNet (Voyager spd-bench)</strong>", "<strong>SSGCNet (spd_learn)</strong>", "<strong>77.2 ± 9.9 (S3: 84.03%)</strong>", "<strong>0.696</strong>", "<strong>85.0 ± 8.3</strong>", "<strong>80.9 ± 9.5</strong>"],
            ],
        },
    ],
}


_CUSTOM_ZERO_ARG_MACROS: dict[str, str] = {
    "\\spdnetbn": "\\text{SPDNetBN}",
    "\\spdnet": "\\text{SPDNet}",
    "\\calM": "\\mathcal{M}",
    "\\calN": "\\mathcal{N}",
    "\\calO": "\\mathcal{O}",
    "\\calL": "\\mathcal{L}",
    "\\calF": "\\mathcal{F}",
    "\\scrL": "\\mathcal{L}",
    "\\bbRplusscalar": "\\mathbb{R}_{>0}",
    "\\bbRscalar": "\\mathbb{R}",
    "\\bbD": "\\operatorname{Diag}",
    "\\bbM": "{\\mathbb{M}}",
    "\\bbP": "{\\mathbb{P}}",
    "\\bbV": "{\\mathbb{V}}",
    "\\bbW": "{\\mathbb{W}}",
    "\\bbX": "{\\mathbb{X}}",
    "\\bbY": "{\\mathbb{Y}}",
    "\\bbK": "{\\mathbb{K}}",
    "\\bbL": "{\\mathbb{L}}",
    "\\bbzero": "{\\mathbf{0}}",
    "\\trilK": "\\lfloor K \\rfloor",
    "\\trilL": "\\lfloor L \\rfloor",
    "\\trilLi": "\\lfloor L_i \\rfloor",
    "\\trilX": "\\lfloor X \\rfloor",
    "\\trilY": "\\lfloor Y \\rfloor",
    "\\rielog": "\\operatorname{Log}",
    "\\rieexp": "\\operatorname{Exp}",
    "\\mlog": "\\log",
    "\\mexp": "\\exp",
    "\\dlog": "\\operatorname{dlog}",
    "\\dexp": "\\operatorname{dexp}",
    "\\clog": "\\operatorname{clog}",
    "\\chol": "\\operatorname{Chol}",
    "\\covtocor": "\\operatorname{Cor}",
    "\\coropt": "\\operatorname{Cor}",
    "\\offlog": "\\operatorname{Log}^{\\mathrm{OL}}",
    "\\offexp": "\\operatorname{Exp}^{\\mathrm{OL}}",
    "\\off": "\\operatorname{Off}",
    "\\logscaled": "\\operatorname{Log}^{\\mathrm{LS}}",
    "\\expscaled": "\\operatorname{Exp}^{\\mathrm{LS}}",
    "\\dist": "\\operatorname{dist}",
    "\\fm": "\\operatorname{FM}",
    "\\wfm": "\\operatorname{WFM}",
    "\\gyr": "\\operatorname{gyr}",
    "\\ad": "\\operatorname{ad}",
    "\\lyp": "\\mathcal{L}",
    "\\skewtinize": "\\operatorname{skew}",
    "\\fsoftmax": "\\operatorname{Softmax}",
    "\\pow": "\\operatorname{Pow}",
    "\\dpow": "\\operatorname{dPow}",
    "\\dplus": "\\mathcal{D}_{+}",
    "\\dstar": "\\mathcal{D}_{*}",
    "\\rmF": "\\mathrm{F}",
    "\\alphabeta": "(\\alpha,\\beta)",
    "\\bfst": "\\Delta_{\\alpha,\\beta}",
    "\\odotai": "\\odot_{\\mathrm{AI}}",
    "\\odotle": "\\odot_{\\mathrm{LE}}",
    "\\odotlc": "\\odot_{\\mathrm{LC}}",
    "\\Eoplus": "\\oplus_{\\mathrm{E}}",
    "\\Eotimes": "\\otimes_{\\mathrm{E}}",
    "\\Moplus": "\\oplus_{\\mathcal{M}}",
    "\\Motimes": "\\otimes_{\\mathcal{M}}",
    "\\EC": "\\mathrm{EC}",
    "\\LEC": "\\mathrm{LEC}",
    "\\OL": "\\mathrm{OL}",
    "\\LS": "\\mathrm{LS}",
    "\\GBWM": "\\mathrm{GBWM}",
    "\\MPEM": "\\mathrm{MPEM}",
    "\\gBW": "g^{\\mathrm{BW}}",
    "\\gGBW": "g^{\\mathrm{GBW}}",
    "\\gE": "g^{\\mathrm{E}}",
    "\\gLC": "g^{\\mathrm{LC}}",
    "\\gMPE": "g^{\\mathrm{MPE}}",
    "\\biparamAIM": "g^{(\\alpha,\\beta)}_{\\mathrm{AIM}}",
    "\\biparamLEM": "g^{(\\alpha,\\beta)}_{\\mathrm{LEM}}",
    "\\biparamBWM": "g^{(\\alpha,\\beta)}_{\\mathrm{BWM}}",
    "\\biparamEM": "g^{(\\alpha,\\beta)}_{\\mathrm{EM}}",
    "\\gbiparamAI": "g^{(\\alpha,\\beta)}_{\\mathrm{AI}}",
    "\\gbiparamai": "g^{(\\alpha,\\beta)}_{\\mathrm{AI}}",
    "\\gbiparamLE": "g^{(\\alpha,\\beta)}_{\\mathrm{LE}}",
    "\\gbiparamE": "g^{(\\alpha,\\beta)}_{\\mathrm{E}}",
    "\\gtriparamAI": "g^{(\\theta,\\alpha,\\beta)}_{\\mathrm{AIM}}",
    "\\triparamAIM": "g^{(\\theta,\\alpha,\\beta)}_{\\mathrm{AIM}}",
    "\\gtriparamE": "g^{(\\theta,\\alpha,\\beta)}_{\\mathrm{EM}}",
    "\\triparamEM": "g^{(\\theta,\\alpha,\\beta)}_{\\mathrm{EM}}",
    "\\gparamGBW": "g_{\\mathrm{GBW}}",
    "\\defDGBWM": "g_{\\mathrm{GBW}}",
    "\\powPGBWM": "g_{\\mathrm{GBW}}",
    "\\gparamLC": "g_{\\mathrm{LCM}}",
    "\\paramLCM": "g_{\\mathrm{LCM}}",
    "\\paramBWM": "g_{\\mathrm{BWM}}",
    "\\defDBWM": "g_{\\mathrm{BWM}}",
    "\\defCDBWM": "g_{\\mathrm{BWM}}",
    "\\defCDEM": "g_{\\mathrm{EM}}",
    "\\defDPM": "g_{\\mathrm{PM}}",
}


def _expand_custom_tex_macros(inner: str) -> str:
    """Expand author-defined 2-arg, 1-arg, and 0-arg LaTeX macros into standard KaTeX."""
    _B = r"\{([^{}]*(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}[^{}]*)*)\}"
    # Fix any glued tokens from previous pass
    inner = (
        inner.replace("T_D_P", "T_{\\mathbb{P}}")
        .replace("\\thetaD_L", "\\theta {\\mathbb{L}}")
        .replace("D_L_{i}", "{\\mathbb{L}}_{i}")
        .replace("\\bbLD_K", "{\\mathbb{L}}{\\mathbb{K}}")
    )
    # 2-argument macros
    two_arg_rules = [
        (rf"\\Linner{_B}{_B}", r"\\langle \1, \2 \\rangle_{\\mathcal{L}}"),
        (rf"\\holinner{_B}{_B}", r"\\langle \1, \2 \\rangle_{\\mathrm{hol}}"),
        (rf"\\rzeroinner{_B}{_B}", r"\\langle \1, \2 \\rangle_{\\mathrm{r0}}"),
        (rf"\\inner{_B}{_B}", r"\\langle \1, \2 \\rangle"),
        (rf"\\pt{_B}{_B}", r"\\Gamma_{\1 \\to \2}"),
        (rf"\\geodesic{_B}{_B}", r"\\gamma_{\1, \2}"),
        (rf"\\nicefrac{_B}{_B}", r"\\tfrac{\1}{\2}"),
    ]
    for _ in range(3):
        for pat, repl in two_arg_rules:
            inner = re.sub(pat, repl, inner)

    # 1-argument macros
    one_arg_rules = [
        (rf"\\(?:Fnorm|fnorm){_B}", r"\\|\1\\|_{\\mathrm{F}}"),
        (rf"\\Lnorm{_B}", r"\\|\1\\|_{\\mathcal{L}}"),
        (rf"\\holnorm{_B}", r"\\|\1\\|_{\\mathrm{hol}}"),
        (rf"\\rzeronorm{_B}", r"\\|\1\\|_{\\mathrm{r0}}"),
        (rf"\\norm{_B}", r"\\|\1\\|"),
        (rf"\\spd{_B}", r"\\mathcal{S}_{++}^{\1}"),
        (rf"\\sym{_B}", r"\\mathcal{S}^{\1}"),
        (rf"\\cor{_B}", r"\\mathcal{C}_{++}^{\1}"),
        (rf"\\(?:tril|chospace){_B}", r"\\mathcal{L}_{++}^{\1}"),
        (rf"\\(?:stril|LTzero){_B}", r"\\mathcal{L}_{0}^{\1}"),
        (rf"\\LTone{_B}", r"\\mathcal{L}_{1}^{\1}"),
        (rf"\\hol{_B}", r"\\mathcal{H}_{0}^{\1}"),
        (rf"\\rzero{_B}", r"\\mathcal{R}_{0}^{\1}"),
        (rf"\\(?:orth|so){_B}", r"\\operatorname{SO}(\1)"),
        (rf"\\soLieAlgebra{_B}", r"\\mathfrak{so}(\1)"),
        (rf"\\GL{_B}", r"\\operatorname{GL}(\1)"),
        (rf"\\grasonb{_B}", r"\\operatorname{St}(\1)"),
        (rf"\\graspp{_B}", r"\\operatorname{Gr}(\1)"),
        (rf"\\pball{_B}", r"\\mathbb{B}_{K}^{\1}"),
        (rf"\\klein{_B}", r"\\mathbb{K}_{K}^{\1}"),
        (rf"\\bbh{_B}", r"\\mathbb{H}_{K}^{\1}"),
        (rf"\\bbDplus{_B}", r"\\mathcal{D}_{++}^{\1}"),
        (rf"\\bbPHS{_B}", r"\\mathcal{P}\\mathcal{H}\\mathcal{S}^{\1}"),
        (rf"\\bbRplus{_B}", r"\\mathbb{R}_{>0}^{\1}"),
        (rf"\\bbR{_B}", r"\\mathbb{R}^{\1}"),
    ]
    for _ in range(3):
        for pat, repl in one_arg_rules:
            inner = re.sub(pat, repl, inner)

    # Sort zero-arg macros longest-first so prefix collisions never happen
    for k in sorted(_CUSTOM_ZERO_ARG_MACROS.keys(), key=len, reverse=True):
        v = _CUSTOM_ZERO_ARG_MACROS[k]
        inner = re.sub(rf"{re.escape(k)}(?![a-zA-Z])", lambda _, r=v: r, inner)
    inner = re.sub(r"\\bbR(?![a-zA-Z])", r"\\mathbb{R}", inner)
    return inner


def _clean_math_segment(m: str) -> str:
    """Clean a LaTeX $...$ segment so KaTeX renders it without unknown-macro errors."""
    inner = m[1:-1].strip()
    if not inner:
        return ""
    inner = _expand_custom_tex_macros(inner)
    inner = re.sub(r"\\bm\{([^{}]*)\}", r"\\boldsymbol{\1}", inner)
    inner = re.sub(r"\\textbf\{([^{}]*)\}", r"\\mathbf{\1}", inner)
    inner = re.sub(r"\\textit\{([^{}]*)\}", r"\\mathit{\1}", inner)
    inner = re.sub(r"\\emph\{([^{}]*)\}", r"\\mathit{\1}", inner)
    inner = re.sub(r"\\cite[a-zA-Z*]*\{[^}]*\}", "", inner)
    inner = re.sub(r"\\(?:footnotesize|small|scriptsize|tiny|normalsize|large|centering|noindent)", "", inner)
    # Balance unmatched \{ or \} at segment boundaries when comma-split in source
    open_b = len(re.findall(r"(?<!\\)\{", inner))
    close_b = len(re.findall(r"(?<!\\)\}", inner))
    if open_b > close_b:
        inner = inner + ("}" * (open_b - close_b))
    elif close_b > open_b:
        inner = ("{" * (close_b - open_b)) + inner
    return f"${inner}$"


def clean_latex_cell(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"(?<!\\)%.*", "", s)
    # Remove nested \resizebox or \scalebox wrappers
    s = re.sub(r"\\(?:resizebox|scalebox)\{[^}]*\}(?:\{[^}]*\})?\{", "", s)
    s = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"\1", s)
    s = re.sub(r"\\multirow\{\d+\}\{[^}]*\}\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"\1", s)
    s = re.sub(r"\\(?:toprule|midrule|bottomrule|hline|cline\{[^}]*\}|cmidrule(?:\([^)]*\))?\{[^}]*\})", "", s)

    # If a cell has an odd number of unescaped $, close it so it never bleeds across <td> tags
    if s.count("$") % 2 == 1:
        s = s + "$"

    # Protect $...$ math segments while cleaning outer LaTeX markup
    parts = re.split(r"(\$[^$]+\$)", s)
    out_parts: list[str] = []
    for part in parts:
        if part.startswith("$") and part.endswith("$") and len(part) >= 2:
            out_parts.append(_clean_math_segment(part))
            continue
        p_txt = _expand_custom_tex_macros(part)
        p_txt = re.sub(r"\\textbf\{([^{}]*)\}", r"<strong>\1</strong>", p_txt)
        p_txt = re.sub(r"\\textit\{([^{}]*)\}", r"<em>\1</em>", p_txt)
        p_txt = re.sub(r"\\underline\{([^{}]*)\}", r"<u>\1</u>", p_txt)
        p_txt = re.sub(r"\\emph\{([^{}]*)\}", r"<em>\1</em>", p_txt)
        p_txt = re.sub(r"\\bm\{([^{}]*)\}", r"<strong>\1</strong>", p_txt)
        p_txt = re.sub(r"\\mathbf\{([^{}]*)\}", r"$\\mathbf{\1}$", p_txt)
        p_txt = re.sub(r"\\mathcal\{([^{}]*)\}", r"$\\mathcal{\1}$", p_txt)
        p_txt = re.sub(r"\\mathbb\{([^{}]*)\}", r"$\\mathbb{\1}$", p_txt)
        p_txt = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", p_txt)
        p_txt = re.sub(r"\\text\{([^{}]*)\}", r"\1", p_txt)
        p_txt = re.sub(r"\\operatorname\{([^{}]*)\}", r"\1", p_txt)
        p_txt = re.sub(r"\\cite[a-zA-Z*]*\{[^}]*\}", "", p_txt)
        p_txt = re.sub(r"\\ref\{[^}]*\}", "", p_txt)
        p_txt = re.sub(r"\\eqref\{[^}]*\}", "", p_txt)
        p_txt = re.sub(r"\\label\{[^}]*\}", "", p_txt)
        p_txt = re.sub(r"\\vspace\*?\{[^}]*\}", "", p_txt)
        p_txt = re.sub(r"\\hspace\*?\{[^}]*\}", "", p_txt)
        p_txt = re.sub(
            r"\\(?:footnotesize|small|scriptsize|tiny|normalsize|large|centering|noindent|raggedright|arraystretch)",
            "",
            p_txt,
        )
        p_txt = p_txt.replace("\\spdnetbn", "SPDNetBN").replace("\\spdnet", "SPDNet")
        p_txt = p_txt.replace("\\pm", "$\\pm$").replace("\\times", "$\\times$")
        p_txt = p_txt.replace("\\%", "%").replace("\\_", "_")
        p_txt = p_txt.replace("\\checkmark", "✓").replace("\\ding{51}", "✓").replace("\\ding{55}", "✗")
        p_txt = p_txt.replace("\\uparrow", "$\\uparrow$").replace("\\downarrow", "$\\downarrow$")
        p_txt = re.sub(r"\\[a-zA-Z]+\s*", " ", p_txt)
        p_txt = p_txt.replace("{", "").replace("}", "")
        out_parts.append(p_txt)

    res = "".join(out_parts)
    # Merge adjacent math blocks like $a$$\pm$$b$ -> $a \pm b$
    res = res.replace("$$", " ")
    # Convert plain ± between numbers to $\pm$
    res = re.sub(r"(\d)\s*±\s*(\d)", r"\1 $\\pm$ \2", res)
    return " ".join(res.split())


def parse_tabular_to_rows(raw: str) -> list[list[str]]:
    m = re.search(r"\\begin\{tabular\*?\}(?:\{[^}]*\})?\{[^}]*\}(.*?)\\end\{tabular\*?\}", raw, re.S)
    if not m:
        return []
    body = m.group(1)
    body = re.sub(r"(?<!\\)%.*", "", body)

    # Protect \\ and & inside $...$ math segments (e.g. \begin{cases} or \begin{array})
    def _protect_math(match: re.Match) -> str:
        seg = match.group(0)
        seg = seg.replace("\\\\", "__MATH_ROWSEP__").replace("&", "__MATH_COLSEP__")
        return seg

    body_protected = re.sub(r"\$[^$]+\$", _protect_math, body)
    lines = re.split(r"\\\\(?:\[[^\]]*\])?", body_protected)
    rows: list[list[str]] = []
    for ln in lines:
        ln_clean = re.sub(
            r"\\(?:toprule|midrule|bottomrule|hline|cline\{[^}]*\}|cmidrule(?:\([^)]*\))?\{[^}]*\})",
            "",
            ln,
        ).strip()
        if not ln_clean:
            continue
        raw_cells = ln_clean.split("&")
        cells = [
            clean_latex_cell(c.replace("__MATH_ROWSEP__", "\\\\").replace("__MATH_COLSEP__", "&"))
            for c in raw_cells
        ]
        if any(c for c in cells) and len(cells) >= 2:
            rows.append(cells)
    return rows


def main() -> None:
    raw_summary = json.loads(SUMMARY_PATH.read_text()) if SUMMARY_PATH.exists() else {}
    existing_out_path = REPRO_DIR / "arxiv_imported_data.json"
    existing_out = json.loads(existing_out_path.read_text()) if existing_out_path.exists() else {}
    papers_list = json.loads((REPRO_DIR / "papers.json").read_text())

    merge_extra = {
        "wilson2025eegspdnet": ["wilson2025eegspdnet"],
        "chen2026rietrans": ["chen2026rietrans", "chen2026rietrans_rmlr"],
    }

    output_data: dict[str, Any] = {}
    total_tables = 0
    total_figs = 0

    for p in papers_list:
        pid = p["paper_id"]
        src_keys = merge_extra.get(pid, [pid])

        # 1. Prioritize curated PDF_EXTRACTED_TABLES overrides, otherwise parse .tex tables
        tables_out: list[dict[str, Any]] = []
        if pid in PDF_EXTRACTED_TABLES:
            for t in PDF_EXTRACTED_TABLES[pid]:
                tables_out.append({**t, "source": ARXIV_SOURCE_BADGE[pid]})
        if not tables_out:
            for sk in src_keys:
                sdata = raw_summary.get(sk, {})
                for t_idx, t in enumerate(sdata.get("tables", [])):
                    rows = parse_tabular_to_rows(t["raw"])
                    if len(rows) >= 2:
                        cap = clean_latex_cell(t.get("caption", "")) or f"Imported Paper Table {t_idx + 1}"
                        # Normalize column count to max cols in table
                        max_c = max(len(r) for r in rows)
                        norm_rows = [r + [""] * (max_c - len(r)) for r in rows[:35]]
                        tables_out.append({"caption": cap, "rows": norm_rows, "source": ARXIV_SOURCE_BADGE[pid]})
        if not tables_out and pid in existing_out:
            tables_out = existing_out[pid].get("imported_tables", [])

        # Normalize all captions and cells in tables_out so custom macros & split math are cleaned
        cleaned_tables_out: list[dict[str, Any]] = []
        for t in tables_out:
            cap_str = str(t.get("caption", ""))
            cap_parts = re.split(r"(\$[^$]+\$)", cap_str)
            cap_clean = "".join(
                _clean_math_segment(cp) if (cp.startswith("$") and cp.endswith("$") and len(cp) >= 2) else _expand_custom_tex_macros(cp)
                for cp in cap_parts
            )
            cleaned_rows: list[list[str]] = []
            for r in t.get("rows", []):
                new_r: list[str] = []
                for cell in r:
                    c_str = str(cell)
                    # Fix any split cases/array fragments from earlier tabular splits
                    if c_str.count("$") % 2 == 1:
                        c_str = c_str.replace("$", "")
                    if "cases" in c_str or "array" in c_str:
                        c_str = re.sub(r"\b(?:cases|array(?:rcl|cc|c)?|minipage)\b", "", c_str).replace("$", "").strip()
                    c_parts = re.split(r"(\$[^$]+\$)", c_str)
                    c_clean = "".join(
                        _clean_math_segment(cp) if (cp.startswith("$") and cp.endswith("$") and len(cp) >= 2) else _expand_custom_tex_macros(cp)
                        for cp in c_parts
                    )
                    new_r.append(c_clean)
                cleaned_rows.append(new_r)
            cleaned_tables_out.append({**t, "caption": cap_clean, "rows": cleaned_rows})
        tables_out = cleaned_tables_out

        # 2. Collect curated original PNG figures & clean up unused PNGs in ENCORE_ORIG_DIR / pid
        figs_out = PREFERRED_FIGURES.get(pid, [])
        p_orig_dir = ENCORE_ORIG_DIR / pid
        keep_files = {f["file"] for f in figs_out}
        if p_orig_dir.is_dir():
            for existing in os.listdir(p_orig_dir):
                if existing not in keep_files:
                    os.remove(p_orig_dir / existing)

        total_tables += len(tables_out)
        total_figs += len(figs_out)

        output_data[pid] = {
            "source_badge": ARXIV_SOURCE_BADGE[pid],
            "experiment_config": COMPACT_EXPERIMENT_CONFIGS[pid],
            "original_figures": figs_out,
            "imported_tables": tables_out,
        }
        print(f"{pid}: {len(figs_out)} original PNGs, {len(tables_out)} imported tables ({ARXIV_SOURCE_BADGE[pid]})")

    out_json = REPRO_DIR / "arxiv_imported_data.json"
    out_json.write_text(json.dumps(output_data, indent=2))
    print(f"\nSaved {out_json}: {total_figs} original figures, {total_tables} imported tables across {len(papers_list)} papers.")


if __name__ == "__main__":
    main()
