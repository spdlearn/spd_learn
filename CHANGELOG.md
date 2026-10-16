# Changelog

All notable changes to SPD Learn will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Reset BiMap weights through the orthogonal parametrization instead of modifying a derived temporary tensor.
- Initialize each depthwise BiMap matrix independently when using orthogonal initialization.
- Stop applying a trainable bias in `SPDBatchNormMean` when `rebias=False`.
- Reset parametrized biases and scales correctly in `SPDBatchNormMean`,
  `SPDBatchNormMeanVar`, and `SPDBatchNormLie`.

## [0.2.1] - 2026-10-11

### Fixed

- Corrected the attribution of the GREEN model and its wavelet layer: both are
  re-implementations of GREEN (Paillard, Hipp and Engemann, 2025;
  https://github.com/Roche/neuro-green) and are distributed under BSD-3-Clause
  like the rest of the package.
- The integration test job now runs on pushes to `main` (it was gated on a
  `master` branch that does not exist).

### Tests and continuous integration

- New tests for the logging utilities, the initialization functions, the
  manifold parametrizations, the Riemannian PGD projection and the
  Log-Euclidean exponential and logarithmic maps, which had no or partial
  coverage.
- New CI job that installs the lowest dependency versions allowed by
  `pyproject.toml` on Python 3.11 and runs the test suite.

### Packaging

- Declare the license as `BSD-3-Clause` in the package metadata (PEP 639),
  ship `LICENSE.txt` as a license file, and add the project URLs.
- State the minimum versions of the runtime dependencies: `torch>=2.3`,
  `einops>=0.6`, `numpy>=2.0`. pyRiemann 0.12 declares `numpy>=1.25` but uses
  `ndarray.mT`, which requires NumPy 2.0, and PyTorch supports NumPy 2 from
  version 2.3.
- Add an SPDX license header to every source file, including tests, examples
  and documentation scripts.
- Add `CITATION.cff` and update the citation in the README and the
  documentation (title, full author list, software version).
- Remove a duplicated logo image from the documentation sources.
- Keep version-control files, unfinished examples and website verification
  files out of source archives (`.gitattributes` `export-ignore`).

## [0.2.0] - 2026-09-27

### Fixed

- EEGSPDNet now passes `spd_drop_prob` to each SPD dropout layer instead of
  silently using the layer's default probability of 0.5.
- Green uses the stable spectral matrix-log derivative, avoiding NaN gradients
  at repeated eigenvalues.
- BiMap defaults to the Cayley parametrization, preserving orthogonality during
  AdamW updates with weight decay. Explicit Householder users should exclude
  the parametrization's original parameter from weight decay.
- Corrected `matrix_power` exponent gradients, including clamped eigenvalues
  and Python scalar exponents.
- Corrected Bures-Wasserstein geodesics, parallel transport, and ExpEig input
  decoding and shape validation.
- Fixed device mismatches in wavelet and covariance operations and their tests.

### Performance

- WaveletConv evaluates real inputs using a single real convolution over the
  real and imaginary kernels. Complex inputs retain the complex convolution
  path. Regression tests cover outputs and input, frequency, and bandwidth
  gradients for both precisions, input layouts, padding, stride, and CPU autocast
  settings. The real convolution preserves complex-kernel precision under autocast.
- WaveletConv also accepts non-contiguous inputs without requiring callers to
  make a contiguous copy.

### Added

- `SPDBatchNormLie` and functional Lie-group batch-normalization operations.
- Source-free SPD domain-adaptation and batch-normalization examples.
- Regression coverage for EEGSPDNet dropout, Green's matrix-log derivative,
  and BiMap orthogonality after optimizer updates.

### Dependencies

- Require the released `pyriemann>=0.12` package for Array API support.
- Require `moabb>=1.7.2` in the `brain` extra for pyRiemann mean compatibility.
- Cap the test dependency at `pytest<9.1` for pytest-cases compatibility.

## [0.1.0] - 2026-02-05

### 🎉 Initial Release

We are excited to announce the first public release of **SPD Learn** — a pure PyTorch library for geometric deep learning on Symmetric Positive Definite (SPD) matrices.

SPD Learn provides differentiable Riemannian operations, broadcast-compatible layers, and reference implementations of published neural network architectures for SPD data, with a focus on EEG/BCI applications.

---

### ✨ Features

#### Neural Network Models

Seven state-of-the-art deep learning architectures for SPD matrix data:

| Model | Description | Reference |
|-------|-------------|-----------|
| **SPDNet** | Foundational architecture for deep learning on SPD manifolds with dimension reduction | Huang & Van Gool, AAAI 2017 |
| **EEGSPDNet** | Specialized for EEG classification, combining covariance estimation with SPD layers | Wilson, D. et al., 2024 |
| **TSMNet** | Tangent Space Mapping Network with convolutional features and SPD batch normalization | Kobler et al., 2022 |
| **TensorCSPNet** | Multi-band EEG feature extraction using Tensor Common Spatial Patterns | Ju, C. et al. 2022 |
| **PhaseSPDNet** | Phase-space embedding combined with SPDNet for geometric EEG dynamics analysis | Carrara et al., 2024 |
| **Green** | Gabor Riemann EEGNet combining Gabor wavelets with Riemannian geometry | Paillard, J. et al., 2025 |
| **MAtt** | Manifold Attention mechanism for SPD matrices | Pan, Yue-Ting, et al. 2022|

#### SPD Neural Network Layers

A comprehensive set of differentiable layers that respect SPD geometry:

- **BiMap** — Bilinear mapping layer for SPD dimension reduction
- **BiMapIncreaseDim** — Bilinear mapping layer for SPD dimension increase
- **ReEig** — Eigenvalue rectification to ensure positive definiteness
- **LogEig** — Logarithmic map to tangent space (Euclidean)
- **ExpEig** — Exponential map from tangent space back to SPD manifold
- **CovLayer** — Differentiable covariance matrix estimation
- **Shrinkage** — Regularized covariance estimation (Ledoit-Wolf, Oracle)
- **TraceNorm** — Trace normalization for SPD matrices
- **SPDDropout** — Dropout layer for SPD matrices
- **LogEuclideanResidual** — Residual connection in Log-Euclidean space
- **WaveletConv** — Gabor wavelet convolution layer
- **PatchEmbeddingLayer** — Patch embedding for SPD matrices
- **Vec** — Full matrix vectorization
- **Vech** — Half-vectorization (upper triangular)

#### Manifold Parametrizations

PyTorch parametrizations for constraining weight matrices:

- **SymmetricPositiveDefinite** — Constrains a parameter to be SPD
- **PositiveDefiniteScalar** — Constrains a scalar parameter to be positive

#### Batch Normalization

SPD-specific batch normalization layers respecting Riemannian geometry:

- **SPDBatchNormMean** — SPD batch normalization using Riemannian mean centering
- **SPDBatchNormMeanVar** — SPD batch normalization with both mean centering and tangent space variance normalization
- **BatchReNorm** — SPD batch renormalization with relaxed constraints for small batches

#### Riemannian Metrics

Four Riemannian metrics for SPD manifolds:

| Metric | Description |
|--------|-------------|
| **AffineInvariantRiemannian** | The canonical metric on SPD manifolds (AIRM) |
| **LogEuclidean** | Computationally efficient metric via matrix logarithm |
| **LogCholesky** | Metric based on Cholesky decomposition |
| **BuresWasserstein** | Optimal transport metric between Gaussians |

Each metric provides geodesic distance computation and Fréchet mean computation. Additionally:
- **AIRM**: Exponential/logarithmic maps, parallel transport, Karcher mean iteration
- **Log-Euclidean**: Geodesics, Lie group operations (`log_euclidean_multiply`, `log_euclidean_scalar_multiply`)
- **Log-Cholesky**: Cholesky log/exp maps, parallel transport
- **Bures-Wasserstein**: Optimal transport map for domain adaptation (not Riemannian parallel transport)

Parallel transport is available for AIRM, Log-Euclidean, and Log-Cholesky metrics, plus metric-agnostic numerical methods (Schild's ladder, pole ladder).

#### Functional Operations

Low-level differentiable operations in `spd_learn.functional`:

- **Matrix Operations**: `matrix_log`, `matrix_exp`, `matrix_sqrt`, `matrix_inv_sqrt`, `matrix_power`
- **Fréchet Derivatives**: `frechet_derivative_log`, `frechet_derivative_exp`
- **Metric-specific Geodesics**: `airm_geodesic`, `log_euclidean_geodesic`
- **Metric-specific Maps**: `exp_map_airm`, `log_map_airm`
- **Statistics**: `log_euclidean_mean`, `bures_wasserstein_mean`, `karcher_mean_iteration`
- **Parallel Transport**: `parallel_transport_airm`, `parallel_transport_lem`, `parallel_transport_log_cholesky`, `schild_ladder`, `pole_ladder`, `transport_tangent_vector`
- **Covariance**: `sample_covariance`, `ledoit_wolf`
- **Bilinear**: `bimap_transform`, `bimap_increase_dim`
- **Regularization**: `trace_normalization`, `dropout_spd`
- **Batch Normalization Utilities**: `spd_centering`, `spd_rebiasing`, `tangent_space_variance`
- **Adversarial**: `spd_rpgd_attack`
- **Wavelets**: `compute_gabor_wavelet`
- **Numerical Stability**: `NumericalConfig`, `NumericalContext`

#### Additional Features

- **GPU Acceleration** — Full CUDA support with efficient batched operations
- **Automatic Differentiation** — Seamless gradient computation on manifolds via PyTorch
- **scikit-learn Compatible** — Integration with ML pipelines via skorch/Braindecode wrappers
- **Comprehensive Documentation** — Tutorials, API reference, and theoretical background
- **Examples Gallery** — Ready-to-run examples for common use cases

---

### 📚 Documentation

- **Installation Guide** — Step-by-step setup instructions
- **User Guide** — Comprehensive introduction to SPD matrices and Riemannian geometry
- **Theory Section** — Mathematical background, layer descriptions, and metric details
- **API Reference** — Complete documentation of all modules and functions
- **Examples Gallery** — Practical examples including EEG classification

---

### 🔧 Technical Details

- **Python**: 3.11, 3.12, 3.13
- **PyTorch**: 2.0+
- **License**: BSD-3-Clause

---

### 🙏 Acknowledgments

SPD Learn is developed and maintained by researchers from:

- Inria (French National Institute for Research in Digital Science and Technology)
- CNRS (French National Centre for Scientific Research)
- CEA (French Alternative Energies and Atomic Energy Commission)
- Université Paris-Saclay
- ATR (Advanced Telecommunications Research Institute International)
- Université Savoie Mont Blanc

---

### 📖 Citation

If you use SPD Learn in your research, please cite:

```bibtex
@article{aristimunha2025spdlearn,
  title     = {SPDlearn: A Geometric Deep Learning Python Library for Neural
               Decoding Through Trivialization},
  author    = {Aristimunha, Bruno and Ju, Ce and Collas, Antoine and
               Bouchard, Florent and Thirion, Bertrand and
               Chevallier, Sylvain and Kobler, Reinmar},
  journal   = {To be submitted},
  year      = {2026},
  url       = {https://github.com/spdlearn/spd_learn}
}
```

---

**Full Changelog**: https://github.com/spdlearn/spd_learn/commits/v0.1.0
