# Changelog

All notable changes to SPD Learn will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
| **GREEN** | Gabor Riemann EEGNet combining Gabor wavelets with Riemannian geometry | Paillard, J. et al., 2025 |
| **MAtt** | Manifold Attention mechanism for SPD matrices | Pan, Yue-Ting, et al. 2022|

#### SPD Neural Network Layers

A comprehensive set of differentiable layers that respect SPD geometry:

- **BiMap** — Bilinear mapping layer for SPD dimension reduction
- **ReEig** — Eigenvalue rectification to ensure positive definiteness
- **LogEig** — Logarithmic map to tangent space (Euclidean)
- **ExpEig** — Exponential map from tangent space back to SPD manifold
- **SqrtEig** — Matrix square root via eigendecomposition
- **InvSqrtEig** — Inverse matrix square root
- **PowerEig** — Matrix power function
- **VecMat** — Vectorization/matricization operations
- **CovLayer** — Differentiable covariance matrix estimation
- **Shrinkage** — Regularized covariance estimation (Ledoit-Wolf, Oracle)

#### Batch Normalization

SPD-specific batch normalization layers respecting Riemannian geometry:

- **SPDBatchNorm** — Standard SPD batch normalization
- **TSMBatchNorm** — Batch normalization for Tangent Space Mapping
- **AdaMomSPDBatchNorm** — Adaptive momentum batch normalization
- **DomainSPDBatchNorm** — Domain-specific batch normalization for transfer learning
- **TrackingMeanBatchNorm** — Batch normalization with running mean tracking

#### Riemannian Metrics

Four Riemannian metrics for SPD manifolds:

| Metric | Description |
|--------|-------------|
| **AffineInvariantRiemannian** | The canonical metric on SPD manifolds (AIRM) |
| **LogEuclidean** | Computationally efficient metric via matrix logarithm |
| **LogCholesky** | Metric based on Cholesky decomposition |
| **BuresWasserstein** | Optimal transport metric between Gaussians |

Each metric provides:
- Geodesic distance computation
- Exponential and logarithmic maps
- Parallel transport along geodesics
- Fréchet/Karcher mean computation

#### Functional Operations

Low-level differentiable operations in `spd_learn.functional`:

- **Matrix Operations**: `logm`, `expm`, `sqrtm`, `invsqrtm`, `powm`
- **Geodesics**: `geodesic`, `log_map`, `exp_map`
- **Statistics**: `frechet_mean`, `log_euclidean_mean`
- **Parallel Transport**: `parallel_transport`
- **Covariance**: `covariance`, `scm`, `shrinkage_covariance`

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
