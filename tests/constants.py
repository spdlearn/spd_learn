"""Test constants and example SPD matrices for unit tests."""

import torch


# Example 2x2 SPD matrix with known eigendecomposition
# Eigenvalues: 5, 15
# Eigenvectors: [-1, 1], [1, 1] (unnormalized)
EXAMPLE_2X2_SPD_MATRIX = torch.tensor(
    [
        [10.0, 5.0],
        [5.0, 10.0],
    ]
)
