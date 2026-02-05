# Contributing to SPD Learn

Thank you for your interest in contributing to SPD Learn! This guide will help you get started with contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Types of Contributions](#types-of-contributions)
- [Adding New Features](#adding-new-features)
- [Documentation](#documentation)
- [Code Organization](#code-organization)
- [Testing Guidelines](#testing-guidelines)
- [Community](#community)

---

## Getting Started

### Setting Up Development Environment

1. **Fork and clone the repository**:

   ```bash
   git clone https://github.com/spdlearn/spd_learn.git
   cd spd_learn
   ```

2. **Create a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode with all dependencies**:

   ```bash
   pip install -e ".[all]"
   ```

4. **Install pre-commit hooks** (required for consistent formatting):

   ```bash
   pip install pre-commit
   pre-commit install
   pre-commit run --all-files
   ```

---

## Development Workflow

### Creating a Branch

Create a new branch for your feature or bug fix:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### Running Tests

Run the test suite to ensure your changes don't break existing functionality:

```bash
pytest
```

Run tests with coverage:

```bash
pytest --cov=spd_learn --cov-report=html
```

### Code Style

We use `ruff` for linting and `black` for formatting. Run before committing:

```bash
ruff check spd_learn/
black spd_learn/
```

---

## Types of Contributions

### Bug Reports

If you find a bug, please open an issue on GitHub with:

1. A clear, descriptive title
2. Steps to reproduce the bug
3. Expected behavior vs. actual behavior
4. Your environment (Python version, PyTorch version, OS)
5. Minimal code example that reproduces the issue

### Feature Requests

We welcome feature requests! Please open an issue describing:

1. The problem you're trying to solve
2. Your proposed solution
3. Any alternatives you've considered

### Code Contributions

#### Pull Requests

1. **Create an issue first** for significant changes
2. **Write tests** for new functionality
3. **Update documentation** if needed
4. **Follow the code style** guidelines
5. **Keep PRs focused** - one feature/fix per PR

#### PR Checklist

Before submitting a PR, ensure:

- [ ] All tests pass (`pytest`)
- [ ] Code is formatted (`black`, `ruff`)
- [ ] Documentation is updated
- [ ] Commit messages are clear and descriptive
- [ ] `pre-commit run --all-files` passes

---

## Adding New Features

### Adding a New Layer

To add a new neural network layer:

1. Create the layer in the appropriate file under `spd_learn/modules/`
2. Add comprehensive docstrings following NumPy format
3. Export in `spd_learn/modules/__init__.py`
4. Add to `docs/source/api.rst`
5. Write unit tests in `tests/`

Example layer structure:

```python
import torch
import torch.nn as nn


class MyNewLayer(nn.Module):
    """Short description of the layer.

    Longer description explaining what the layer does,
    its mathematical formulation, and when to use it.

    Parameters
    ----------
    in_features : int
        Input dimension.
    out_features : int
        Output dimension.

    References
    ----------
    .. [1] Author, A. (Year). Paper Title. Journal.

    Examples
    --------
    >>> layer = MyNewLayer(64, 32)
    >>> x = torch.randn(16, 64, 64)
    >>> output = layer(x)
    >>> output.shape
    torch.Size([16, 32, 32])
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Initialize parameters...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, in_features, in_features)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(batch, out_features, out_features)``.
        """
        # Implementation...
        return x
```

### Adding a New Model

To add a new model architecture:

1. Create the model in `spd_learn/models/`
2. Include a docstring with:
   - Architecture description
   - Figure reference (if available)
   - All parameters documented
   - Original paper reference
3. Export in `spd_learn/models/__init__.py`
4. Add to `docs/source/api.rst`
5. Consider adding an example script

### Adding a Functional Operation

For low-level operations:

1. Add to `spd_learn/functional/`
2. Implement both forward and backward passes if using custom autograd
3. Export in `spd_learn/functional/__init__.py`
4. Document in `docs/source/api.rst`

---

## Documentation

### Building Documentation

Build the documentation locally:

```bash
cd docs
make html
```

View the built documentation:

```bash
open build/html/index.html  # macOS
xdg-open build/html/index.html  # Linux
```

### Writing Documentation

- Use **NumPy-style docstrings** for all public functions and classes
- Include **examples** in docstrings where helpful
- Add **references** to papers when implementing published methods
- Update the **User Guide** for conceptual changes

---

## Code Organization

### Package Structure

```
spd_learn/
├── __init__.py          # Package initialization
├── version.py           # Version string
├── functional/          # Low-level operations
│   ├── __init__.py
│   ├── functional.py    # Matrix operations
│   ├── covariance.py    # Covariance estimators
│   ├── manifolds.py     # Manifold operations
│   └── ...
├── modules/             # Neural network layers
│   ├── __init__.py
│   ├── bilinear.py      # BiMap layers
│   ├── modeig.py        # LogEig, ReEig, ExpEig
│   ├── batchnorm.py     # SPD batch normalization
│   └── ...
└── models/              # Pre-built architectures
    ├── __init__.py
    ├── spdnet.py
    ├── tensorcsp.py
    └── ...
```

### Naming Conventions

- **Modules**: `CamelCase` (e.g., `BiMap`, `SPDBatchNormMeanVar`)
- **Functions**: `snake_case` (e.g., `matrix_log`, `sample_covariance`)
- **Private methods**: prefix with `_` (e.g., `_compute_mean`)

---

## Testing Guidelines

### Test Structure

Tests are organized in `tests/` mirroring the package structure:

```
tests/
├── test_functional.py
├── test_modules.py
└── test_models.py
```

### Writing Tests

- Test both **forward and backward passes**
- Test **edge cases** (empty batches, single samples)
- Test **numerical stability** with extreme values
- Use **parametrized tests** for multiple configurations

Example test:

```python
import pytest
import torch
from spd_learn.modules import BiMap


@pytest.mark.parametrize(
    "in_features,out_features",
    [
        (64, 32),
        (32, 16),
        (16, 8),
    ],
)
def test_bimap_output_shape(in_features, out_features):
    layer = BiMap(in_features, out_features)
    x = torch.randn(8, in_features, in_features)
    x = x @ x.transpose(-1, -2)  # Make SPD
    y = layer(x)
    assert y.shape == (8, out_features, out_features)


def test_bimap_preserves_spd():
    layer = BiMap(32, 16)
    x = torch.randn(8, 32, 32)
    x = x @ x.transpose(-1, -2) + torch.eye(32)  # SPD
    y = layer(x)
    # Check positive definiteness via eigenvalues
    eigvals = torch.linalg.eigvalsh(y)
    assert (eigvals > 0).all()
```

---

## Community

- **GitHub Issues** — Bug reports and feature requests
- **Pull Requests** — Code contributions
- **Discussions** — Questions and ideas

Thank you for contributing to SPD Learn!
