import inspect

import pytest
import torch

import spd_learn

from spd_learn.models import __all__ as model_list
from spd_learn.modules import CovLayer
from spd_learn.modules import __all__ as module_list


# parameters for wavelet initialization
n_freqs_init = 10
oct_min = 0.5
oct_max = 3.0

mandatory_parameters_per_module = {
    "BiMap": dict(in_features=10, out_features=5),
    "BiMapDepthWise": dict(in_features=10, out_features=5, depthwise=2),
    "SPDBatchNormMean": dict(num_features=10),
    "BatchReNorm": dict(num_features=10),
    "SPDBatchNormMeanVar": dict(num_features=10),
    "PatchEmbeddingLayer": dict(n_chans=10, n_patches=2),
    "BiMapIncreaseDim": dict(in_features=10, out_features=20),
    "Shrinkage": dict(n_chans=10),
    "WaveletConv": dict(
        kernel_width_s=0.5,
        foi_init=torch.rand(n_freqs_init) * (oct_max - oct_min) + oct_min,
    ),
}


@pytest.mark.parametrize("model_name", model_list)
def test_integration(model_name):
    model_class = getattr(spd_learn.models, model_name)

    params = {}
    if model_name == "TensorCSPNet":
        # TensorCSPNet requires a different input shape
        x = torch.randn(2, 9, 22, 1000)
    elif model_name == "Green":
        params = {"sfreq": 125}
        x = torch.randn(2, 22, 1000)
    else:
        x = torch.randn(2, 22, 1000)

    model = model_class(n_chans=22, n_outputs=2, **params)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 2)


@pytest.mark.parametrize("model_name", model_list)
def test_integration_braindecode(model_name):
    try:
        # Soft dependency
        from braindecode import EEGClassifier
    except ImportError:
        pytest.skip("braindecode is not installed")

    model = getattr(spd_learn.models, model_name)

    clf = EEGClassifier(
        module=model,
        module__n_chans=22,
        module__n_outputs=2,
    )
    # Inplace initialization
    clf.initialize()
    # check that the model is correctly initialized
    print(clf)


@pytest.mark.parametrize("module_name", module_list)
def test_module_expose_device_dtype(module_name):
    # In this test we check if all the layers
    # have the parameters device and dtype exposed
    # to the user
    module = getattr(spd_learn.modules, module_name)

    params = inspect.signature(module).parameters
    assert "device" in params
    assert "dtype" in params
    mandatory_param = mandatory_parameters_per_module.get(module_name, {})
    layer = module(device="cpu", dtype=torch.float32, **mandatory_param)
    assert layer is not None


# Test that all parameters of the module are on the expected device.
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
        # pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available (MAC only)"))
    ],
)
@pytest.mark.parametrize("module_name", module_list)
def test_module_parameters_on_device(module_name, device):
    """Instantiate the module on the given device and verify that each parameter is located on that device."""
    if module_name == "BiMap" and device == "mps":
        pytest.skip("Bimap does not support float16 for MPS.")

    module_class = getattr(spd_learn.modules, module_name)
    dtype = torch.float32
    mandatory_param = mandatory_parameters_per_module.get(module_name, {})
    module = module_class(device=device, dtype=dtype, **mandatory_param)
    for name, param in module.named_parameters():
        assert param.device.type == device, (
            f"Parameter '{name}' is on {param.device} but expected {device}"
        )


# Optionally, test that all submodules’ parameters are on the expected device.
@pytest.mark.parametrize(
    "device", ["cpu"]
)  # if you want to test submodules only on CPU in CI, or parameterize as above
@pytest.mark.parametrize("module_name", module_list)
def test_module_submodules_on_device(module_name, device):
    """Verify that for each submodule in the module, its parameters are on the correct device."""
    module_class = getattr(spd_learn.modules, module_name)
    dtype = torch.float32
    mandatory_param = mandatory_parameters_per_module.get(module_name, {})

    module = module_class(device=device, dtype=dtype, **mandatory_param)
    for submodule in module.modules():
        for name, param in submodule.named_parameters(recurse=False):
            assert param.device.type == device, (
                f"Submodule parameter '{name}' in {submodule} is on {param.device} but expected {device}"
            )


# Optionally, test that all buffers are on the expected device.
@pytest.mark.parametrize(
    "device", ["cpu"]
)  # if you want to test buffers only on CPU in CI, or parameterize as above
@pytest.mark.parametrize("module_name", module_list)
def test_module_buffers_on_device(module_name, device):
    """Verify that all buffers in the module are on the correct device."""
    module_class = getattr(spd_learn.modules, module_name)
    dtype = torch.float32
    mandatory_param = mandatory_parameters_per_module.get(module_name, {})

    module = module_class(device=device, dtype=dtype, **mandatory_param)
    for name, buffer in module.named_buffers():
        assert buffer.device.type == device, (
            f"Buffer '{name}' is on {buffer.device} but expected {device}"
        )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64, torch.complex64, torch.complex128],
    ids=["float32", "float64", "complex64", "complex128"],
)
@pytest.mark.parametrize("module_name", module_list)
def test_module_dtype(module_name, dtype, device):
    """Verify that the module is created with the correct dtype."""
    if module_name == "PositiveDefiniteScalar":
        # PositiveDefiniteScalar is a parametrization for scalars, not matrices
        pytest.skip(
            "PositiveDefiniteScalar is a scalar parametrization, not a matrix layer."
        )

    module_class = getattr(spd_learn.modules, module_name)
    mandatory_param = mandatory_parameters_per_module.get(module_name, {})

    module = module_class(device=device, dtype=dtype, **mandatory_param)

    if module_name == "CovLayer":
        x = torch.randn(2, 10, 1000, dtype=dtype)
        with torch.no_grad():
            out = module(x)
    elif module_name == "LogEuclideanResidual":
        # LogEuclideanResidual takes two SPD inputs
        x = torch.randn(2, 10, 1000, dtype=dtype)
        x = CovLayer(device=device, dtype=dtype)(x)
        y = torch.randn(2, 10, 1000, dtype=dtype)
        y = CovLayer(device=device, dtype=dtype)(y)
        with torch.no_grad():
            out = module(x, y)
        assert out.dtype == dtype
        return
    elif module_name == "WaveletConv":
        # WaveletConv processes raw time series, not covariance matrices
        # It uses complex wavelets internally but outputs the dtype it was initialized with
        x = torch.randn(2, 10, 1000, dtype=dtype)
        with torch.no_grad():
            out = module(x)
        assert out.dtype == dtype
        return
    else:
        x = torch.randn(2, 10, 1000, dtype=dtype)
        x = CovLayer(device=device, dtype=dtype)(x)

        # checking if torch.linalg.eigh is available
        if dtype == torch.float16:
            with pytest.raises(RuntimeError):
                with torch.no_grad():
                    out = module(x)

        with torch.no_grad():
            out = module(x)

    assert out.dtype == dtype


# Batch shapes to test broadcast compatibility
@pytest.mark.parametrize(
    "extra_dim",
    [
        (2, 3),
        (2, 3, 4),
        (2, 3, 4, 5),
    ],
)
@pytest.mark.parametrize("module_name", module_list)
def test_module_broadcast_compatible(module_name, extra_dim):
    """Verify that each module only applies operations on the last two dimensions,
    preserving arbitrary leading (batch) dimensions.
    """
    if module_name in [
        "PatchEmbeddingLayer",
        "BiMapDepthWise",
        "WaveletConv",
        "LogEuclideanResidual",
        "PositiveDefiniteScalar",
    ]:
        # Skip these modules as they are not broadcast compatible or have special interfaces
        # WaveletConv only supports 3D and 4D inputs (time series data)
        # LogEuclideanResidual takes two inputs
        # PositiveDefiniteScalar is a parametrization for scalars, not matrices
        pytest.skip(f"{module_name} is not broadcast compatible.")
    # Use default device and dtype
    device = "cpu"
    dtype = torch.float32

    # Get module class and its mandatory parameters
    module_class = getattr(spd_learn.modules, module_name)
    mandatory_param = mandatory_parameters_per_module.get(module_name, {})

    module = module_class(device=device, dtype=dtype, **mandatory_param)

    # Create input tensor with specified batch shape
    x = torch.randn(2, *extra_dim, 10, 1000, dtype=dtype)

    # For non-CovLayer modules, apply CovLayer first
    if module_name != "CovLayer":
        x = CovLayer(device=device, dtype=dtype)(x)

    # Compute output
    with torch.no_grad():
        _ = module(x)
