import pytest
import torch

from spd_learn.models import Green, PhaseSPDNet


def test_green_model_initialization_and_forward():
    """Test Green model initialization and forward pass."""
    model = Green(n_outputs=2, n_chans=21)
    x = torch.randn(2, 21, 1000)  # Batch of 2, 21 channels, 1000 time points
    output = model(x)
    assert output.shape == (2, 2)


@pytest.mark.parametrize("random_f_init", [True, False])
@pytest.mark.parametrize("shrinkage_init", [None, 0.0, 2.5])
@pytest.mark.parametrize("logref", ["logeuclid", "identity"])
@pytest.mark.parametrize("dropout", [None, 0.5])
@pytest.mark.parametrize("bi_out", [None, [16], [8, 4]])
@pytest.mark.parametrize("hidden_dim", [(8,), (16, 8)])
def test_green_model_various_configs(
    random_f_init, shrinkage_init, logref, dropout, bi_out, hidden_dim
):
    """Test Green model with various configurations."""
    model = Green(
        n_outputs=2,
        n_chans=21,
        random_f_init=random_f_init,
        shrinkage_init=shrinkage_init,
        logref=logref,
        dropout=dropout,
        bi_out=bi_out,
        hidden_dim=hidden_dim,
    )
    x = torch.randn(2, 21, 1000)
    output = model(x)
    assert output.shape == (2, 2)


@pytest.mark.parametrize(
    "order",
    list(range(1, 5)),
)
@pytest.mark.parametrize(
    "lag",
    list(range(1, 5)),
)
def test_delay_order(order, lag):
    """Test if  input/output dimension combinations"""
    n_chans = 22
    n_outputs = 2

    layer = PhaseSPDNet(order=order, lag=lag, n_outputs=2, n_chans=22)
    x = torch.randn((2, n_chans, 1000))
    output = layer(x)

    # Verify output shape matches expected dimensions
    expected_shape = (2, n_outputs)
    assert (
        output.shape == expected_shape
    ), f"Failed for {order}→{lag} with input {x.shape}"
