import torch

from torch.testing import assert_close

from spd_learn.functional import dropout_spd
from spd_learn.modules import SPDDropout


def _make_spd_batch(shape, *, dtype=torch.float32):
    mat = torch.randn(*shape, dtype=dtype)
    spd = mat @ mat.mT + torch.eye(shape[-1], dtype=dtype) * 1e-3
    return spd


def test_dropout_spd_zero_prob():
    x = _make_spd_batch((2, 4, 4))
    out = dropout_spd(x, p=0.0)
    assert_close(out, x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_dropout_spd_full_prob():
    epsilon = 1e-5
    x = _make_spd_batch((3, 3, 3))
    out = dropout_spd(x, p=1.0, epsilon=epsilon, use_scaling=False)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    # off-diagonal entries should be zero
    off = out - torch.diag_embed(out.diagonal(dim1=-2, dim2=-1))
    assert_close(off, torch.zeros_like(off), atol=1e-6, rtol=1e-5)
    # diagonal entries should equal epsilon (with tolerance for numerical precision)
    diag = out.diagonal(dim1=-2, dim2=-1)
    assert_close(diag, torch.full_like(diag, epsilon), atol=2e-5, rtol=1e-4)


def test_spddropout_module_behavior():
    torch.manual_seed(0)
    x = _make_spd_batch((2, 2, 2))
    layer = SPDDropout(p=0.5, use_scaling=False, epsilon=1e-5)

    torch.manual_seed(0)
    expected = dropout_spd(x, p=0.5, use_scaling=False, epsilon=1e-5)

    # Reset seed so the module draws the same dropout mask
    torch.manual_seed(0)

    layer.train()
    out_train = layer(x)
    assert_close(out_train, expected)

    layer.eval()
    out_eval = layer(x)
    assert_close(out_eval, x)
