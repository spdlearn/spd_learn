# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
"""Synthetic CPU comparison of static and periodically recentered charts.

Run from the checkout with PYTHONPATH=. and OMP_NUM_THREADS=1. Prints JSON
metadata, per-seed training rows and microbenchmark medians/IQR to stdout.
This measures optimization and compute cost, not EEG generalization.
"""

import json
import platform
import time

from copy import deepcopy

import torch

from torch import nn
from torch.nn import functional as F
from torch.utils.benchmark import Timer

from spd_learn.modules import BiMap, SPDBatchNormMeanVar, step_dynamic_trivialization


def run_training(
    seed: int,
    dtype: torch.dtype,
    interval: int | None,
    *,
    steps: int = 200,
    n: int = 22,
    k: int = 8,
    enabled: bool = True,
) -> dict:
    """Fit a fixed teacher; time updates including recentering, excluding diagnostics."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 1000)
    model = nn.Sequential(
        BiMap(n, k, dtype=dtype), SPDBatchNormMeanVar(k, dtype=dtype)
    ).eval()
    teacher = deepcopy(model)
    teacher[0].reset_parameters()
    with torch.no_grad():
        teacher[1].bias = torch.diag(torch.linspace(0.5, 2, k, dtype=dtype)).unsqueeze(
            0
        )
    a = torch.randn(32, n, n, dtype=dtype, generator=generator)
    x = a @ a.mT / n + 0.1 * torch.eye(n, dtype=dtype)
    with torch.no_grad():
        target = teacher(x)
    if enabled:
        step_dynamic_trivialization(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0, weight_decay=0)
    with torch.no_grad():
        losses = {0: F.mse_loss(model(x), target).detach()}
    elapsed = 0.0
    for step in range(1, steps + 1):
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x), target)
        loss.backward()
        if step == steps:
            gradients = [p.grad for p in model.parameters()]
        optimizer.step()
        if enabled and interval is not None and step % interval == 0:
            step_dynamic_trivialization(model, optimizer)
        elapsed += time.perf_counter() - start
        if step in (50, 100, 200, steps):
            with torch.no_grad():
                losses[step] = F.mse_loss(model(x), target).detach()

    with torch.no_grad():
        output = model(x)
        weight = model[0].weight
        orthogonality = (
            (weight.mT @ weight - torch.eye(k, dtype=dtype)).abs().max().item()
        )
        minimum = torch.linalg.eigvalsh(output).min().item()
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert torch.isfinite(output).all() and minimum > 0
        tolerance = 1e-5 if dtype == torch.float32 else 1e-10
        assert orthogonality < tolerance
        continuity = None
        if enabled:
            step_dynamic_trivialization(model, optimizer)
            updated = model(x)
            torch.testing.assert_close(updated, output, atol=tolerance, rtol=tolerance)
            continuity = (updated - output).abs().max().item()
    return {
        "seed": seed,
        "dtype": str(dtype),
        "enabled": enabled,
        "interval": interval,
        "losses": {step: loss.item() for step, loss in losses.items()},
        "elapsed_seconds": elapsed,
        "orthogonality_error": orthogonality,
        "min_eigenvalue": minimum,
        "continuity_error": continuity,
    }


def main() -> None:
    torch.set_num_threads(1)
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "platform": platform.platform(),
                "device": "cpu",
                "threads": 1,
                "protocol": "200 SGD steps, lr=0.05, momentum=0, decay=0, batch=32",
            }
        ),
        flush=True,
    )
    for dtype in (torch.float32, torch.float64):
        for seed in (0, 1, 2):
            for enabled, interval in (
                (False, None),
                (True, None),
                (True, 1),
                (True, 50),
                (True, 100),
            ):
                print(
                    json.dumps(
                        {
                            "training": run_training(
                                seed, dtype, interval, enabled=enabled
                            )
                        }
                    ),
                    flush=True,
                )

        for kind, n, k in (
            ("bimap", 22, 8),
            ("bimap", 64, 32),
            ("spd_bias", 8, 8),
            ("spd_bias", 32, 32),
        ):
            for enabled in (False, True):
                torch.manual_seed(42)
                layer = (
                    BiMap(n, k, dtype=dtype)
                    if kind == "bimap"
                    else SPDBatchNormMeanVar(n, dtype=dtype)
                )
                a = torch.randn(32, n, n, dtype=dtype)
                x = a @ a.mT / n + 0.1 * torch.eye(n, dtype=dtype)
                if kind == "spd_bias":
                    with torch.no_grad():
                        layer.bias = torch.diag(
                            torch.linspace(0.5, 2, n, dtype=dtype)
                        ).unsqueeze(0)
                if enabled:
                    step_dynamic_trivialization(layer)

                @torch.no_grad()
                def forward():
                    return layer(x) if kind == "bimap" else layer.bias

                def backward():
                    layer.zero_grad(set_to_none=True)
                    value = layer(x) if kind == "bimap" else layer.bias
                    value.square().mean().backward()

                def recenter():
                    step_dynamic_trivialization(layer)

                operations = [("forward", forward), ("forward_backward", backward)]
                if enabled:
                    operations.append(("recenter", recenter))
                for operation, fn in operations:
                    for _ in range(10):
                        fn()
                    result = Timer(
                        "fn()", globals={"fn": fn}, num_threads=1
                    ).blocked_autorange(min_run_time=1.0)
                    print(
                        json.dumps(
                            {
                                "micro": {
                                    "kind": kind,
                                    "n": n,
                                    "k": k,
                                    "dtype": str(dtype),
                                    "enabled": enabled,
                                    "operation": operation,
                                    "median_ms": result.median * 1000,
                                    "iqr_ms": result.iqr * 1000,
                                }
                            }
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
