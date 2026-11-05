# Dynamic trivialization benchmark — 2026-11-05

On this synthetic task, recentering did not meaningfully improve final loss. K=50 added about 1–2% update time relative to the enabled K=∞ control; K=1 added about 22%. These are measurements on one host, not a general performance guarantee.

## Reproduce

Apple M4 Max, macOS 27.0.1 arm64, Python 3.12.12, PyTorch 2.6.0; CPU, one PyTorch thread. Source revision: `b5fbf689bb2b11ad5f3ea053775ed1f468718697`. No datasets or new dependencies. Later review fixes add pre-mutation assignment compatibility checks and typing casts; the recorded timings refer to the revision above, with the same chart formulas.

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python examples/applied_examples/benchmark_dynamic_trivialization.py
```

The [script](../../../examples/applied_examples/benchmark_dynamic_trivialization.py) emits all raw rows as JSON lines. The final run was separate from test and mypy execution. Timings vary with system load.

## Protocol

Teacher/student BiMap(22,8) → SPDBatchNormMeanVar(8), eval mode with autograd active; fixed batch of 32 SPD inputs and fixed teacher outputs. Teacher projection differs and its bias has eigenvalues 0.5 through 2. Each paired mode starts from the same seeded student effective weights; a dedicated data generator prevents rectangular QR randomness from changing inputs. Seeds 0/1/2; float32/64; 200 SGD updates, lr=0.05, momentum=0, weight_decay=0. Loss is mean squared error over output elements.

Time is the sum of update durations, including backward, optimizer and scheduled recentering. Setup, diagnostic forwards and loss logging are excluded. Enabled K=∞ calls the helper once at setup; existing static never calls it. Enabling has its own congruence cost, distinct from periodic update overhead. All modes keep the same learning rate.

## Aggregate training results

Time is median ± IQR across three seeds (inclusive quartiles); loss is the mean final loss. The IQR is not a confidence interval.

| Dtype | Mode | Loss at 200 | Update time (ms) |
| --- | --- | ---: | ---: |
| float32 | existing static | 0.12002616 | 226.72 ± 19.30 |
| float32 | enabled, K=∞ | 0.12002613 | 241.69 ± 2.96 |
| float32 | K=1 | 0.12000275 | 295.62 ± 5.29 |
| float32 | K=50 | 0.12000422 | 244.29 ± 4.83 |
| float32 | K=100 | 0.12000865 | 245.55 ± 5.58 |
| float64 | existing static | 0.11948874 | 262.00 ± 2.59 |
| float64 | enabled, K=∞ | 0.11948874 | 272.28 ± 3.84 |
| float64 | K=1 | 0.11947368 | 332.78 ± 14.49 |
| float64 | K=50 | 0.11947462 | 276.01 ± 1.98 |
| float64 | K=100 | 0.11947746 | 276.12 ± 2.08 |

## Per-seed training results

| Dtype | Seed | Mode | Loss 0 | Loss 50 | Loss 100 | Loss 200 | Time (ms) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| float32 | 0 | existing static | 0.17221111 | 0.15281430 | 0.13956258 | 0.11874551 | 263.73 |
| float32 | 0 | enabled, K=∞ | 0.17221110 | 0.15281433 | 0.13956252 | 0.11874543 | 241.69 |
| float32 | 0 | K=1 | 0.17221110 | 0.15281309 | 0.13955499 | 0.11870275 | 295.62 |
| float32 | 0 | K=50 | 0.17221110 | 0.15281437 | 0.13955697 | 0.11870569 | 244.75 |
| float32 | 0 | K=100 | 0.17221110 | 0.15281433 | 0.13956247 | 0.11871376 | 245.55 |
| float32 | 1 | existing static | 0.16815814 | 0.14609268 | 0.13358785 | 0.11581640 | 225.12 |
| float32 | 1 | enabled, K=∞ | 0.16815813 | 0.14609276 | 0.13358778 | 0.11581643 | 243.90 |
| float32 | 1 | K=1 | 0.16815813 | 0.14609224 | 0.13358563 | 0.11580549 | 298.10 |
| float32 | 1 | K=50 | 0.16815813 | 0.14609271 | 0.13358620 | 0.11580583 | 244.29 |
| float32 | 1 | K=100 | 0.16815813 | 0.14609276 | 0.13358781 | 0.11580800 | 245.85 |
| float32 | 2 | existing static | 0.18237221 | 0.15854505 | 0.14501818 | 0.12551656 | 226.72 |
| float32 | 2 | enabled, K=∞ | 0.18237221 | 0.15854502 | 0.14501822 | 0.12551653 | 237.97 |
| float32 | 2 | K=1 | 0.18237221 | 0.15854442 | 0.14501536 | 0.12550002 | 287.52 |
| float32 | 2 | K=50 | 0.18237221 | 0.15854497 | 0.14501607 | 0.12550113 | 235.10 |
| float32 | 2 | K=100 | 0.18237221 | 0.15854502 | 0.14501825 | 0.12550420 | 234.69 |
| float64 | 0 | existing static | 0.17966716 | 0.15610515 | 0.14249262 | 0.12238655 | 265.23 |
| float64 | 0 | enabled, K=∞ | 0.17966716 | 0.15610515 | 0.14249262 | 0.12238655 | 279.61 |
| float64 | 0 | K=1 | 0.17966716 | 0.15610454 | 0.14248913 | 0.12236772 | 359.54 |
| float64 | 0 | K=50 | 0.17966716 | 0.15610515 | 0.14248999 | 0.12236891 | 279.60 |
| float64 | 0 | K=100 | 0.17966716 | 0.15610515 | 0.14249262 | 0.12237247 | 276.12 |
| float64 | 1 | existing static | 0.16797382 | 0.14761877 | 0.13512498 | 0.11678109 | 262.00 |
| float64 | 1 | enabled, K=∞ | 0.16797382 | 0.14761877 | 0.13512498 | 0.11678109 | 272.28 |
| float64 | 1 | K=1 | 0.16797382 | 0.14761821 | 0.13512184 | 0.11676541 | 332.78 |
| float64 | 1 | K=50 | 0.16797382 | 0.14761877 | 0.13512262 | 0.11676638 | 276.01 |
| float64 | 1 | K=100 | 0.16797382 | 0.14761877 | 0.13512498 | 0.11676933 | 276.14 |
| float64 | 2 | existing static | 0.17436679 | 0.15086525 | 0.13775355 | 0.11929857 | 260.04 |
| float64 | 2 | enabled, K=∞ | 0.17436679 | 0.15086525 | 0.13775355 | 0.11929857 | 271.92 |
| float64 | 2 | K=1 | 0.17436679 | 0.15086480 | 0.13775123 | 0.11928791 | 330.57 |
| float64 | 2 | K=50 | 0.17436679 | 0.15086525 | 0.13775180 | 0.11928857 | 275.64 |
| float64 | 2 | K=100 | 0.17436679 | 0.15086525 | 0.13775355 | 0.11929058 | 271.99 |

## Layer microbenchmarks

Ten warmups then `Timer.blocked_autorange(min_run_time=1.0)`. Median ± within-run IQR, milliseconds. BiMap uses batch 32 and depthwise 1; SPD-bias rows time only the bias parametrization, not the entire normalizer. SPD bias eigenvalues are 0.5 through 2 in both modes. Repeated recenter timings are at zero tangent coordinates; the training timings above include actual updates away from the chart center. Dashes mean no recenter operation was scheduled.

| Dtype | Target | Enabled | Forward, no grad | Forward/backward | Recenter |
| --- | --- | --- | ---: | ---: | ---: |
| float32 | BiMap 22→8 | False | 0.1091 ± 0.0017 | 0.1897 ± 0.0032 | — |
| float32 | BiMap 22→8 | True | 0.1105 ± 0.0020 | 0.1894 ± 0.0017 | 0.0953 ± 0.0009 |
| float32 | BiMap 64→32 | False | 0.1585 ± 0.0025 | 0.4536 ± 0.0172 | — |
| float32 | BiMap 64→32 | True | 0.1594 ± 0.0158 | 0.4018 ± 0.0054 | 0.1678 ± 0.0016 |
| float32 | SPD bias 8 | False | 0.0271 ± 0.0003 | 0.1229 ± 0.0020 | — |
| float32 | SPD bias 8 | True | 0.0395 ± 0.0003 | 0.1608 ± 0.0058 | 0.1061 ± 0.0012 |
| float32 | SPD bias 32 | False | 0.0352 ± 0.0004 | 0.1424 ± 0.0041 | — |
| float32 | SPD bias 32 | True | 0.0485 ± 0.0009 | 0.1947 ± 0.0037 | 0.1200 ± 0.0014 |
| float64 | BiMap 22→8 | False | 0.1196 ± 0.0018 | 0.2018 ± 0.0164 | — |
| float64 | BiMap 22→8 | True | 0.1202 ± 0.0033 | 0.2018 ± 0.0141 | 0.1046 ± 0.0052 |
| float64 | BiMap 64→32 | False | 0.2359 ± 0.0158 | 0.5859 ± 0.0468 | — |
| float64 | BiMap 64→32 | True | 0.2413 ± 0.0154 | 0.6051 ± 0.0598 | 0.2332 ± 0.0077 |
| float64 | SPD bias 8 | False | 0.0321 ± 0.0005 | 0.1325 ± 0.0024 | — |
| float64 | SPD bias 8 | True | 0.0449 ± 0.0010 | 0.1774 ± 0.0055 | 0.1066 ± 0.0009 |
| float64 | SPD bias 32 | False | 0.0355 ± 0.0004 | 0.1431 ± 0.0024 | — |
| float64 | SPD bias 32 | True | 0.0494 ± 0.0003 | 0.1964 ± 0.0027 | 0.1203 ± 0.0008 |

## Correctness and limits

- float32: maximum orthogonality error 4.77e-07; minimum output eigenvalue 0.297974; maximum absolute post-recenter output change 4.77e-06.
- float64: maximum orthogonality error 1.11e-15; minimum output eigenvalue 0.286362; maximum absolute post-recenter output change 8.44e-15.
- All training runs passed finite-gradient/output, positive-eigenvalue, orthogonality and enabled-chart continuity assertions.
- Frequent recentering has measurable compute cost; the loss differences here are too small to establish a practical advantage. This task is deliberately small and is not EEG accuracy or a reproduction of the paper’s RNN experiments.
- Historical preimplementation float32 BiMap forward/backward/SGD timing references were 0.214 ms (22→8) and 0.392 ms (64→32), using lr=0 and fixed weights. They are compute references, not directly comparable to the real training times above.
- GPU/MPS, dataset training and torch 2.3 execution were not measured locally; existing lowest-dependency CI covers the supported version floor. Adaptive optimizer state restarts on recentering when supplied; this benchmark uses stateless SGD.

## Verification

- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/ --ignore=tests/test_correctness.py -k "not braindecode" -q`: 859 passed, 149 skipped, 7 deselected. Slow/GPU/optional cases follow project skip rules; dataset correctness and Braindecode integration excluded.
- Ruff lint and formatting checks passed for changed Python files; targeted mypy passed for `modules/manifold.py`.
- Documentation collection encountered an installed-plugin conflict; `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest docs/ -q` passed (1 item). The new RST training/checkpoint example was executed separately and restored outputs matched.
- Existing MeanVar running buffers change shape on the first training forward. The checkpoint example initializes the restored model with the saved input layout before strict loading; this PR preserves that existing behavior.
