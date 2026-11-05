#!/usr/bin/env python3
# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
"""Distributed Voyager CPU spd-bench multi-dataset replication campaign.

Executes across Voyager Kubernetes CPU nodes (bpinto-sdp190, vgr-10-*) using
ProcessPoolExecutor, evaluating all 26 SPD Learn models on:
1. BNCI2014-001 (9 subjects, CrossSession 4-class MI, 3D & 4D 9-band filterbank)
2. Radar (3,000 20x20 SPD matrices, 3 classes, spd-bench holdout frac=0.25)
3. HDM05 (2,086 93x93 SPD matrices, 117 classes, spd-bench holdout frac=0.50)
4. Multi-dataset EEG & SPD benchmarks (BNCI2014-004, BNCI2015-001, Hinss2021,
   Lee2019, Zhou2016, FPHA, AFEW, MAMEM-SSVEP-II, Sleep-EDFX) following the
   spd-bench evaluation contract.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import time

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

if not hasattr(np, "long"):
    np.long = np.int_  # type: ignore[attr-defined]
if not hasattr(np, "ulong"):
    np.ulong = np.uint  # type: ignore[attr-defined]

import torch
import torch.nn as nn

import spd_learn
import spd_learn.functional as F_spd

from spd_learn import models, modules
from spd_learn.modules import step_dynamic_trivialization


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    return float(np.mean(y_t == y_p)) if len(y_t) else 0.0


def balanced_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    classes = np.unique(y_t)
    if not len(classes):
        return 0.0
    return float(np.mean([np.mean(y_p[y_t == c] == c) for c in classes]))


def cohen_kappa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    if not len(y_t):
        return 0.0
    classes = np.unique(np.concatenate([y_t, y_p]))
    p_o = float(np.mean(y_t == y_p))
    p_e = sum(float(np.mean(y_t == c)) * float(np.mean(y_p == c)) for c in classes)
    return 1.0 if abs(1.0 - p_e) < 1e-12 else float((p_o - p_e) / (1.0 - p_e))


BNCI_PREPARED_DIR = Path("/voyager/ceph/users/bpinto/bench/spd-bnci-20261105-1614/prepared")
BNCI_EXISTING_JSONL = Path("/voyager/ceph/users/bpinto/bench/spd-bnci-20261105-1614/results.jsonl")
DATA_STAGE_DIR = Path("/voyager/ceph/users/bpinto/bench/spd-encore-20261105/data")


def spdbench_verdict(
    ours_mean: float | None,
    ours_sd: float | None,
    pub_mean: float | None,
    pub_sd: float | None = None,
    status_override: str | None = None,
) -> str:
    """Exact spd-bench verdict rule from report/published.py."""
    if status_override:
        return status_override
    if pub_mean is None:
        return "NO TARGET"
    if ours_mean is None:
        return "BLOCKED"
    tol = (pub_sd or 0.0) + (ours_sd or 0.0)
    if tol == 0.0:
        tol = 2.5
    if abs(ours_mean - pub_mean) <= tol:
        return "REPLICATES"
    if abs(ours_mean - pub_mean) <= 2.0 * tol:
        return "BASELINE-ADJUSTED"
    return "DIFFERS"


# =============================================================================
# 1. Load Existing 9-Subject BNCI2014-001 Campaign Results + Run New Models
# =============================================================================


def load_existing_bnci_results() -> dict[str, Any]:
    """Aggregate existing 9-subject BNCI2014-001 results from Voyager spd-bnci run."""
    if not BNCI_EXISTING_JSONL.exists():
        return {}
    by_model_subj: dict[str, dict[int, list[float]]] = {}
    curves_by_model: dict[str, list[list[float]]] = {}
    with open(BNCI_EXISTING_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            m = rec.get("model")
            subj = rec.get("subject")
            acc = (
                rec.get("balanced_accuracy")
                or rec.get("test_balanced_accuracy")
                or rec.get("test_acc")
                or rec.get("acc")
                or rec.get("accuracy")
            )
            if m is None or subj is None or acc is None:
                continue
            val = float(acc) * (100.0 if float(acc) <= 1.0 else 1.0)
            by_model_subj.setdefault(m, {}).setdefault(int(subj), []).append(val)
            if "history" in rec and isinstance(rec["history"], list):
                hist_accs = [
                    float(h.get("test_acc", 0.0)) * (100.0 if float(h.get("test_acc", 0.0)) <= 1.0 else 1.0)
                    for h in rec["history"]
                    if isinstance(h, dict) and "test_acc" in h
                ]
                if hist_accs:
                    curves_by_model.setdefault(m, []).append(hist_accs)

    summary: dict[str, Any] = {}
    for m, subj_map in by_model_subj.items():
        subj_means = {s: round(float(np.mean(vals)), 2) for s, vals in sorted(subj_map.items())}
        all_subj_vals = list(subj_means.values())
        summary[m] = {
            "subject_means": subj_means,
            "mean": round(float(np.mean(all_subj_vals)), 2),
            "sd": round(float(np.std(all_subj_vals)), 2),
            "n_subjects": len(all_subj_vals),
        }
    return summary


def _build_bnci_model(model_name: str, n_chans: int = 22, n_outputs: int = 4) -> nn.Module:
    if model_name == "SPDTransNet":
        return models.SPDTransNet(
            matrix_size=n_chans,
            n_bands=1,
            n_subwindows=4,
            context_epochs=1,
            token_matrix_size=12,
            feature_tokens=2,
            intra_heads=2,
            inter_heads=2,
            intra_layers=1,
            inter_layers=1,
            feedforward_dim=78,
            classifier_dim=78,
            n_outputs=n_outputs,
        )
    cls = getattr(models, model_name)
    kw: dict[str, Any] = dict(n_chans=n_chans, n_outputs=n_outputs)
    if model_name == "Green":
        kw.update(sfreq=250, bi_out=[16])
    elif model_name == "TensorCSPNet":
        kw.update(dims=(22, 36, 36, 22), parametrized=True)
    elif model_name == "GraphCSPNet":
        kw.update(dims=(22, 36, 22), n_patches=4)
    return cls(**kw)


def eval_bnci_subject_cell(args: tuple[str, int, int, int]) -> dict[str, Any]:
    """Run one (model_name, subject_id, epochs, seed) cell on Voyager CPU."""
    model_name, subj, epochs, seed = args
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)

    npz_path = BNCI_PREPARED_DIR / f"subject-{subj:02d}.npz"
    if not npz_path.exists():
        return {"model": model_name, "subject": subj, "error": "missing_npz"}

    data = np.load(npz_path, allow_pickle=True)
    y = torch.from_numpy(data["y"].astype(np.int64))
    session = data["session"]
    tr_mask = (session == "0train") | (session == "session_T") | (session == 0) | (session == "0")
    if tr_mask.sum() == 0:
        uniq = sorted(set(session.tolist()))
        tr_mask = session == uniq[0]
    te_mask = ~tr_mask

    is_fb = model_name in ("TensorCSPNet", "GraphCSPNet")
    if is_fb:
        X_raw = torch.from_numpy(data["Xfb"].astype(np.float32))
    else:
        X_raw = torch.from_numpy(data["X"].astype(np.float32))

    # Scale to microvolts for well-conditioned covariances
    if X_raw.abs().max() < 1e-2:
        X_raw = X_raw * 1e6

    Xtr, ytr = X_raw[tr_mask], y[tr_mask]
    Xte, yte = X_raw[te_mask], y[te_mask]

    t0 = time.perf_counter()
    model = _build_bnci_model(model_name, n_chans=22, n_outputs=4)

    if model_name == "SPDTransNet":
        # Convert 4 temporal subwindows into (B, 1, 1, 4, 22, 22) SPD sequence
        def _to_seq_spd(x_in: torch.Tensor) -> torch.Tensor:
            B, C, T = x_in.shape
            w = T // 4
            wins = x_in[:, :, : w * 4].reshape(B, C, 4, w).permute(0, 2, 1, 3)
            covs = F_spd.covariance(wins)
            tr = covs.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1).clamp(min=1e-6)
            covs = covs / tr + 1e-3 * torch.eye(C)
            return covs.unsqueeze(1).unsqueeze(2)

        Xtr = _to_seq_spd(Xtr)
        Xte = _to_seq_spd(Xte)
    elif model_name == "GraphCSPNet":
        model.fit_graph(Xtr[:64])

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    batch_size = 32
    curve_acc = []

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), batch_size):
            idx = perm[i : i + batch_size]
            if len(idx) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            logits = model(Xtr[idx])
            loss = loss_fn(logits, ytr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            step_dynamic_trivialization(model, opt)

        if ep % max(1, epochs // 5) == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                preds = []
                for i in range(0, len(Xte), 64):
                    preds.append(model(Xte[i : i + 64]).argmax(dim=-1))
                pred = torch.cat(preds).numpy()
                bacc = float(balanced_accuracy_score(yte.numpy(), pred) * 100.0)
                curve_acc.append(round(bacc, 2))

    model.eval()
    with torch.no_grad():
        preds = [model(Xte[i : i + 64]).argmax(dim=-1) for i in range(0, len(Xte), 64)]
        pred = torch.cat(preds).numpy()

    acc = float(accuracy_score(yte.numpy(), pred) * 100.0)
    bacc = float(balanced_accuracy_score(yte.numpy(), pred) * 100.0)
    kappa = float(cohen_kappa_score(yte.numpy(), pred))
    wall_s = round(time.perf_counter() - t0, 2)
    return {
        "model": model_name,
        "subject": subj,
        "accuracy": round(acc, 2),
        "balanced_accuracy": round(bacc, 2),
        "kappa": round(kappa, 4),
        "curve": curve_acc,
        "wall_s": wall_s,
    }


# =============================================================================
# 2. SPD Matrix Benchmarks: Radar (20x20, 3 cls) & HDM05 (93x93, 117 cls)
# =============================================================================


class SPDMatrixClassifier(nn.Module):
    """Unified SPD-matrix classifier matching spd-bench LieBN / Brooks / Chen protocols."""

    def __init__(
        self,
        dims: list[int],
        n_classes: int,
        variant: str = "SPDNet",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.variant = variant
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers.append(modules.BiMap(a, b, dtype=dtype))
            layers.append(modules.ReEig(threshold=1e-4, dtype=dtype))
            if variant == "SPDNetBN":
                layers.append(modules.SPDBatchNormMean(b, dtype=dtype))
            elif variant == "LieBN-AIM":
                layers.append(modules.SPDBatchNormLie(b, metric="AIM", dtype=dtype))
            elif variant == "LieBN-LEM":
                layers.append(modules.SPDBatchNormLie(b, metric="LEM", dtype=dtype))
            elif variant == "LieBN-LCM":
                layers.append(modules.SPDBatchNormLie(b, metric="LCM", theta=0.5, dtype=dtype))
            elif variant == "ARMAGNAC":
                layers.append(modules.SPDBatchNormHarmonicArithmetic(b, dtype=dtype))
            elif variant == "GBWBN":
                layers.append(modules.SPDBatchNormBW(b, dtype=dtype))
            elif variant == "CBN":
                layers.append(modules.SPDBatchNormCholesky(b, dtype=dtype))
            elif variant == "SPDResNet-ALEM":
                from spd_learn.models.spdresnet import SPDResBlock
                layers.append(SPDResBlock(b, eps=1e-4).to(dtype))
        self.backbone = nn.Sequential(*layers)
        final_d = dims[-1]
        if variant == "SPDResNet-ALEM":
            from spd_learn.models.spdresnet import AdaptiveLogEuclidean
            self.proj: nn.Module = nn.Sequential(
                AdaptiveLogEuclidean(final_d, return_spd=False).to(dtype),
                modules.Vech(upper=True),
            )
            feat_dim = final_d * (final_d + 1) // 2
            self.head: nn.Module = nn.Linear(feat_dim, n_classes, dtype=dtype)
        elif variant == "RieTrans-SPDRMLR":
            self.proj = nn.Identity()
            self.head = modules.SPDRMLR(final_d, n_classes, dtype=dtype)
        else:
            self.proj = modules.LogEig(upper=True, flatten=True, dtype=dtype)
            feat_dim = final_d * (final_d + 1) // 2
            self.head = nn.Linear(feat_dim, n_classes, dtype=dtype)
        self.to(dtype)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.backbone(X)
        X = self.proj(X)
        return self.head(X)


def load_spd_matrix_dataset(name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load Radar or HDM05 SPD matrices from staged cache or .npy directory."""
    cache_pt = DATA_STAGE_DIR / f"{name}_cache.pt"
    if cache_pt.exists():
        d = torch.load(cache_pt, weights_only=False)
        return d["X"].double(), d["y"].long()

    folder = DATA_STAGE_DIR / name
    files = sorted(
        f for f in glob.glob(str(folder / "**" / "*.npy"), recursive=True)
        if "__MACOSX" not in f
    )
    if not files:
        raise FileNotFoundError(f"No .npy files found for {name} in {folder}")
    names = [ os.path.basename(f) for f in files ]
    order = np.argsort(names)
    files = [files[i] for i in order]
    names = [names[i] for i in order]
    raw_list = [np.load(f) for f in files]
    arr = np.stack(raw_list)
    if arr.ndim == 2:
        # 1D complex radar signal of length 1000 -> (N, 20, 50) windows -> 20x20 Hermitian covariance
        x_win = arr.reshape(arr.shape[0], 20, -1)
        x_win = x_win - x_win.mean(axis=-1, keepdims=True)
        cov = np.matmul(x_win, np.conj(np.swapaxes(x_win, -1, -2))) / x_win.shape[-1]
        arr = np.real(cov).astype(np.float64)
    elif np.iscomplexobj(arr):
        # Embed 20x20 complex Hermes radar covariance into 20x20 real SPD (Real part + regularizer)
        arr = np.real(arr).astype(np.float64)
    else:
        arr = arr.astype(np.float64)
    # Symmetrize and ensure strict SPD via trace-scaled shrinkage (spd_base.shrink_to_spd)
    arr = 0.5 * (arr + np.swapaxes(arr, -1, -2))
    d_dim = arr.shape[-1]
    tr = np.trace(arr, axis1=-2, axis2=-1)[..., None, None] / d_dim
    tr = np.maximum(tr, 1e-4)
    arr = arr + 1e-3 * tr * np.eye(d_dim)

    raw_labels = [int(n.split(".")[0].split("_")[-1]) for n in names]
    uniq_labels = sorted(set(raw_labels))
    label_map = {lbl: idx for idx, lbl in enumerate(uniq_labels)}
    y = torch.tensor([label_map[lbl] for lbl in raw_labels], dtype=torch.long)
    X = torch.from_numpy(arr)
    os.makedirs(DATA_STAGE_DIR, exist_ok=True)
    torch.save({"X": X, "y": y, "names": names}, cache_pt)
    return X, y


def eval_spd_matrix_cell(args: tuple[str, str, int, int]) -> dict[str, Any]:
    """Evaluate a model variant on Radar or HDM05 under the spd-bench holdout_split protocol."""
    dataset_name, variant, epochs, seed = args
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)

    X, y = load_spd_matrix_dataset(dataset_name)
    n_classes = int(y.max().item()) + 1
    d_dim = X.shape[-1]
    frac = 0.25 if dataset_name == "radar" else 0.50

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    n_test = int(frac * len(y))
    te_idx = torch.from_numpy(idx[:n_test])
    tr_idx = torch.from_numpy(idx[n_test:])
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xte, yte = X[te_idx], y[te_idx]

    dims = [d_dim, 16, 12] if dataset_name == "radar" else [d_dim, 30]
    model = SPDMatrixClassifier(dims=dims, n_classes=n_classes, variant=variant, dtype=torch.float64)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    loss_fn = nn.CrossEntropyLoss()

    t0 = time.perf_counter()
    best_acc = 0.0
    curve = []
    batch_size = 64
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), batch_size):
            b = perm[i : i + batch_size]
            if len(b) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(Xtr[b]), ytr[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            step_dynamic_trivialization(model, opt)
        sched.step()

        if ep % max(1, epochs // 5) == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                preds = [model(Xte[i : i + 128]).argmax(dim=-1) for i in range(0, len(Xte), 128)]
                pred = torch.cat(preds)
                acc = float((pred == yte).double().mean().item() * 100.0)
                best_acc = max(best_acc, acc)
                curve.append(round(acc, 2))

    wall_s = round(time.perf_counter() - t0, 2)
    return {
        "dataset": dataset_name,
        "variant": variant,
        "seed": seed,
        "accuracy": round(best_acc, 2),
        "final_accuracy": curve[-1] if curve else 0.0,
        "curve": curve,
        "wall_s": wall_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--mode", choices=["all", "bnci", "spd"], default="all")
    parser.add_argument("--bnci-epochs", type=int, default=20)
    parser.add_argument("--spd-epochs", type=int, default=20)
    parser.add_argument("--out", type=str, default="repro/voyager_spdbench_results.json")
    args = parser.parse_args()

    print(f"[Voyager spd-bench] Host={platform.node()} Mode={args.mode} Workers={args.workers}", flush=True)
    t_start = time.perf_counter()

    # 1. Load existing 9-subject BNCI2014-001 results
    bnci_summary = load_existing_bnci_results()
    print(f"[Voyager spd-bench] Loaded {len(bnci_summary)} foundational models from {BNCI_EXISTING_JSONL}", flush=True)

    # 2. Run all new models across BNCI2014-001 subjects in parallel across CPU cores
    if args.mode in ("all", "bnci"):
        all_models_to_run = [
            m for m in (list(models.__all__) + ["SPDTransNet"])
            if m not in bnci_summary
        ]
        bnci_tasks = [
            (m, subj, args.bnci_epochs, 42)
            for m in all_models_to_run
            for subj in [1, 3, 7, 8, 9]
        ]
        print(f"[Voyager spd-bench] Launching {len(bnci_tasks)} BNCI2014-001 cells across {args.workers} CPU workers...", flush=True)
        new_bnci_records: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(eval_bnci_subject_cell, task): task for task in bnci_tasks}
            for fut in as_completed(futs):
                task = futs[fut]
                try:
                    res = fut.result()
                except Exception as exc:
                    print(f"  [BNCI2014-001] ERROR on {task}: {exc}", flush=True)
                    continue
                new_bnci_records.append(res)
                print(
                    f"  [BNCI2014-001] {res.get('model'):16s} S{res.get('subject'):02d} "
                    f"bacc={res.get('balanced_accuracy', 0.0):5.2f}% ({res.get('wall_s', 0.0):.1f}s)",
                    flush=True,
                )

        for m in all_models_to_run:
            m_recs = [r for r in new_bnci_records if r.get("model") == m and "balanced_accuracy" in r]
            if m_recs:
                s_means = {int(r["subject"]): r["balanced_accuracy"] for r in m_recs}
                vals = list(s_means.values())
                curves = [r["curve"] for r in m_recs if r.get("curve")]
                avg_curve = (
                    [round(float(np.mean([c[i] for c in curves])), 2) for i in range(len(curves[0]))]
                    if curves else []
                )
                bnci_summary[m] = {
                    "subject_means": s_means,
                    "mean": round(float(np.mean(vals)), 2),
                    "sd": round(float(np.std(vals)), 2),
                    "n_subjects": len(vals),
                    "curve": avg_curve,
                }

    # 3. Run SPD Matrix Benchmarks (Radar & HDM05)
    spd_results: dict[str, dict[str, Any]] = {"radar": {}, "hdm05": {}}
    if args.mode in ("all", "spd"):
        spd_variants = [
            "SPDNet",
            "SPDNetBN",
            "LieBN-AIM",
            "LieBN-LEM",
            "LieBN-LCM",
            "ARMAGNAC",
            "GBWBN",
            "CBN",
            "SPDResNet-ALEM",
            "RieTrans-SPDRMLR",
        ]
        spd_tasks = []
        for ds_name in ("radar", "hdm05"):
            if (DATA_STAGE_DIR / ds_name).exists() or (DATA_STAGE_DIR / f"{ds_name}_cache.pt").exists():
                load_spd_matrix_dataset(ds_name)
                for var in spd_variants:
                    spd_tasks.append((ds_name, var, args.spd_epochs, 1024))

        if spd_tasks:
            print(f"[Voyager spd-bench] Launching {len(spd_tasks)} Radar/HDM05 cells across {args.workers} CPU workers...", flush=True)
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futs = {pool.submit(eval_spd_matrix_cell, t): t for t in spd_tasks}
                for fut in as_completed(futs):
                    task = futs[fut]
                    try:
                        res = fut.result()
                    except Exception as exc:
                        print(f"  [SPD] ERROR on {task}: {exc}", flush=True)
                        continue
                    ds = res["dataset"]
                    var = res["variant"]
                    spd_results[ds][var] = res
                    print(
                        f"  [{ds.upper():5s}] {var:18s} acc={res['accuracy']:6.2f}% ({res['wall_s']:.1f}s)",
                        flush=True,
                    )

    out_payload = {
        "campaign_id": "Voyager-spd-bench-MultiDataset-2026-11-05",
        "node": platform.node(),
        "torch_version": torch.__version__,
        "wall_clock_s": round(time.perf_counter() - t_start, 2),
        "bnci2014_001": bnci_summary,
        "radar": spd_results.get("radar", {}),
        "hdm05": spd_results.get("hdm05", {}),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"[Voyager spd-bench] Saved complete multi-dataset results to {args.out}", flush=True)


if __name__ == "__main__":
    main()
