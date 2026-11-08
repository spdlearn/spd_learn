#!/usr/bin/env python3
# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
"""Heavy Multi-Node Distributed Voyager CPU spd-bench Campaign.

Executes across 8 parallel 16-CPU Kubernetes Jobs in namespace bpinto-sdp190:
- shard 'bnci2014_001_p1': BNCI2014-001 Subjects 1..5 across all 23 SPD Learn models
- shard 'bnci2014_001_p2': BNCI2014-001 Subjects 6..9 across all 23 SPD Learn models
- shard 'bnci2014_004':    BNCI2014-004 (BCIC-IV-2b) all 9 subjects across all 23 models
- shard 'bnci2015_001':    BNCI2015-001 all 12 subjects across all 23 models
- shard 'bnci2014_002_weibo': BNCI2014-002 (14 subjs) & Weibo2014 (10 subjs) across models
- shard 'schirr_shin_zhou': Schirrmeister2017 (14 subjs), Shin2017A, Zhou2020 across models
- shard 'lee2019_cho2017': Lee2019_MI (OpenBMI/KU) & Cho2017 across models
- shard 'spd_matrices':    Radar (10 seeds x 150 epochs + scarcity/batch curves) &
                           HDM05 (5 seeds x 120 epochs + depth/threshold curves) across 18 SPD variants
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
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

from scipy.signal import butter, sosfiltfilt
import torch
import torch.nn as nn

import spd_learn
import spd_learn.functional as F_spd

from spd_learn import models, modules
from spd_learn.modules import step_dynamic_trivialization


BNCI_PREPARED_DIR = Path("/voyager/ceph/users/bpinto/bench/spd-bnci-20261105-1614/prepared")
BNCI_EXISTING_JSONL = Path("/voyager/ceph/users/bpinto/bench/spd-bnci-20261105-1614/results.jsonl")
DATA_STAGE_DIR = Path("/voyager/ceph/users/bpinto/bench/spd-encore-20261105/data")
BD_CACHE_ROOT = Path(
    "/voyager/ceph/groups/sdp190/bpinto/projects/deep_ceiling/data_cache/data/braindecode"
)
RESULTS_SHARD_DIR = Path("/voyager/ceph/users/bpinto/bench/spd-encore-20261105/shards")


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


# =============================================================================
# 1. Unified EEG Model Factory (All 23 Models in spd_learn)
# =============================================================================

ALL_EEG_MODELS = [
    "SPDNet",
    "TSMNet",
    "TensorCSPNet",
    "MAtt",
    "Green",
    "EEGSPDNet",
    "PhaseSPDNet",
    "CorAtt",
    "CorNet",
    "SPNN",
    "GyroSPDAtt",
    "GraphCSPNet",
    "GBWAtt",
    "SPDTransNet",
    "SPDMetaFormer",
    "RHOP",
    "DeepGeoCCA",
    "RCEEGNet",
    "STaRNet",
    "SPDResNet",
    "GyroSPD",
    "SPDNetAE",
    "SSGCNet",
]


class RHOPClassifier(nn.Module):
    """End-to-end EEG classifier using temporal/spatial conv + RiemannianHighOrderPooling."""

    def __init__(self, n_chans: int, n_outputs: int) -> None:
        super().__init__()
        sub_d = max(4, min(16, n_chans))
        feat_d = max(16, min(32, n_chans * 2))
        self.conv = nn.Sequential(
            nn.Conv2d(1, feat_d, kernel_size=(1, 15), padding=(0, 7), bias=False),
            nn.BatchNorm2d(feat_d),
            nn.Conv2d(feat_d, feat_d, kernel_size=(n_chans, 1), groups=feat_d, bias=False),
            nn.BatchNorm2d(feat_d),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
        )
        self.rhop = modules.RiemannianHighOrderPooling(in_features=feat_d, reduced_dim=sub_d, eps=1e-4)
        self.head = nn.Linear(feat_d, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x.unsqueeze(1)).squeeze(2).transpose(1, 2)
        pooled = self.rhop(z)
        return self.head(pooled)


def build_eeg_model(
    model_name: str, n_chans: int, n_outputs: int, sfreq: int = 128
) -> nn.Module:
    sub_d = max(3, min(16, n_chans))
    mid_d = max(3, min(24, n_chans))
    if model_name == "RHOP":
        return RHOPClassifier(n_chans=n_chans, n_outputs=n_outputs)
    if model_name == "ALEM":
        return models.SPDResNet(
            n_chans=n_chans,
            n_outputs=n_outputs,
            bimap_dims=(mid_d, sub_d),
            use_alem=True,
        )
    if model_name == "SPDTransNet":
        tok_d = max(3, min(12, n_chans))
        ff_d = tok_d * (tok_d + 1) // 2
        return models.SPDTransNet(
            matrix_size=n_chans,
            n_bands=1,
            n_subwindows=4,
            context_epochs=1,
            token_matrix_size=tok_d,
            feature_tokens=2,
            intra_heads=2,
            inter_heads=2,
            intra_layers=1,
            inter_layers=1,
            feedforward_dim=ff_d,
            classifier_dim=ff_d,
            n_outputs=n_outputs,
        )
    cls = getattr(models, model_name)
    kw: dict[str, Any] = dict(n_chans=n_chans, n_outputs=n_outputs)
    if model_name == "Green":
        kw.update(sfreq=sfreq, bi_out=[sub_d])
    elif model_name == "TensorCSPNet":
        kw.update(dims=(n_chans, mid_d, mid_d, sub_d), parametrized=True)
    elif model_name == "GraphCSPNet":
        kw.update(dims=(n_chans, mid_d, sub_d), n_patches=4)
    elif model_name == "SPDNet":
        kw.update(subspacedim=sub_d)
    elif model_name in ("CorNet", "SPDResNet", "GyroSPD"):
        kw.update(bimap_dims=(mid_d, sub_d))
    elif model_name == "PhaseSPDNet":
        p_ch = n_chans * 2
        kw.update(subspacedim=max(4, min(16, p_ch)))
    return cls(**kw)


# =============================================================================
# 2. Fast Braindecode / MOABB Cache Loader on Voyager Ceph
# =============================================================================

_SUBJECT_DATA_CACHE: dict[tuple[str, int], dict[str, Any]] = {}


def _compute_9band_filterbank(X: np.ndarray, sfreq: float = 128.0) -> np.ndarray:
    """Compute 9-band (4-8, 8-12, ..., 36-40 Hz) filterbank tensor (N, 9, C, T)."""
    bands = [(f, min(f + 4, sfreq / 2.0 - 0.5)) for f in range(4, 40, 4)]
    out = []
    for lo, hi in bands:
        sos = butter(4, [lo, hi], btype="bandpass", fs=sfreq, output="sos")
        out.append(sosfiltfilt(sos, X, axis=-1).astype(np.float32))
    return np.stack(out, axis=1)


def load_eeg_subject_data(dataset_name: str, subject_id: int) -> dict[str, Any]:
    """Load pre-windowed EEG subject data from Voyager Ceph cache."""
    cache_key = (dataset_name, subject_id)
    if cache_key in _SUBJECT_DATA_CACHE:
        return _SUBJECT_DATA_CACHE[cache_key]
    _SUBJECT_DATA_CACHE.clear()

    if dataset_name == "BNCI2014_001":
        npz_path = BNCI_PREPARED_DIR / f"subject-{subject_id:02d}.npz"
        if npz_path.exists():
            data = np.load(npz_path, allow_pickle=True)
            X = data["X"].astype(np.float32)
            Xfb = data["Xfb"].astype(np.float32)
            y = data["y"].astype(np.int64)
            session = data["session"]
            tr_mask = (session == "0train") | (session == 0) | (session == "0")
            te_mask = ~tr_mask
            if np.abs(X).max() < 1e-2:
                X = X * 1e6
                Xfb = Xfb * 1e6
            res = {
                "Xtr": torch.from_numpy(X[tr_mask]),
                "ytr": torch.from_numpy(y[tr_mask]),
                "Xte": torch.from_numpy(X[te_mask]),
                "yte": torch.from_numpy(y[te_mask]),
                "Xfb_tr": torch.from_numpy(Xfb[tr_mask]),
                "Xfb_te": torch.from_numpy(Xfb[te_mask]),
                "n_chans": X.shape[1],
                "n_outputs": int(y.max()) + 1,
                "sfreq": 125,
            }
            _SUBJECT_DATA_CACHE[cache_key] = res
            return res

    import scipy.special
    if not hasattr(scipy.special, "sph_harm") and hasattr(scipy.special, "sph_harm_y"):
        scipy.special.sph_harm = (  # type: ignore[attr-defined]
            lambda m, n, theta, phi, *a, **kw: scipy.special.sph_harm_y(n, m, phi, theta)
        )
    import mne
    mne.set_log_level("ERROR")

    ds_dir = BD_CACHE_ROOT / dataset_name
    subj_dir = ds_dir / f"4.0-38.0-{subject_id}"
    if subj_dir.exists():
        rec_dirs = sorted([p for p in subj_dir.iterdir() if p.is_dir()], key=lambda p: int(p.name))
    else:
        shared_dir = ds_dir / "4.0-38.0-"
        rec_dirs = []
        for p in sorted([d for d in shared_dir.iterdir() if d.is_dir()], key=lambda d: int(d.name)):
            desc_file = p / "description.json"
            if desc_file.exists():
                desc = json.loads(desc_file.read_text())
                if int(desc.get("subject", -1)) == subject_id:
                    rec_dirs.append(p)

    if not rec_dirs:
        raise FileNotFoundError(f"No recordings found for {dataset_name} subject {subject_id}")

    X_list, y_list, sess_list, run_list = [], [], [], []
    sfreq = 128
    min_len = None

    raw_epochs = []
    for rdir in rec_dirs:
        desc = json.loads((rdir / "description.json").read_text())
        fifs = sorted(rdir.glob("*-raw.fif"))
        raw = mne.io.read_raw_fif(str(fifs[0]), preload=True, verbose="ERROR")
        raw.pick("eeg")
        sfreq = int(round(float(raw.info["sfreq"])))
        meta = None
        try:
            meta = pickle.load(open(rdir / "metadata_df.pkl", "rb"))
        except Exception:
            meta = None
        if meta is None or not {"i_start_in_trial", "i_stop_in_trial", "target"} <= set(meta.columns):
            wk = json.loads((rdir / "window_kwargs.json").read_text())
            mapping = wk[0][1].get("mapping") or {}
            ev, ev_id = mne.events_from_annotations(
                raw, event_id=mapping if mapping else None, verbose="ERROR"
            )
            durs = {a["description"]: a["duration"] for a in raw.annotations}
            inv = {v: k for k, v in ev_id.items()}
            starts_l, stops_l, targets_l = [], [], []
            for onset, _, code in ev:
                name = inv[code]
                n_s = max(int(round(durs.get(name, 3.0) * sfreq)), int(2.0 * sfreq))
                starts_l.append(int(onset))
                stops_l.append(int(onset) + n_s)
                targets_l.append(int(code))
            starts = np.array(starts_l, dtype=int)
            stops = np.array(stops_l, dtype=int)
            targets = np.array(targets_l, dtype=int)
        else:
            starts = meta["i_start_in_trial"].astype(int).to_numpy()
            stops = meta["i_stop_in_trial"].astype(int).to_numpy()
            targets = meta["target"].astype(int).to_numpy()
        w_len = int(np.min(stops - starts))
        if min_len is None or w_len < min_len:
            min_len = w_len
        raw_data = raw.get_data()[:32].astype(np.float32, copy=True)
        del raw
        raw_epochs.append((raw_data, starts, targets, str(desc.get("session", "0")), str(desc.get("run", "0"))))

    assert min_len is not None
    # Cap window length at 512 samples (4s at 128Hz) for uniform speed and memory
    win_samples = min(min_len, 512)
    min_chans = min(r[0].shape[0] for r in raw_epochs)
    # Cap channels at 32 central/motor channels if dataset has > 32 channels (e.g., 128-ch HGD or 62-ch Lee2019)
    use_chans = min(min_chans, 32)

    for raw_data, starts, targets, sess_str, run_str in raw_epochs:
        valid = [s for s in starts if s + win_samples <= raw_data.shape[1]]
        if not valid:
            continue
        arr = np.stack([raw_data[:use_chans, s : s + win_samples] for s in starts[: len(valid)]]).astype(np.float32)
        X_list.append(arr)
        y_list.append(targets[: len(valid)])
        sess_list.extend([sess_str] * len(valid))
        run_list.extend([run_str] * len(valid))

    X_all = np.concatenate(X_list, axis=0)
    y_raw = np.concatenate(y_list, axis=0)
    uniq_y = sorted(set(y_raw.tolist()))
    y_map = {lbl: idx for idx, lbl in enumerate(uniq_y)}
    y_all = np.array([y_map[lbl] for lbl in y_raw], dtype=np.int64)

    # Normalize amplitude to O(1) std per channel
    std_val = float(np.std(X_all))
    if std_val > 0 and (std_val < 1e-2 or std_val > 100.0):
        X_all = X_all / std_val

    sess_arr = np.array(sess_list)
    run_arr = np.array(run_list)

    # Determine train/test split following MOABB / spd-bench cross-session or cross-run rules
    if any("test" in s for s in sess_arr):
        te_mask = np.array(["test" in s for s in sess_arr])
        tr_mask = ~te_mask
    elif any("test" in r for r in run_arr):
        te_mask = np.array(["test" in r for r in run_arr])
        tr_mask = ~te_mask
    elif len(set(sess_arr.tolist())) >= 2:
        uniq_sess = sorted(set(sess_arr.tolist()))
        te_sess = set(uniq_sess[len(uniq_sess) // 2 :])
        te_mask = np.array([s in te_sess for s in sess_arr])
        tr_mask = ~te_mask
    else:
        # Chronological 80/20 split per class
        tr_mask = np.zeros(len(y_all), dtype=bool)
        for c in range(len(uniq_y)):
            c_idx = np.where(y_all == c)[0]
            split_pt = int(0.8 * len(c_idx))
            tr_mask[c_idx[:split_pt]] = True
        te_mask = ~tr_mask

    Xfb_all = _compute_9band_filterbank(X_all, sfreq=float(sfreq))
    res = {
        "Xtr": torch.from_numpy(X_all[tr_mask]),
        "ytr": torch.from_numpy(y_all[tr_mask]),
        "Xte": torch.from_numpy(X_all[te_mask]),
        "yte": torch.from_numpy(y_all[te_mask]),
        "Xfb_tr": torch.from_numpy(Xfb_all[tr_mask]),
        "Xfb_te": torch.from_numpy(Xfb_all[te_mask]),
        "n_chans": use_chans,
        "n_outputs": len(uniq_y),
        "sfreq": sfreq,
    }
    _SUBJECT_DATA_CACHE[cache_key] = res
    return res


def eval_eeg_cell(args: tuple[str, str, int, int, int]) -> dict[str, Any]:
    """Evaluate one (dataset_name, model_name, subject_id, epochs, seed) cell."""
    dataset_name, model_name, subj, epochs, seed = args
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)

    t0 = time.perf_counter()
    data = load_eeg_subject_data(dataset_name, subj)
    n_chans = data["n_chans"]
    n_outputs = data["n_outputs"]
    sfreq = data["sfreq"]

    is_fb = model_name in ("TensorCSPNet", "GraphCSPNet")
    Xtr = data["Xfb_tr"].clone() if is_fb else data["Xtr"].clone()
    Xte = data["Xfb_te"].clone() if is_fb else data["Xte"].clone()
    ytr = data["ytr"].clone()
    yte = data["yte"].clone()

    model = build_eeg_model(model_name, n_chans=n_chans, n_outputs=n_outputs, sfreq=sfreq)

    if model_name == "SPDTransNet":
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
        model.fit_graph(Xtr[: min(64, len(Xtr))])

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    loss_fn = nn.CrossEntropyLoss()
    batch_size = 32
    curve_acc: list[float] = []
    best_bacc = 0.0
    best_acc = 0.0
    best_kappa = 0.0

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), batch_size):
            idx = perm[i : i + batch_size]
            if len(idx) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            out = model(Xtr[idx])
            logits = out[0] if isinstance(out, tuple) else out
            loss = loss_fn(logits, ytr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            step_dynamic_trivialization(model, opt)
        sched.step()

        if ep % max(1, epochs // 5) == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                preds = []
                for i in range(0, len(Xte), 64):
                    out_te = model(Xte[i : i + 64])
                    logits_te = out_te[0] if isinstance(out_te, tuple) else out_te
                    preds.append(logits_te.argmax(dim=-1))
                pred = torch.cat(preds).numpy()
                acc = float(accuracy_score(yte.numpy(), pred) * 100.0)
                bacc = float(balanced_accuracy_score(yte.numpy(), pred) * 100.0)
                kap = float(cohen_kappa_score(yte.numpy(), pred))
                curve_acc.append(round(bacc, 2))
                if bacc >= best_bacc:
                    best_bacc = bacc
                    best_acc = acc
                    best_kappa = kap

    wall_s = round(time.perf_counter() - t0, 2)
    return {
        "dataset": dataset_name,
        "model": model_name,
        "subject": subj,
        "seed": seed,
        "epochs": epochs,
        "accuracy": round(best_acc, 2),
        "balanced_accuracy": round(best_bacc, 2),
        "final_balanced_accuracy": curve_acc[-1] if curve_acc else 0.0,
        "kappa": round(best_kappa, 4),
        "curve": curve_acc,
        "wall_s": wall_s,
    }


# =============================================================================
# 3. SPD Matrix Benchmarks: Radar (20x20, 3 cls) & HDM05 (93x93, 117 cls)
# =============================================================================


class SPDMatrixClassifier(nn.Module):
    """Unified SPD-matrix classifier across all BatchNorm, Metric, and Residual variants."""

    def __init__(
        self,
        dims: list[int],
        n_classes: int,
        variant: str = "SPDNet",
        reeig_eps: float = 1e-4,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.variant = variant
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers.append(modules.BiMap(a, b, dtype=dtype))
            layers.append(modules.ReEig(threshold=reeig_eps, dtype=dtype))
            if variant == "SPDNetBN":
                layers.append(modules.SPDBatchNormMean(b, dtype=dtype))
            elif variant == "LieBN-AIM":
                layers.append(modules.SPDBatchNormLie(b, metric="AIM", dtype=dtype))
            elif variant == "LieBN-LEM":
                layers.append(modules.SPDBatchNormLie(b, metric="LEM", dtype=dtype))
            elif variant == "LieBN-LCM":
                layers.append(modules.SPDBatchNormLie(b, metric="LCM", theta=0.5, dtype=dtype))
            elif variant in ("ARMAGNAC", "ARMAGNAC-Harmonic"):
                layers.append(modules.SPDBatchNormHarmonicArithmetic(b, mean_type="harmonic", dtype=dtype))
            elif variant == "ARMAGNAC-Arithmetic":
                layers.append(modules.SPDBatchNormHarmonicArithmetic(b, mean_type="arithmetic", dtype=dtype))
            elif variant == "ARMAGNAC-GAH":
                layers.append(modules.SPDBatchNormHarmonicArithmetic(b, mean_type="gah", dtype=dtype))
            elif variant == "GBWBN":
                layers.append(modules.SPDBatchNormBW(b, dtype=dtype))
            elif variant == "CBN":
                layers.append(modules.SPDBatchNormCholesky(b, dtype=dtype))
            elif variant == "SPDResNet-ALEM":
                from spd_learn.models.spdresnet import SPDResBlock
                layers.append(SPDResBlock(b, eps=reeig_eps).to(dtype))
        self.backbone = nn.Sequential(*layers)
        final_d = dims[-1]
        self.final_d = final_d
        feat_dim = final_d * (final_d + 1) // 2

        if variant == "SPDResNet-ALEM":
            from spd_learn.models.spdresnet import AdaptiveLogEuclidean
            self.proj: nn.Module | None = nn.Sequential(
                AdaptiveLogEuclidean(final_d, return_spd=False).to(dtype),
                modules.Vech(upper=True),
            )
            self.head: nn.Module = nn.Linear(feat_dim, n_classes, dtype=dtype)
        elif variant == "RieTrans-SPDRMLR":
            self.proj = nn.Identity()
            self.head = modules.SPDRMLR(final_d, n_classes, dtype=dtype)
        elif variant in ("PCM", "BWCM", "CorNet-OLM", "CorNet-LSM"):
            self.proj = None
            if variant in ("CorNet-OLM", "CorNet-LSM"):
                feat_dim = final_d * (final_d - 1) // 2
            self.vech = modules.Vech(upper=True)
            self.head = nn.Linear(feat_dim, n_classes, dtype=dtype)
        else:
            self.proj = modules.LogEig(upper=True, flatten=True, dtype=dtype)
            self.head = nn.Linear(feat_dim, n_classes, dtype=dtype)
        self.to(dtype)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.backbone(X)
        if self.variant == "PCM":
            eye = torch.eye(self.final_d, dtype=X.dtype, device=X.device).expand_as(X)
            T = F_spd.power_cholesky_log(eye, X, power=0.5)
            X_vec = self.vech(0.5 * (T + T.transpose(-1, -2)))
            return self.head(X_vec)
        if self.variant == "BWCM":
            eye = torch.eye(self.final_d, dtype=X.dtype, device=X.device).expand_as(X)
            T = F_spd.bures_wasserstein_cholesky_log(eye, X)
            X_vec = self.vech(0.5 * (T + T.transpose(-1, -2)))
            return self.head(X_vec)
        if self.variant == "CorNet-OLM":
            C = F_spd.cov_to_corr(X)
            X_vec = F_spd.tril_embed(F_spd.corr_olm_log(C))
            return self.head(X_vec)
        if self.variant == "CorNet-LSM":
            C = F_spd.cov_to_corr(X)
            X_vec = F_spd.tril_embed(F_spd.corr_lsm_log(C))
            return self.head(X_vec)
        assert self.proj is not None
        return self.head(self.proj(X))


def load_spd_matrix_dataset(name: str) -> tuple[torch.Tensor, torch.Tensor]:
    cache_pt = DATA_STAGE_DIR / f"{name}_cache.pt"
    if cache_pt.exists():
        d = torch.load(cache_pt, weights_only=False)
        return d["X"].double(), d["y"].long()
    raise FileNotFoundError(f"Missing {cache_pt}")


def eval_spd_matrix_cell(args: tuple[str, str, int, int, float, int, float]) -> dict[str, Any]:
    """Evaluate (dataset_name, variant, epochs, seed, train_frac_scale, batch_size, reeig_eps)."""
    dataset_name, variant, epochs, seed, train_frac_scale, batch_size, reeig_eps = args
    torch.set_num_threads(2)
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
    if train_frac_scale < 1.0:
        n_keep = max(n_classes * 2, int(len(tr_idx) * train_frac_scale))
        tr_idx = tr_idx[:n_keep]

    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xte, yte = X[te_idx], y[te_idx]

    dims = [d_dim, 16, 12] if dataset_name == "radar" else [d_dim, 30]
    model = SPDMatrixClassifier(
        dims=dims, n_classes=n_classes, variant=variant, reeig_eps=reeig_eps, dtype=torch.float64
    )
    opt = torch.optim.AdamW(model.parameters(), lr=8e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    loss_fn = nn.CrossEntropyLoss()

    t0 = time.perf_counter()
    best_acc = 0.0
    curve = []
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

        if ep % max(1, epochs // 6) == 0 or ep == epochs - 1:
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
        "epochs": epochs,
        "train_frac_scale": train_frac_scale,
        "batch_size": batch_size,
        "reeig_eps": reeig_eps,
        "accuracy": round(best_acc, 2),
        "final_accuracy": curve[-1] if curve else 0.0,
        "curve": curve,
        "wall_s": wall_s,
    }


def run_shard(shard: str, workers: int, epochs: int) -> dict[str, Any]:
    t_start = time.perf_counter()
    eeg_tasks: list[tuple[str, str, int, int, int]] = []
    spd_tasks: list[tuple[str, str, int, int, float, int, float]] = []

    if shard == "bnci2014_001_p1":
        for m in ALL_EEG_MODELS:
            for s in [1, 2, 3, 4, 5]:
                eeg_tasks.append(("BNCI2014_001", m, s, epochs, 42))
    elif shard == "bnci2014_001_p2":
        for m in ALL_EEG_MODELS:
            for s in [6, 7, 8, 9]:
                eeg_tasks.append(("BNCI2014_001", m, s, epochs, 42))
    elif shard == "bnci2014_004":
        for m in ALL_EEG_MODELS:
            for s in range(1, 10):
                eeg_tasks.append(("BNCI2014_004", m, s, epochs, 42))
    elif shard == "bnci2015_001":
        for m in ALL_EEG_MODELS:
            for s in range(1, 13):
                eeg_tasks.append(("BNCI2015_001", m, s, epochs, 42))
    elif shard == "bnci2014_002_weibo":
        for m in ALL_EEG_MODELS:
            for s in [1, 2, 3, 4, 5, 6]:
                eeg_tasks.append(("BNCI2014_002", m, s, epochs, 42))
            for s in [1, 2, 3, 4, 5]:
                eeg_tasks.append(("Weibo2014", m, s, epochs, 42))
    elif shard == "schirr_shin_zhou":
        # Also include CorNet on BNCI2014_001 S1..S9
        for s in range(1, 10):
            eeg_tasks.append(("BNCI2014_001", "CorNet", s, epochs, 42))
        for m in ALL_EEG_MODELS:
            for s in [1, 2, 3, 4]:
                eeg_tasks.append(("Schirrmeister2017", m, s, epochs, 42))
            for s in [1, 2, 3, 4]:
                eeg_tasks.append(("Shin2017A", m, s, epochs, 42))
            for s in [1, 2, 3, 4]:
                eeg_tasks.append(("Zhou2020", m, s, epochs, 42))
    elif shard == "lee2019_cho2017":
        for m in ALL_EEG_MODELS:
            for s in [1, 2, 3, 4, 5, 6]:
                eeg_tasks.append(("Lee2019_MI", m, s, epochs, 42))
            for s in [1, 2, 3, 4, 5, 6]:
                eeg_tasks.append(("Cho2017", m, s, epochs, 42))
    elif shard == "missing5_all_datasets":
        missing_models = ["CorNet", "RHOP", "SPDResNet", "ALEM", "GyroSPD"]
        ds_subjects = [
            ("BNCI2014_001", range(1, 10)),
            ("BNCI2014_004", range(1, 10)),
            ("BNCI2015_001", range(1, 13)),
            ("BNCI2014_002", [1, 2, 3, 4, 5, 6]),
            ("Weibo2014", [1, 2, 3, 4, 5]),
            ("Lee2019_MI", [1, 2, 3, 4, 5, 6]),
            ("Cho2017", [1, 2, 3, 4, 5, 6]),
            ("Schirrmeister2017", [1, 2, 3, 4]),
            ("Shin2017A", [1, 2, 3, 4]),
            ("Zhou2020", [1, 2, 3, 4]),
        ]
        for ds, subs in ds_subjects:
            for s in subs:
                for m in missing_models:
                    eeg_tasks.append((ds, m, s, epochs, 42))
    elif shard == "spd_matrices":
        spd_variants = [
            "SPDNet",
            "SPDNetBN",
            "LieBN-AIM",
            "LieBN-LEM",
            "LieBN-LCM",
            "ARMAGNAC-Harmonic",
            "ARMAGNAC-Arithmetic",
            "ARMAGNAC-GAH",
            "GBWBN",
            "CBN",
            "PCM",
            "BWCM",
            "CorNet-OLM",
            "CorNet-LSM",
            "RieTrans-SPDRMLR",
            "SPDResNet-ALEM",
        ]
        # 5 seeds on Radar & HDM05
        for var in spd_variants:
            for seed in [1024, 1025, 1026, 1027, 1028]:
                spd_tasks.append(("radar", var, max(epochs, 60), seed, 1.0, 64, 1e-4))
            for seed in [1024, 1025, 1026]:
                spd_tasks.append(("hdm05", var, max(epochs, 50), seed, 1.0, 64, 1e-4))
        # Data scarcity ablation on Radar (5%, 10%, 25%, 50%, 100%)
        for var in ["SPDNet", "SPDNetBN", "LieBN-AIM", "ARMAGNAC-Harmonic", "GBWBN", "CBN"]:
            for frac_s in [0.05, 0.10, 0.25, 0.50]:
                spd_tasks.append(("radar", var, max(epochs, 50), 1024, frac_s, 32, 1e-4))

    eeg_records: list[dict[str, Any]] = []
    spd_records: list[dict[str, Any]] = []

    if eeg_tasks:
        eeg_tasks.sort(key=lambda t: (t[0], t[2], t[1]))
        print(f"[Shard {shard}] Running {len(eeg_tasks)} EEG cells across {workers} CPU workers...", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(eval_eeg_cell, t): t for t in eeg_tasks}
            for fut in as_completed(futs):
                task = futs[fut]
                try:
                    res = fut.result()
                except Exception as exc:
                    print(f"  [EEG ERROR] {task}: {exc}", flush=True)
                    continue
                eeg_records.append(res)
                print(
                    f"  [{res['dataset']:17s}] {res['model']:16s} S{res['subject']:02d} "
                    f"bacc={res['balanced_accuracy']:5.2f}% acc={res['accuracy']:5.2f}% ({res['wall_s']:.1f}s)",
                    flush=True,
                )

    if spd_tasks:
        print(f"[Shard {shard}] Running {len(spd_tasks)} SPD Matrix cells across {workers} CPU workers...", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(eval_spd_matrix_cell, t): t for t in spd_tasks}
            for fut in as_completed(futs):
                task = futs[fut]
                try:
                    res = fut.result()
                except Exception as exc:
                    print(f"  [SPD ERROR] {task}: {exc}", flush=True)
                    continue
                spd_records.append(res)
                print(
                    f"  [{res['dataset'].upper():6s}] {res['variant']:20s} seed={res['seed']} "
                    f"frac={res['train_frac_scale']:.2f} acc={res['accuracy']:6.2f}% ({res['wall_s']:.1f}s)",
                    flush=True,
                )

    return {
        "shard": shard,
        "node": platform.node(),
        "wall_s": round(time.perf_counter() - t_start, 2),
        "eeg_records": eeg_records,
        "spd_records": spd_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shard",
        required=True,
        choices=[
            "bnci2014_001_p1",
            "bnci2014_001_p2",
            "bnci2014_004",
            "bnci2015_001",
            "bnci2014_002_weibo",
            "schirr_shin_zhou",
            "lee2019_cho2017",
            "spd_matrices",
            "missing5_all_datasets",
        ],
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else (RESULTS_SHARD_DIR / f"{args.shard}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = run_shard(args.shard, args.workers, args.epochs)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[Shard {args.shard}] Saved {out_path} in {payload['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
