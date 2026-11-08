#!/usr/bin/env python3
"""Aggregate all Voyager and local shard JSONs into repro/voyager_spdbench_results.json."""

from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SHARDS_DIR = Path(__file__).resolve().parent / "shards"
OUT_PATH = Path(__file__).resolve().parent / "voyager_spdbench_results.json"

MODEL_RENAME = {
    "GyroSPDAtt": "GyroAtt",
    "ALEMSPDNet": "ALEM",
    "RHOP-BFM": "RHOP",
}

BASE_SHARDS = [
    "bnci2014_001_p1.json",
    "bnci2014_001_p2.json",
    "bnci2014_004.json",
    "bnci2015_001.json",
    "bnci2014_002_weibo.json",
    "schirr_shin_zhou.json",
    "lee2019_cho2017.json",
    "missing5_all_datasets.json",
    "spd_matrices.json",
]

REPAIR_SHARDS = [
    "spd_matrices_repair.json",
    "bnci2014_001_repair.json",
    "multids_repair_a.json",
    "multids_repair_b.json",
    "repair_schirr.json",
    "repair_weibo.json",
    "repair_zhou.json",
]


def main() -> None:
    existing: dict[str, Any] = {}
    if OUT_PATH.exists():
        existing = json.loads(OUT_PATH.read_text())

    # Keyed by (ds_lower, model_name, subject_int)
    eeg_cells: dict[tuple[str, str, int], dict[str, Any]] = {}
    # Keyed by (ds_lower, variant, seed_int, frac_float)
    spd_cells: dict[tuple[str, str, int, float], dict[str, Any]] = {}

    shard_summaries: list[dict[str, Any]] = []
    nodes: set[str] = set(existing.get("nodes", []))

    ordered_files = []
    for name in BASE_SHARDS + REPAIR_SHARDS:
        p = SHARDS_DIR / name
        if p.exists():
            ordered_files.append(p)
    for p in sorted(SHARDS_DIR.glob("*.json")):
        if p not in ordered_files:
            ordered_files.append(p)

    for p in ordered_files:
        data = json.loads(p.read_text())
        if data.get("node"):
            nodes.add(str(data["node"]))
        eeg_recs = data.get("eeg_records", [])
        spd_recs = data.get("spd_records", [])
        shard_summaries.append(
            {
                "shard": data.get("shard", p.stem),
                "wall_clock_s": float(data.get("wall_s", 0.0)),
                "n_eeg": len(eeg_recs),
                "n_spd": len(spd_recs),
            }
        )
        for r in eeg_recs:
            ds = str(r["dataset"]).lower()
            m = MODEL_RENAME.get(str(r["model"]), str(r["model"]))
            s = int(r["subject"])
            eeg_cells[(ds, m, s)] = r
        for r in spd_recs:
            ds = str(r["dataset"]).lower()
            v = str(r["variant"])
            seed = int(r["seed"])
            frac = round(float(r.get("train_frac_scale", 1.0)), 2)
            spd_cells[(ds, v, seed, frac)] = r

    # Build eeg_datasets
    eeg_by_ds_model: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (ds, m, _s), rec in sorted(eeg_cells.items()):
        eeg_by_ds_model[ds][m].append(rec)

    eeg_datasets: dict[str, dict[str, Any]] = {}
    for ds, models_map in sorted(eeg_by_ds_model.items()):
        eeg_datasets[ds] = {}
        for m, recs in sorted(models_map.items()):
            recs_sorted = sorted(recs, key=lambda x: int(x["subject"]))
            subj_means = {
                str(int(r["subject"])): round(
                    float(r.get("balanced_accuracy", r.get("accuracy", 0.0))), 2
                )
                for r in recs_sorted
            }
            subj_kappas = {
                str(int(r["subject"])): round(float(r.get("kappa", 0.0)), 4)
                for r in recs_sorted
            }
            vals = list(subj_means.values())
            curves = [r.get("curve", []) for r in recs_sorted if r.get("curve")]
            min_c = min((len(c) for c in curves), default=0)
            avg_curve = (
                [
                    round(float(np.mean([c[i] for c in curves])), 2)
                    for i in range(min_c)
                ]
                if min_c > 0
                else []
            )
            eeg_datasets[ds][m] = {
                "subject_means": subj_means,
                "subject_kappas": subj_kappas,
                "mean": round(float(np.mean(vals)), 2) if vals else 0.0,
                "sd": round(float(np.std(vals)), 2) if vals else 0.0,
                "max": round(float(np.max(vals)), 2) if vals else 0.0,
                "n_subjects": len(vals),
                "curve": avg_curve,
            }

    # Build radar, hdm05, and radar_scarcity
    spd_full: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    scarcity_map: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for (ds, v, _seed, frac), rec in sorted(spd_cells.items()):
        if abs(frac - 1.0) < 1e-6:
            spd_full[ds][v].append(rec)
        if ds == "radar":
            scarcity_map[v][str(frac)].append(float(rec["accuracy"]))

    spd_out: dict[str, dict[str, Any]] = {"radar": {}, "hdm05": {}}
    for ds in ["radar", "hdm05"]:
        for v, recs in sorted(spd_full[ds].items()):
            recs_sorted = sorted(recs, key=lambda x: int(x["seed"]))
            seeds = [int(r["seed"]) for r in recs_sorted]
            accs = [round(float(r["accuracy"]), 2) for r in recs_sorted]
            finals = [float(r.get("final_accuracy", r["accuracy"])) for r in recs_sorted]
            walls = [float(r.get("wall_s", 0.0)) for r in recs_sorted]
            curves = [r.get("curve", []) for r in recs_sorted if r.get("curve")]
            min_c = min((len(c) for c in curves), default=0)
            avg_curve = (
                [
                    round(float(np.mean([c[i] for c in curves])), 2)
                    for i in range(min_c)
                ]
                if min_c > 0
                else []
            )
            spd_out[ds][v] = {
                "dataset": ds,
                "variant": v,
                "n_seeds": len(accs),
                "seeds": seeds,
                "seed_accuracies": accs,
                "accuracy": round(float(np.mean(accs)), 2) if accs else 0.0,
                "sd": round(float(np.std(accs)), 2) if accs else 0.0,
                "max_accuracy": round(float(np.max(accs)), 2) if accs else 0.0,
                "final_accuracy": round(float(np.mean(finals)), 2) if finals else 0.0,
                "curve": avg_curve,
                "wall_s": round(float(np.sum(walls)), 2),
            }

    radar_scarcity: dict[str, dict[str, Any]] = {}
    for v, frac_dict in sorted(scarcity_map.items()):
        radar_scarcity[v] = {}
        for frac_s, vals in sorted(frac_dict.items(), key=lambda kv: float(kv[0])):
            radar_scarcity[v][frac_s] = {
                "mean": round(float(np.mean(vals)), 2),
                "sd": round(float(np.std(vals)), 2),
                "n_seeds": len(vals),
            }

    payload = {
        "campaign_id": existing.get(
            "campaign_id", "Voyager-spd-bench-MultiDataset-2026-11-08"
        ),
        "nodes": sorted(nodes),
        "torch_version": existing.get("torch_version", "2.6.0+hpu_1.21.4"),
        "bnci_wall_clock_s": existing.get("bnci_wall_clock_s", 310.44),
        "spd_wall_clock_s": existing.get("spd_wall_clock_s", 169.49),
        "bnci2014_001": eeg_datasets.get("bnci2014_001", {}),
        "radar": spd_out["radar"],
        "hdm05": spd_out["hdm05"],
        "eeg_datasets": eeg_datasets,
        "heavy_campaign_summary": {
            "total_eeg_cells": len(eeg_cells),
            "total_spd_cells": len(spd_cells),
            "shards": shard_summaries,
        },
        "radar_scarcity": radar_scarcity,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(
        f"Aggregated {len(eeg_cells)} EEG cells across {len(eeg_datasets)} datasets and "
        f"{len(spd_cells)} SPD cells -> {OUT_PATH}"
    )


if __name__ == "__main__":
    main()
