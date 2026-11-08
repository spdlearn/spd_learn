#!/usr/bin/env python3
"""Rebuild the 5 self-contained PR branches cleanly on top of origin/main."""

from __future__ import annotations

import subprocess
from pathlib import Path

WORKTREES_ROOT = Path("/Users/bruaristimunha/Projects/spd/spd_learn/.worktrees")
WORKTREE = WORKTREES_ROOT / "unified-normalized"
RUFF = "/Users/bruaristimunha/.cache/pre-commit/repoakt3esgl/py_env-python3.12/bin/ruff"
MYPY = "/Users/bruaristimunha/.cache/pre-commit/repo146rjn67/py_env-python3/bin/mypy"
SPHINX_LINT = "/Users/bruaristimunha/.cache/pre-commit/repoh355c1s6/py_env-python3/bin/sphinx-lint"
CODESPELL = "/Users/bruaristimunha/.cache/pre-commit/repoa01y5naz/py_env-python3.12/bin/codespell"


def run(cmd: list[str], cwd: Path = WORKTREE, check: bool = True) -> subprocess.CompletedProcess:
    res = subprocess.run(cmd, cwd=cwd, check=False, text=True, capture_output=True)
    if check and res.returncode != 0:
        print("STDOUT:", res.stdout)
        print("STDERR:", res.stderr)
        res.check_returncode()
    return res


# Base origin/main exports
BASE_MODELS = [
    "TensorCSPNet",
    "SPDNet",
    "TSMNet",
    "MAtt",
    "EEGSPDNet",
    "PhaseSPDNet",
    "Green",
]

BASE_MODULES = [
    # covariance
    "CovLayer",
    # regularize
    "TraceNorm",
    "Shrinkage",
    # manifold
    "PositiveDefiniteScalar",
    "SymmetricPositiveDefinite",
    "step_dynamic_trivialization",
    # bilinear
    "BiMap",
    "BiMapIncreaseDim",
    # modeig
    "ReEig",
    "LogEig",
    "ExpEig",
    # batchnorm
    "SPDBatchNormMean",
    "BatchReNorm",
    "SPDBatchNormMeanVar",
    "SPDBatchNormLie",
    # dropout
    "SPDDropout",
    # residual
    "LogEuclideanResidual",
    # utils
    "PatchEmbeddingLayer",
    "Vec",
    "Vech",
    # wavelet
    "WaveletConv",
]


PR_SPECS = [
    {
        "branch": "feat/pr-A-metrics-and-batchnorm",
        "worktree": "pr-A-metrics-and-batchnorm",
        "title": "feat(metrics,batchnorm): add PCM, BWCM, ARMAGNAC, GBWBN, and Cholesky BatchNorm",
        "files": [
            "spd_learn/functional/metrics/cholesky_product.py",
            "spd_learn/functional/metrics/log_cholesky.py",
            "spd_learn/modules/batchnorm.py",
            "tests/test_cholesky_product_metrics.py",
            "tests/test_new_batchnorms.py",
            "tests/test_integration.py",
            "docs/source/references.bib",
        ],
        "metrics_imports": """from .cholesky_product import (
    bwcm_distance,
    bwcm_geodesic,
    bwcm_mean,
    exp_map_bwcm,
    exp_map_pcm,
    log_map_bwcm,
    log_map_pcm,
    pcm_coords,
    pcm_distance,
    pcm_geodesic,
    pcm_inv_coords,
    pcm_mean,
)""",
        "metrics_all": [
            "pcm_coords",
            "pcm_inv_coords",
            "log_map_pcm",
            "exp_map_pcm",
            "pcm_distance",
            "pcm_geodesic",
            "pcm_mean",
            "log_map_bwcm",
            "exp_map_bwcm",
            "bwcm_distance",
            "bwcm_geodesic",
            "bwcm_mean",
        ],
        "func_extra_imports": "",
        "func_extra_all": [],
        "modules_extra_imports": """from .batchnorm import (
    SPDBatchNormBW,
    SPDBatchNormCholesky,
    SPDBatchNormHarmonic,
)""",
        "modules_extra_all": [
            "SPDBatchNormHarmonic",
            "SPDBatchNormBW",
            "SPDBatchNormCholesky",
        ],
        "modules_top_extra": [
            "SPDBatchNormBW",
            "SPDBatchNormCholesky",
            "SPDBatchNormHarmonic",
        ],
        "models_extra_imports": "",
        "models_extra_all": [],
        "filter_bank_models": ["TensorCSPNet"],
        "sequence_models": [],
        "sequence_modules": [],
    },
    {
        "branch": "feat/pr-B-correlation-manifold",
        "worktree": "pr-B-correlation-manifold",
        "title": "feat(correlation): add correlation manifold metrics (OLM, LSM) and CorAtt, CorNet models",
        "files": [
            "spd_learn/functional/correlation.py",
            "spd_learn/models/coratt.py",
            "spd_learn/models/cornet.py",
            "tests/test_coratt.py",
            "tests/test_integration.py",
            "docs/source/references.bib",
        ],
        "metrics_imports": "",
        "metrics_all": [],
        "func_extra_imports": """from .correlation import (
    cov_to_corr,
    exp_map_lsm,
    exp_map_olm,
    log_map_lsm,
    log_map_olm,
    off_diag,
    tril_embed,
    untril_embed,
)""",
        "func_extra_all": [
            "cov_to_corr",
            "off_diag",
            "log_map_lsm",
            "exp_map_lsm",
            "log_map_olm",
            "exp_map_olm",
            "tril_embed",
            "untril_embed",
        ],
        "modules_extra_imports": "",
        "modules_extra_all": [],
        "modules_top_extra": [],
        "models_extra_imports": """from .coratt import CorAtt, CorrAttention  # noqa: F401
from .cornet import CorNet""",
        "models_extra_all": ["CorAtt", "CorNet"],
        "filter_bank_models": ["TensorCSPNet"],
        "sequence_models": [],
        "sequence_modules": [],
    },
    {
        "branch": "feat/pr-C-riemannian-attention-transformers",
        "worktree": "pr-C-riemannian-attention-transformers",
        "title": "feat(attention): add GBWAtt, SPDTransNet, SPDMetaFormer, GyroAtt, SPNN/RieTrans, and RHOP",
        "files": [
            "spd_learn/functional/metrics/generalized_bures_wasserstein.py",
            "spd_learn/functional/spdtransnet.py",
            "spd_learn/modules/gbw.py",
            "spd_learn/modules/pooling.py",
            "spd_learn/modules/rietrans.py",
            "spd_learn/modules/spdtransnet.py",
            "spd_learn/models/gbwatt.py",
            "spd_learn/models/gyroatt.py",
            "spd_learn/models/spdmetaformer.py",
            "spd_learn/models/spdtransnet.py",
            "spd_learn/models/spnn.py",
            "tests/test_gbwatt.py",
            "tests/test_rhop.py",
            "tests/test_rietrans_gyroatt.py",
            "tests/test_spd_metaformer.py",
            "tests/test_spdtransnet.py",
            "tests/test_integration.py",
            "tests/test_correctness.py",
            "docs/source/references.bib",
        ],
        "metrics_imports": """from .generalized_bures_wasserstein import (
    bw_attention_mean,
    gbw_chart,
    gbw_distance,
    gbw_inverse_chart,
    gbw_mean,
)""",
        "metrics_all": [
            "gbw_chart",
            "gbw_inverse_chart",
            "gbw_distance",
            "gbw_mean",
            "bw_attention_mean",
        ],
        "func_extra_imports": """from .spdtransnet import (
    log_tokens_to_spd,
    spd_to_log_tokens,
    triangular_dimension,
)""",
        "func_extra_all": [
            "spd_to_log_tokens",
            "log_tokens_to_spd",
            "triangular_dimension",
        ],
        "modules_extra_imports": """from .gbw import GBWAttention, GBWGeometry, GBWPatchCovariance  # noqa: F401
from .pooling import RHOP, GaussianSPD
from .rietrans import SPDRMLR, SPDLinear
from .spdtransnet import (  # noqa: F401
    LearnedSinusoidalPosition,
    SPDInterEpochEncoder,
    SPDIntraEpochEncoder,
    SPDLogTokenizer,
    SPDTemporalEncoder,
    SPDTransformerEncoderLayer,
    SPMultiheadAttention,
)""",
        "modules_extra_all": [
            "GaussianSPD",
            "RHOP",
            "SPDRMLR",
            "SPDLinear",
        ],
        "modules_top_extra": [
            "GaussianSPD",
            "RHOP",
            "SPDLinear",
            "SPDRMLR",
            "SPMultiheadAttention",
        ],
        "models_extra_imports": """from .gbwatt import GBWAtt
from .gyroatt import GyroAtt, GyroSPDAttention, SymSPDAttention  # noqa: F401
from .spdmetaformer import SPDMetaFormer, SPDMetaFormerBlock  # noqa: F401
from .spdtransnet import SPDTransNet  # noqa: F401
from .spnn import SPNN""",
        "models_extra_all": [
            "GBWAtt",
            "SPNN",
            "GyroAtt",
            "SPDMetaFormer",
        ],
        "models_top_extra": ["SPDTransNet"],
        "filter_bank_models": ["TensorCSPNet"],
        "sequence_models": ["SPDTransNet"],
        "sequence_modules": [
            "SPDLogTokenizer",
            "SPMultiheadAttention",
            "SPDTransformerEncoderLayer",
            "LearnedSinusoidalPosition",
            "SPDTemporalEncoder",
            "SPDIntraEpochEncoder",
            "SPDInterEpochEncoder",
        ],
    },
    {
        "branch": "feat/pr-D-graph-and-multiview-spd",
        "worktree": "pr-D-graph-and-multiview-spd",
        "title": "feat(graph,multiview): add GraphCSPNet, DeepGeoCCA, SPDNetAE, and SSGCNet",
        "files": [
            "spd_learn/functional/graph.py",
            "spd_learn/modules/graph.py",
            "spd_learn/models/graphcsp.py",
            "spd_learn/models/deepgeocca.py",
            "spd_learn/models/spdnetae.py",
            "spd_learn/models/ssgcnet.py",
            "tests/test_graphcsp.py",
            "tests/test_deepgeocca.py",
            "tests/test_spdnet_ae_ssgcnet.py",
            "tests/test_integration.py",
            "tests/test_correctness.py",
            "tests/test_dynamic_trivialization.py",
            "docs/source/references.bib",
        ],
        "metrics_imports": "",
        "metrics_all": [],
        "func_extra_imports": "from .graph import normalize_spd_adjacency, spd_graph_aggregate",
        "func_extra_all": ["normalize_spd_adjacency", "spd_graph_aggregate"],
        "modules_extra_imports": """from .graph import (  # noqa: F401
    GraphBiMap,
    TimeFrequencyCovariances,
    TimeFrequencyGraph,
)""",
        "modules_extra_all": [],
        "modules_top_extra": [],
        "models_extra_imports": """from .deepgeocca import DeepGeoCCA, GeodesicCCALoss  # noqa: F401
from .graphcsp import GraphCSPNet
from .spdnetae import SPDDecoderBlock, SPDNetAE  # noqa: F401
from .ssgcnet import SSGCNet, TangentGraphSPDAggregation  # noqa: F401""",
        "models_extra_all": [
            "GraphCSPNet",
            "DeepGeoCCA",
            "SPDNetAE",
            "SSGCNet",
        ],
        "filter_bank_models": ["TensorCSPNet", "GraphCSPNet"],
        "sequence_models": [],
        "sequence_modules": [],
    },
    {
        "branch": "feat/pr-E-residual-and-spatiotemporal-spd",
        "worktree": "pr-E-residual-and-spatiotemporal-spd",
        "title": "feat(models): add RCEEGNet, STaRNet, SPDResNet, AdaptiveLogEuclidean (ALEM), and GyroSPD",
        "files": [
            "spd_learn/models/rceegnet.py",
            "spd_learn/models/starnet.py",
            "spd_learn/models/spdresnet.py",
            "spd_learn/models/gyrospd.py",
            "tests/test_rceegnet_starnet.py",
            "tests/test_spdresnet_gyrospd.py",
            "tests/test_integration.py",
            "tests/test_correctness.py",
            "docs/source/_static/models/starnet.png",
            "docs/source/references.bib",
        ],
        "metrics_imports": "",
        "metrics_all": [],
        "func_extra_imports": "",
        "func_extra_all": [],
        "modules_extra_imports": "",
        "modules_extra_all": [],
        "modules_top_extra": [],
        "models_extra_imports": """from .gyrospd import GyroSPD, GyroSPDLayer  # noqa: F401
from .rceegnet import RCEEGNet
from .spdresnet import AdaptiveLogEuclidean, SPDResBlock, SPDResNet  # noqa: F401
from .starnet import STaRNet""",
        "models_extra_all": [
            "RCEEGNet",
            "STaRNet",
            "SPDResNet",
            "GyroSPD",
        ],
        "filter_bank_models": ["TensorCSPNet"],
        "sequence_models": [],
        "sequence_modules": [],
    },
]


def write_init_files(spec: dict, wt: Path) -> None:
    # 1. spd_learn/functional/metrics/__init__.py
    base_metrics = run(["git", "show", "origin/main:spd_learn/functional/metrics/__init__.py"], cwd=wt).stdout
    if spec["metrics_imports"]:
        base_metrics = base_metrics.replace(
            "from .log_cholesky import (",
            spec["metrics_imports"] + "\nfrom .log_cholesky import (",
        )
        extra_all_str = "".join(f'    "{x}",\n' for x in spec["metrics_all"])
        base_metrics = base_metrics.replace(
            '    "log_cholesky_mean",\n]',
            f'    "log_cholesky_mean",\n{extra_all_str}]',
        )
    (wt / "spd_learn/functional/metrics/__init__.py").write_text(base_metrics)

    # 2. spd_learn/functional/__init__.py
    base_func = run(["git", "show", "origin/main:spd_learn/functional/__init__.py"], cwd=wt).stdout
    if spec["metrics_all"]:
        extra_m_imports = "".join(f"    {x},\n" for x in spec["metrics_all"])
        base_func = base_func.replace(
            "    spd_egrad2rgrad,\n)",
            f"    spd_egrad2rgrad,\n{extra_m_imports})",
        )
    if spec["func_extra_imports"]:
        base_func = base_func.replace(
            "from .wavelet import compute_gabor_wavelet",
            f"from .wavelet import compute_gabor_wavelet\n{spec['func_extra_imports']}",
        )
    all_func_extras = spec["metrics_all"] + spec["func_extra_all"]
    if all_func_extras:
        extra_f_all = "".join(f'    "{x}",\n' for x in all_func_extras)
        base_func = base_func.replace(
            '    "uwedge",\n]',
            f'    "uwedge",\n{extra_f_all}]',
        )
    (wt / "spd_learn/functional/__init__.py").write_text(base_func)

    # 3. spd_learn/modules/__init__.py
    base_mod = run(["git", "show", "origin/main:spd_learn/modules/__init__.py"], cwd=wt).stdout
    if spec["modules_extra_imports"]:
        base_mod = base_mod.replace(
            "from .wavelet import WaveletConv",
            f"from .wavelet import WaveletConv\n{spec['modules_extra_imports']}",
        )
    if spec["modules_extra_all"]:
        extra_mod_all = "".join(f'    "{x}",\n' for x in spec["modules_extra_all"])
        base_mod = base_mod.replace(
            '    "WaveletConv",\n]',
            f'    "WaveletConv",\n{extra_mod_all}]',
        )
    if spec["sequence_modules"]:
        seq_str = "\n__sequence_modules__ = [\n" + "".join(
            f'    "{x}",\n' for x in spec["sequence_modules"]
        ) + "]\n"
        base_mod += seq_str
    (wt / "spd_learn/modules/__init__.py").write_text(base_mod)

    # 4. spd_learn/models/__init__.py
    base_mdl = run(["git", "show", "origin/main:spd_learn/models/__init__.py"], cwd=wt).stdout
    if spec["models_extra_imports"]:
        base_mdl = base_mdl.replace(
            "from .tsmnet import TSMNet",
            f"from .tsmnet import TSMNet\n{spec['models_extra_imports']}",
        )
    if spec["models_extra_all"]:
        extra_mdl_all = "".join(f'    "{x}",\n' for x in spec["models_extra_all"])
        base_mdl = base_mdl.replace(
            '    "Green",\n]',
            f'    "Green",\n{extra_mdl_all}]',
        )
    fb_str = "[" + ", ".join(f'"{x}"' for x in spec["filter_bank_models"]) + "]"
    base_mdl = base_mdl.replace('__filter_bank_models__ = ["TensorCSPNet"]', f"__filter_bank_models__ = {fb_str}")
    if spec["sequence_models"]:
        seq_m_str = "\n__sequence_models__ = [" + ", ".join(f'"{x}"' for x in spec["sequence_models"]) + "]\n"
        base_mdl += seq_m_str
    (wt / "spd_learn/models/__init__.py").write_text(base_mdl)

    # 5. spd_learn/__init__.py
    base_top = run(["git", "show", "origin/main:spd_learn/__init__.py"], cwd=wt).stdout
    top_models = BASE_MODELS + spec["models_extra_all"] + spec.get("models_top_extra", [])
    top_models_import = "from .models import (\n" + "".join(f"    {m},\n" for m in sorted(top_models)) + ")"
    base_top = base_top.replace(
        "from .models import EEGSPDNet, Green, MAtt, PhaseSPDNet, SPDNet, TensorCSPNet, TSMNet",
        top_models_import,
    )
    if spec["modules_top_extra"]:
        extra_top_mod_import = "".join(f"    {m},\n" for m in spec["modules_top_extra"])
        base_top = base_top.replace(
            "    WaveletConv,\n)",
            f"    WaveletConv,\n{extra_top_mod_import})",
        )
    extra_top_all = spec["models_extra_all"] + spec.get("models_top_extra", []) + spec["modules_top_extra"]
    if extra_top_all:
        extra_top_all_str = "".join(f'    "{x}",\n' for x in extra_top_all)
        base_top = base_top.replace(
            '    "WaveletConv",\n]',
            f'    "WaveletConv",\n{extra_top_all_str}]',
        )
    (wt / "spd_learn/__init__.py").write_text(base_top)


def main() -> None:
    unified_sha = run(["git", "rev-parse", "feat/unified-normalized"]).stdout.strip()
    print(f"Unified normalized SHA: {unified_sha}")

    for spec in PR_SPECS:
        branch = spec["branch"]
        wt = WORKTREES_ROOT / spec["worktree"]
        print(f"\n=== Rebuilding {branch} in {wt.name} ===")
        run(["git", "checkout", "-f", branch], cwd=wt)
        run(["git", "reset", "--hard", "origin/main"], cwd=wt)
        run(["git", "clean", "-fd"], cwd=wt)
        # Checkout PR-specific files from unified_sha
        run(["git", "checkout", unified_sha, "--", *spec["files"]], cwd=wt)
        write_init_files(spec, wt)

        # Format & lint with ruff
        run([RUFF, "check", "--fix", "spd_learn", "tests"], cwd=wt)
        run([RUFF, "format", "spd_learn", "tests"], cwd=wt)
        run([RUFF, "check", "spd_learn", "tests"], cwd=wt)
        run([RUFF, "format", "--check", "spd_learn", "tests"], cwd=wt)

        # Run mypy
        res_mypy = run([MYPY, "--ignore-missing-imports", "--no-strict-optional", "--warn-unused-ignores", "spd_learn"], cwd=wt)
        print("  mypy:", res_mypy.stdout.strip())

        # Stage and commit
        run(
            [
                "git",
                "add",
                "spd_learn/__init__.py",
                "spd_learn/functional/__init__.py",
                "spd_learn/functional/metrics/__init__.py",
                "spd_learn/modules/__init__.py",
                "spd_learn/models/__init__.py",
                *spec["files"],
            ],
            cwd=wt,
        )
        commit_msg = f"{spec['title']}\n\nCo-authored-by: Lucas Heck <lucasheck17@gmail.com>"
        run(["git", "commit", "-m", commit_msg], cwd=wt)
        print(f"  Committed {branch}: {run(['git', 'rev-parse', '--short', 'HEAD'], cwd=wt).stdout.strip()}")

    print("\nAll 5 PR branches rebuilt and verified cleanly!")


if __name__ == "__main__":
    main()
