"""
.. _moabb-hydra-benchmark:

Benchmarking SPD Learn Models with MOABB and Hydra
===================================================

This tutorial demonstrates how to set up a comprehensive benchmarking
pipeline for SPD Learn models using MOABB datasets and Hydra for
configuration management. We compare multiple geometric deep learning
architectures on motor imagery EEG classification.

.. contents:: This example covers:
   :local:
   :depth: 2

"""

######################################################################
# Introduction
# ------------
#
# Reproducible machine learning experiments require systematic
# configuration management. This tutorial shows how to:
#
# 1. **Use Hydra** :cite:p:`hydra2019` for declarative experiment configuration
# 2. **Benchmark multiple models** from SPD Learn (SPDNet
#    :cite:p:`huang2017riemannian`, TSMNet, etc.)
# 3. **Leverage MOABB** :cite:p:`moabb2018` for standardized EEG dataset access
# 4. **Implement proper cross-validation** for reliable performance estimates
# 5. **Visualize and compare results** across models
#
# .. note::
#
#    Hydra is a powerful framework for managing complex configurations.
#    While this tutorial shows inline configuration for simplicity,
#    in practice you would use YAML files for better organization.
#
# .. important::
#
#    **Model-Specific Training Requirements**
#
#    Different SPD models have different training requirements:
#
#    - **SPDNet**: Works on covariance matrices, can use higher learning rates (1e-3)
#    - **TSMNet**: Works on raw signals with SPDBatchNormMeanVar
#      :cite:p:`kobler2022spd`, requires lower learning rate (1e-4) and more
#      epochs (100+) for stable SPD learning
#    - **EEGSPDNet**: Works on raw signals, also requires lower learning rate (1e-4)
#      and more epochs for the channel-specific convolutions to converge
#
#    This benchmark uses model-specific training configurations to ensure
#    each model achieves optimal performance.
#

######################################################################
# Setup and Imports
# -----------------
#
# We import the necessary libraries for this benchmark:
#
# - **MOABB**: Standardized EEG datasets and paradigms
# - **Braindecode**: EEGClassifier wrapper for PyTorch models
# - **SPD Learn**: Geometric deep learning models
# - **Hydra/OmegaConf**: Configuration management
#

import os
import tempfile
import warnings

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from braindecode import EEGClassifier
from einops.layers.torch import Rearrange
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from omegaconf import MISSING, OmegaConf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from skorch.callbacks import (
    Checkpoint,
    EarlyStopping,
    EpochScoring,
    GradientNormClipping,
    LRScheduler,
)
from skorch.dataset import ValidSplit
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from spd_learn.models import EEGSPDNet, SPDNet, TensorCSPNet, TSMNet


warnings.filterwarnings("ignore")

######################################################################
# Hydra Configuration with Dataclasses
# -------------------------------------
#
# Hydra uses structured configurations that can be defined as dataclasses.
# This provides type safety and autocompletion while maintaining the
# flexibility of YAML-based configuration.
#
# We define configuration schemas for:
#
# - **Model configurations**: Architecture-specific parameters
# - **Training configurations**: Optimizer, scheduler, and training settings
# - **Experiment configurations**: Dataset, paradigm, and evaluation settings
#


@dataclass
class ModelConfig:
    """Base configuration for all models.

    This dataclass defines the common parameters shared by all SPD Learn models.
    Model-specific configurations inherit from this class and add their own
    parameters.

    Parameters
    ----------
    name : str
        Name of the model (e.g., "SPDNet", "TSMNet", "EEGSPDNet").
    n_chans : int
        Number of input EEG channels.
    n_outputs : int
        Number of output classes for classification.
    """

    name: str = MISSING
    n_chans: int = MISSING
    n_outputs: int = MISSING


@dataclass
class SPDNetConfig(ModelConfig):
    """Configuration for SPDNet model.

    SPDNet operates on covariance matrices and can use higher learning rates.
    It performs a single BiMap + ReEig + LogEig transformation.

    Parameters
    ----------
    input_type : str, default="raw"
        Type of input data. "raw" computes covariance internally,
        "cov" expects pre-computed covariance matrices.
    subspacedim : int, optional
        Output dimension of BiMap layer. If None, uses n_chans.
    threshold : float, default=1e-4
        Eigenvalue threshold for ReEig layer to ensure numerical stability.
    upper : bool, default=True
        If True, use only upper triangular part in LogEig output.
    """

    name: str = "SPDNet"
    input_type: str = "raw"
    subspacedim: Optional[int] = None
    threshold: float = 1e-4
    upper: bool = True


@dataclass
class TSMNetConfig(ModelConfig):
    """Configuration for TSMNet model.

    TSMNet (Tangent Space Mapping Network) combines convolutional feature
    extraction with SPD processing and SPDBatchNormMeanVar :cite:p:`kobler2022spd` for
    domain adaptation.

    .. note::

       TSMNet requires lower learning rates (1e-4) and more epochs (100+)
       compared to SPDNet for stable training on the Riemannian manifold.

    Parameters
    ----------
    n_temp_filters : int, default=8
        Number of temporal convolution filters. More filters capture
        richer temporal dynamics but increase computation.
    temp_kernel_length : int, default=50
        Length of temporal convolution kernel. At 250Hz, 50 samples = 200ms.
    n_spatiotemp_filters : int, default=32
        Number of spatiotemporal filters after the spatial convolution.
    n_bimap_filters : int, default=16
        Output dimension of the BiMap layer. Controls the SPD manifold dimension.
    reeig_threshold : float, default=1e-4
        Eigenvalue threshold for ReEig to prevent numerical instability.
    """

    name: str = "TSMNet"
    n_temp_filters: int = 8
    temp_kernel_length: int = 50
    n_spatiotemp_filters: int = 32
    n_bimap_filters: int = 16
    reeig_threshold: float = 1e-4


@dataclass
class EEGSPDNetConfig(ModelConfig):
    """Configuration for EEGSPDNet model.

    EEGSPDNet uses channel-specific convolutions followed by covariance pooling
    and multiple BiMap layers for hierarchical SPD feature learning.

    .. note::

       EEGSPDNet requires lower learning rates (1e-4) and sufficient epochs
       for the channel-specific convolutions to learn meaningful features.

    Parameters
    ----------
    n_filters : int, default=4
        Number of convolutional filters per channel. Total filters = n_filters * n_chans.
    bimap_sizes : tuple, default=(2, 2)
        Tuple of (scale_factor, n_layers). Creates n_layers BiMap layers,
        each reducing dimension by scale_factor.
    filter_time_length : int, default=25
        Length of temporal filter. At 250Hz, 25 samples = 100ms.
    spd_drop_prob : float, default=0.0
        Dropout probability for SPDDropout layers. Set to 0 for stability.
    final_layer_drop_prob : float, default=0.5
        Standard dropout probability before the final classifier.
    """

    name: str = "EEGSPDNet"
    n_filters: int = 4
    bimap_sizes: tuple = (2, 2)
    filter_time_length: int = 25
    spd_drop_prob: float = 0.0
    final_layer_drop_prob: float = 0.5


@dataclass
class TensorCSPNetConfig(ModelConfig):
    """Configuration for TensorCSPNet model.

    TensorCSPNet is designed for filter bank paradigms, processing
    multi-frequency covariance tensors.

    .. note::

       TensorCSPNet requires FilterBankMotorImagery paradigm for data loading.

    Parameters
    ----------
    n_patches : int, default=4
        Number of temporal patches for local covariance computation.
    n_freqs : int, default=9
        Number of frequency bands. Must match the filter bank configuration.
    use_mlp : bool, default=False
        If True, use MLP instead of TCN for classification.
    tcn_channels : int, default=16
        Number of channels in TCN blocks.
    dims : tuple, default=(22, 36, 36, 22)
        Dimensions for BiMap layers in the network.
    """

    name: str = "TensorCSPNet"
    n_patches: int = 4
    n_freqs: int = 9
    use_mlp: bool = False
    tcn_channels: int = 16
    dims: tuple = (22, 36, 36, 22)


@dataclass
class TrainingConfig:
    """Configuration for training parameters.

    This dataclass contains all hyperparameters related to the training process.
    Note that model-specific overrides may be applied for optimal performance.

    Parameters
    ----------
    batch_size : int, default=32
        Number of samples per training batch.
    max_epochs : int, default=150
        Maximum number of training epochs.
    learning_rate : float, default=1e-3
        Initial learning rate for the optimizer.
    weight_decay : float, default=1e-4
        L2 regularization strength.
    gradient_clip_value : float, default=1.0
        Maximum gradient norm for gradient clipping. Essential for SPD networks.
    early_stopping_patience : int, default=30
        Number of epochs without improvement before stopping.
    lr_patience : int, default=15
        Number of epochs without improvement before reducing learning rate.
    lr_factor : float, default=0.5
        Factor by which to reduce learning rate on plateau.
    min_lr : float, default=1e-6
        Minimum learning rate after reductions.
    validation_split : float, default=0.1
        Fraction of training data to use for validation.
    seed : int, default=42
        Random seed for reproducibility.
    """

    batch_size: int = 32
    max_epochs: int = 150
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_value: float = 1.0
    early_stopping_patience: int = 30
    lr_patience: int = 15
    lr_factor: float = 0.5
    min_lr: float = 1e-6
    validation_split: float = 0.1
    seed: int = 42


@dataclass
class ModelTrainingOverrides:
    """Model-specific training parameter overrides.

    Different SPD models require different training configurations for optimal
    performance. This dataclass defines overrides for each model type.

    Parameters
    ----------
    learning_rate : float, optional
        Override learning rate for this model.
    max_epochs : int, optional
        Override maximum epochs for this model.
    batch_size : int, optional
        Override batch size for this model.
    optimizer : str, default="AdamW"
        Optimizer to use ("Adam" or "AdamW").
    """

    learning_rate: Optional[float] = None
    max_epochs: Optional[int] = None
    batch_size: Optional[int] = None
    optimizer: str = "AdamW"


@dataclass
class DataConfig:
    """Configuration for dataset and paradigm.

    Parameters
    ----------
    dataset_name : str, default="BNCI2014_001"
        Name of the MOABB dataset to use.
    subjects : List[int], default=[1]
        List of subject IDs to include in the benchmark.
    n_classes : int, default=4
        Number of motor imagery classes.
    paradigm : str, default="MotorImagery"
        Paradigm type ("MotorImagery" or "FilterBankMotorImagery").
    filters : List[List[int]], optional
        Filter bank specification for FilterBankMotorImagery.
    resample : float, optional
        Resampling frequency in Hz. None keeps original sampling rate.
    fmin : float, default=4.0
        Lower frequency bound for bandpass filter.
    fmax : float, default=38.0
        Upper frequency bound for bandpass filter.
    """

    dataset_name: str = "BNCI2014_001"
    subjects: List[int] = field(default_factory=lambda: [1])
    n_classes: int = 4
    paradigm: str = "MotorImagery"  # or "FilterBankMotorImagery"
    filters: Optional[List[List[int]]] = None
    resample: Optional[float] = None
    fmin: float = 4.0
    fmax: float = 38.0


@dataclass
class ExperimentConfig:
    """Main experiment configuration.

    This is the top-level configuration that combines all other configurations
    and defines the overall experiment structure.

    Parameters
    ----------
    training : TrainingConfig
        Training hyperparameters.
    data : DataConfig
        Dataset and paradigm configuration.
    models : List[str], default=["SPDNet", "TSMNet", "EEGSPDNet"]
        List of model names to benchmark.
    model_training_overrides : Dict[str, ModelTrainingOverrides], optional
        Model-specific training parameter overrides.
    n_folds : int, default=5
        Number of cross-validation folds.
    use_session_split : bool, default=True
        If True, use session-based split (train on session 0, test on session 1).
    device : str, default="auto"
        Device for training ("auto", "cpu", or "cuda").
    checkpoint_dir : str, optional
        Directory to save model checkpoints. If None, uses a temp directory.
    """

    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    models: List[str] = field(default_factory=lambda: ["SPDNet", "TSMNet", "EEGSPDNet"])
    model_training_overrides: Dict[str, Any] = field(default_factory=dict)
    n_folds: int = 5
    use_session_split: bool = True
    device: str = "auto"
    checkpoint_dir: Optional[str] = None


######################################################################
# Configuration Factory
# ---------------------
#
# We create a factory that generates model instances from configurations.
# This pattern allows easy switching between models via configuration.
#


def create_model(
    model_name: str,
    n_chans: int,
    n_outputs: int,
    **kwargs: Any,
) -> nn.Module:
    """Create a model instance from configuration.

    This factory function instantiates the appropriate SPD Learn model
    based on the provided name and configuration parameters.

    Parameters
    ----------
    model_name : str
        Name of the model to create. Supported: "SPDNet", "TSMNet",
        "EEGSPDNet", "TensorCSPNet".
    n_chans : int
        Number of input EEG channels.
    n_outputs : int
        Number of output classes.
    **kwargs : Any
        Additional model-specific parameters. See individual model
        configurations for available parameters.

    Returns
    -------
    nn.Module
        Instantiated PyTorch model ready for training.

    Raises
    ------
    ValueError
        If an unknown model name is provided.

    Examples
    --------
    >>> model = create_model("SPDNet", n_chans=22, n_outputs=4)
    >>> model = create_model("TSMNet", n_chans=22, n_outputs=4, n_temp_filters=8)
    """
    if model_name == "SPDNet":
        return SPDNet(
            n_chans=n_chans,
            n_outputs=n_outputs,
            input_type=kwargs.get("input_type", "raw"),
            subspacedim=kwargs.get("subspacedim", n_chans),
            threshold=kwargs.get("threshold", 1e-4),
            upper=kwargs.get("upper", True),
        )
    elif model_name == "TSMNet":
        return TSMNet(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_temp_filters=kwargs.get("n_temp_filters", 8),
            temp_kernel_length=kwargs.get("temp_kernel_length", 50),
            n_spatiotemp_filters=kwargs.get("n_spatiotemp_filters", 32),
            n_bimap_filters=kwargs.get("n_bimap_filters", 16),
            reeig_threshold=kwargs.get("reeig_threshold", 1e-4),
        )
    elif model_name == "EEGSPDNet":
        return EEGSPDNet(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_filters=kwargs.get("n_filters", 4),
            bimap_sizes=kwargs.get("bimap_sizes", (2, 2)),
            filter_time_length=kwargs.get("filter_time_length", 25),
            spd_drop_prob=kwargs.get("spd_drop_prob", 0.0),
            final_layer_drop_prob=kwargs.get("final_layer_drop_prob", 0.5),
        )
    elif model_name == "TensorCSPNet":
        # TensorCSPNet requires special input format handling
        n_freqs = kwargs.get("n_freqs", 9)
        model = nn.Sequential(
            Rearrange("b c t f -> b f c t"),
            TensorCSPNet(
                n_chans=n_chans,
                n_outputs=n_outputs,
                n_patches=kwargs.get("n_patches", 4),
                n_freqs=n_freqs,
                use_mlp=kwargs.get("use_mlp", False),
                tcn_channels=kwargs.get("tcn_channels", 16),
                dims=kwargs.get("dims", (n_chans, 36, 36, n_chans)),
            ),
        )
        return model
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported models: SPDNet, TSMNet, EEGSPDNet, TensorCSPNet"
        )


def get_default_model_training_overrides() -> Dict[str, Dict[str, Any]]:
    """Get default training parameter overrides for each model.

    Different SPD models require different training configurations:

    - **SPDNet**: Works on covariances, can use higher learning rates
    - **TSMNet**: Needs lower learning rate and more epochs for SPDBatchNormMeanVar
    - **EEGSPDNet**: Needs lower learning rate for channel-specific convolutions

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping model names to their training overrides.

    Examples
    --------
    >>> overrides = get_default_model_training_overrides()
    >>> overrides["TSMNet"]["learning_rate"]
    0.0001
    """
    return {
        "SPDNet": {
            "learning_rate": 1e-3,
            "max_epochs": 20,  # Reduced from 100 for faster documentation build
            "optimizer": "AdamW",
        },
        "TSMNet": {
            # TSMNet requires lower LR for stable SPD learning
            # Reference: plot_tsmnet_domain_adaptation.py
            "learning_rate": 1e-4,
            "max_epochs": 30,  # Reduced from 150 for faster documentation build
            "optimizer": "Adam",
        },
        "EEGSPDNet": {
            # EEGSPDNet requires lower LR for channel-specific convolutions
            # Reference: plot_eegspdnet.py
            "learning_rate": 1e-4,
            "max_epochs": 30,  # Reduced from 150 for faster documentation build
            "optimizer": "Adam",
        },
        "TensorCSPNet": {
            "learning_rate": 1e-3,
            "max_epochs": 20,  # Reduced from 100 for faster documentation build
            "optimizer": "AdamW",
        },
    }


######################################################################
# Example Hydra YAML Configuration
# --------------------------------
#
# In a production setting, you would store configurations in YAML files.
# Here's an example of what such a configuration file might look like:
#
# .. code-block:: yaml
#
#    # config/experiment/benchmark.yaml
#    defaults:
#      - _self_
#      - training: default
#      - data: bnci2014_001
#
#    models:
#      - SPDNet
#      - TSMNet
#      - EEGSPDNet
#
#    n_folds: 5
#    use_session_split: true
#    device: auto
#
#    # Model-specific training overrides
#    # These are CRITICAL for TSMNet and EEGSPDNet to train properly
#    model_training_overrides:
#      SPDNet:
#        learning_rate: 1e-3
#        max_epochs: 100
#        optimizer: AdamW
#      TSMNet:
#        learning_rate: 1e-4  # Lower LR for stable SPD learning
#        max_epochs: 150      # More epochs needed
#        optimizer: Adam
#      EEGSPDNet:
#        learning_rate: 1e-4  # Lower LR for channel convolutions
#        max_epochs: 150
#        optimizer: Adam
#
#    # Override model hyperparameters
#    model_params:
#      SPDNet:
#        subspacedim: null  # Use n_chans
#        threshold: 1e-4
#      TSMNet:
#        n_temp_filters: 8
#        temp_kernel_length: 50
#        n_spatiotemp_filters: 32
#        n_bimap_filters: 16
#      EEGSPDNet:
#        n_filters: 4
#        bimap_sizes: [2, 2]
#
# .. code-block:: yaml
#
#    # config/training/default.yaml
#    batch_size: 32
#    max_epochs: 150
#    learning_rate: 1e-3
#    weight_decay: 1e-4
#    gradient_clip_value: 1.0
#    early_stopping_patience: 30
#    lr_patience: 15
#    lr_factor: 0.5
#    min_lr: 1e-6
#    validation_split: 0.1
#    seed: 42
#
# .. code-block:: yaml
#
#    # config/data/bnci2014_001.yaml
#    dataset_name: BNCI2014_001
#    subjects: [1, 2]
#    n_classes: 4
#    paradigm: MotorImagery
#    fmin: 4.0
#    fmax: 38.0
#

######################################################################
# Setting Up the Benchmark Configuration
# --------------------------------------
#
# We create a configuration using OmegaConf, which provides the same
# functionality as Hydra YAML files but defined programmatically.
#

# Create experiment configuration
config = OmegaConf.structured(
    ExperimentConfig(
        training=TrainingConfig(
            batch_size=32,
            max_epochs=30,  # Reduced from 150 for faster documentation build
            learning_rate=1e-3,
            weight_decay=1e-4,
            gradient_clip_value=1.0,
            early_stopping_patience=30,
            lr_patience=15,
            lr_factor=0.5,
            validation_split=0.1,
            seed=42,
        ),
        data=DataConfig(
            dataset_name="BNCI2014_001",
            subjects=[1],  # Single subject for faster demonstration
            n_classes=4,
            paradigm="MotorImagery",
            fmin=4.0,
            fmax=38.0,
        ),
        models=["SPDNet", "TSMNet", "EEGSPDNet"],
        n_folds=1,  # Reduced from 3 for faster documentation build
        use_session_split=True,
        device="auto",
    )
)

print("Experiment Configuration:")
print(OmegaConf.to_yaml(config))

######################################################################
# Loading the Dataset
# -------------------
#
# We use MOABB to load the dataset with the configured parameters.
#

# Determine device
device = (
    "cuda"
    if config.device == "auto" and torch.cuda.is_available()
    else config.device
    if config.device != "auto"
    else "cpu"
)
print(f"\nUsing device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(config.training.seed)
np.random.seed(config.training.seed)

# Load dataset
dataset = BNCI2014_001()
paradigm = MotorImagery(
    n_classes=config.data.n_classes,
    fmin=config.data.fmin,
    fmax=config.data.fmax,
)

# Cache configuration for faster repeated runs
# Note: Cross-platform compatible cache configuration
# Set use=False if you encounter caching issues with older MOABB versions
# or on systems where the cache directory is not accessible
cache_config = dict(
    save_raw=False,
    save_epochs=False,
    save_array=True,
    use=False,  # Disable cache to avoid preload issues on some systems
    overwrite_raw=False,
    overwrite_epochs=False,
    overwrite_array=False,
)

print(f"\nLoading dataset: {config.data.dataset_name}")
print(f"Subjects: {config.data.subjects}")
print(f"Paradigm: {config.data.paradigm}")

X, labels, meta = paradigm.get_data(
    dataset=dataset,
    subjects=list(config.data.subjects),  # Convert OmegaConf list to Python list
    cache_config=cache_config,
)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)

n_chans = X.shape[1]
n_outputs = len(le.classes_)

print(f"\nData shape: {X.shape}")
print(f"Classes: {le.classes_}")
print(f"Number of channels: {n_chans}")
print(f"Number of classes: {n_outputs}")

######################################################################
# Creating the Benchmark Pipeline
# --------------------------------
#
# We define a benchmarking class that encapsulates the evaluation logic.
# This makes it easy to run experiments with different configurations.
#


class SPDLearnBenchmark:
    """Benchmark pipeline for SPD Learn models.

    This class provides a structured way to evaluate multiple models
    on EEG datasets using cross-validation with proper training configurations
    for each model type.

    The benchmark supports:

    - Session-based splits (train on session 0, test on session 1)
    - K-fold cross-validation
    - Model-specific training configurations
    - Early stopping and learning rate scheduling
    - Model checkpointing
    - Per-class accuracy breakdown

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration containing training, data, and model settings.
    X : np.ndarray
        Input data of shape (n_samples, n_channels, n_times).
    y : np.ndarray
        Labels of shape (n_samples,).
    meta : pd.DataFrame
        Metadata containing session and subject information.
    device : str, default="cpu"
        Device to use for training ("cpu" or "cuda").
    label_encoder : LabelEncoder, optional
        Fitted label encoder for class names.

    Attributes
    ----------
    results : List[Dict[str, Any]]
        List of evaluation results for each model.

    Examples
    --------
    >>> benchmark = SPDLearnBenchmark(config, X, y, meta, device="cuda")
    >>> results_df = benchmark.run_benchmark(model_configs)
    >>> print(results_df[["Model", "Accuracy", "Balanced Accuracy"]])
    """

    def __init__(
        self,
        config: ExperimentConfig,
        X: np.ndarray,
        y: np.ndarray,
        meta: pd.DataFrame,
        device: str = "cpu",
        label_encoder: Optional[LabelEncoder] = None,
    ) -> None:
        """Initialize the benchmark pipeline."""
        self.config = config
        self.X = X
        self.y = y
        self.meta = meta
        self.device = device
        self.label_encoder = label_encoder
        self.results: List[Dict[str, Any]] = []
        self._checkpoint_dir = config.checkpoint_dir or tempfile.mkdtemp()

        # Get default training overrides and merge with config overrides
        self.model_training_overrides = get_default_model_training_overrides()
        if (
            hasattr(config, "model_training_overrides")
            and config.model_training_overrides
        ):
            for model_name, overrides in config.model_training_overrides.items():
                if model_name in self.model_training_overrides:
                    self.model_training_overrides[model_name].update(overrides)
                else:
                    self.model_training_overrides[model_name] = overrides

    def _get_optimizer_class(self, optimizer_name: str) -> type:
        """Get the optimizer class from its name.

        Parameters
        ----------
        optimizer_name : str
            Name of the optimizer ("Adam" or "AdamW").

        Returns
        -------
        type
            PyTorch optimizer class.

        Raises
        ------
        ValueError
            If an unknown optimizer name is provided.
        """
        optimizers = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "SGD": torch.optim.SGD,
        }
        if optimizer_name not in optimizers:
            raise ValueError(
                f"Unknown optimizer: {optimizer_name}. "
                f"Supported: {list(optimizers.keys())}"
            )
        return optimizers[optimizer_name]

    def create_classifier(
        self,
        model: nn.Module,
        model_name: str,
        checkpoint_path: Optional[str] = None,
    ) -> EEGClassifier:
        """Create an EEGClassifier with model-specific training parameters.

        This method applies model-specific training overrides to ensure
        optimal performance for each model type.

        Parameters
        ----------
        model : nn.Module
            PyTorch model to wrap.
        model_name : str
            Name of the model (used to look up training overrides).
        checkpoint_path : str, optional
            Path to save model checkpoints.

        Returns
        -------
        EEGClassifier
            Configured classifier ready for training.
        """
        # Get model-specific training overrides
        overrides = self.model_training_overrides.get(model_name, {})

        # Apply overrides or use defaults
        learning_rate = overrides.get(
            "learning_rate", self.config.training.learning_rate
        )
        max_epochs = overrides.get("max_epochs", self.config.training.max_epochs)
        batch_size = overrides.get("batch_size", self.config.training.batch_size)
        optimizer_name = overrides.get("optimizer", "AdamW")
        optimizer_class = self._get_optimizer_class(optimizer_name)

        # Build callbacks
        callbacks = [
            (
                "train_acc",
                EpochScoring(
                    "accuracy",
                    lower_is_better=False,
                    on_train=True,
                    name="train_acc",
                ),
            ),
            (
                "gradient_clip",
                GradientNormClipping(
                    gradient_clip_value=self.config.training.gradient_clip_value
                ),
            ),
            # Learning rate scheduler - reduce on plateau
            (
                "lr_scheduler",
                LRScheduler(
                    policy=ReduceLROnPlateau,
                    mode="min",
                    factor=self.config.training.lr_factor,
                    patience=self.config.training.lr_patience,
                    min_lr=self.config.training.min_lr,
                    monitor="valid_loss",
                ),
            ),
            # Early stopping
            (
                "early_stopping",
                EarlyStopping(
                    monitor="valid_loss",
                    patience=self.config.training.early_stopping_patience,
                    threshold=1e-4,
                    threshold_mode="rel",
                    lower_is_better=True,
                ),
            ),
        ]

        # Add checkpointing if path is provided
        if checkpoint_path:
            callbacks.append(
                (
                    "checkpoint",
                    Checkpoint(
                        monitor="valid_loss_best",
                        f_pickle=None,
                        dirname=os.path.dirname(checkpoint_path),
                        f_params=os.path.basename(checkpoint_path),
                    ),
                )
            )

        return EEGClassifier(
            model,
            criterion=torch.nn.CrossEntropyLoss,
            optimizer=optimizer_class,
            optimizer__lr=learning_rate,
            optimizer__weight_decay=self.config.training.weight_decay,
            train_split=ValidSplit(
                self.config.training.validation_split,
                stratified=True,
                random_state=self.config.training.seed,
            ),
            batch_size=batch_size,
            max_epochs=max_epochs,
            callbacks=callbacks,
            device=self.device,
            verbose=0,  # Reduce verbosity for benchmark
        )

    def _compute_per_class_accuracy(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute per-class accuracy.

        Parameters
        ----------
        y_true : np.ndarray
            True labels.
        y_pred : np.ndarray
            Predicted labels.

        Returns
        -------
        Dict[str, float]
            Dictionary mapping class names to their accuracies.
        """
        per_class_acc = {}
        unique_classes = np.unique(y_true)

        for cls in unique_classes:
            mask = y_true == cls
            if mask.sum() > 0:
                cls_acc = (y_pred[mask] == y_true[mask]).mean()
                # Get class name if label encoder is available
                if self.label_encoder is not None:
                    cls_name = self.label_encoder.classes_[cls]
                else:
                    cls_name = str(cls)
                per_class_acc[cls_name] = cls_acc

        return per_class_acc

    def evaluate_model(
        self,
        model_name: str,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate a single model using cross-validation.

        This method handles:

        - Creating fresh model instances for each fold
        - Applying model-specific training configurations
        - Computing accuracy, balanced accuracy, and per-class metrics
        - Saving training history and confusion matrices

        Parameters
        ----------
        model_name : str
            Name of the model to evaluate.
        model_params : dict, optional
            Additional model parameters to override defaults.

        Returns
        -------
        dict
            Dictionary containing evaluation results including:

            - model: Model name
            - mean_accuracy: Mean accuracy across folds
            - std_accuracy: Standard deviation of accuracy
            - mean_balanced_accuracy: Mean balanced accuracy
            - std_balanced_accuracy: Standard deviation of balanced accuracy
            - per_class_accuracy: Per-class accuracy breakdown
            - fold_results: Detailed results for each fold
            - model_params: Parameters used for the model
            - training_overrides: Training parameters used

        Raises
        ------
        RuntimeError
            If training fails for all folds.
        """
        model_params = model_params or {}
        n_chans = self.X.shape[1]

        # Get training overrides for display
        overrides = self.model_training_overrides.get(model_name, {})
        lr = overrides.get("learning_rate", self.config.training.learning_rate)
        epochs = overrides.get("max_epochs", self.config.training.max_epochs)
        optimizer = overrides.get("optimizer", "AdamW")

        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        print(f"Training config: lr={lr}, max_epochs={epochs}, optimizer={optimizer}")

        fold_results = []
        all_y_true = []
        all_y_pred = []
        failed_folds = []

        if self.config.use_session_split:
            # Use session-based split (train on session 0, test on session 1)
            train_idx = self.meta.query("session == '0train'").index.to_numpy()
            test_idx = self.meta.query("session == '1test'").index.to_numpy()

            try:
                # Create fresh model
                model = create_model(
                    model_name,
                    n_chans=n_chans,
                    n_outputs=n_outputs,
                    **model_params,
                )

                checkpoint_path = os.path.join(
                    self._checkpoint_dir, f"{model_name}_fold0_best.pt"
                )
                clf = self.create_classifier(model, model_name, checkpoint_path)

                print(f"Training on {len(train_idx)} samples...")
                clf.fit(self.X[train_idx], self.y[train_idx])

                # Get actual epochs trained (may stop early)
                actual_epochs = len(clf.history)
                print(f"Training completed in {actual_epochs} epochs")

                # Evaluate
                y_pred = clf.predict(self.X[test_idx])
                acc = accuracy_score(self.y[test_idx], y_pred)
                bal_acc = balanced_accuracy_score(self.y[test_idx], y_pred)
                per_class_acc = self._compute_per_class_accuracy(
                    self.y[test_idx], y_pred
                )

                # Store predictions for confusion matrix
                all_y_true.extend(self.y[test_idx])
                all_y_pred.extend(y_pred)

                fold_results.append(
                    {
                        "fold": 0,
                        "accuracy": acc,
                        "balanced_accuracy": bal_acc,
                        "per_class_accuracy": per_class_acc,
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        "actual_epochs": actual_epochs,
                        "history": clf.history,
                    }
                )

                print(f"  Accuracy: {acc:.4f}, Balanced Acc: {bal_acc:.4f}")
                print(f"  Per-class: {per_class_acc}")

            except Exception as e:
                print(f"  ERROR: Training failed - {str(e)}")
                failed_folds.append((0, str(e)))

        else:
            # Use k-fold cross-validation
            skf = StratifiedKFold(
                n_splits=self.config.n_folds,
                shuffle=True,
                random_state=self.config.training.seed,
            )

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(self.X, self.y)):
                print(f"\nFold {fold_idx + 1}/{self.config.n_folds}")

                try:
                    # Create fresh model for each fold
                    model = create_model(
                        model_name,
                        n_chans=n_chans,
                        n_outputs=n_outputs,
                        **model_params,
                    )

                    checkpoint_path = os.path.join(
                        self._checkpoint_dir,
                        f"{model_name}_fold{fold_idx}_best.pt",
                    )
                    clf = self.create_classifier(model, model_name, checkpoint_path)

                    print(f"  Training on {len(train_idx)} samples...")
                    clf.fit(self.X[train_idx], self.y[train_idx])

                    # Get actual epochs trained
                    actual_epochs = len(clf.history)
                    print(f"  Completed in {actual_epochs} epochs")

                    # Evaluate
                    y_pred = clf.predict(self.X[test_idx])
                    acc = accuracy_score(self.y[test_idx], y_pred)
                    bal_acc = balanced_accuracy_score(self.y[test_idx], y_pred)
                    per_class_acc = self._compute_per_class_accuracy(
                        self.y[test_idx], y_pred
                    )

                    # Store predictions for confusion matrix
                    all_y_true.extend(self.y[test_idx])
                    all_y_pred.extend(y_pred)

                    fold_results.append(
                        {
                            "fold": fold_idx,
                            "accuracy": acc,
                            "balanced_accuracy": bal_acc,
                            "per_class_accuracy": per_class_acc,
                            "n_train": len(train_idx),
                            "n_test": len(test_idx),
                            "actual_epochs": actual_epochs,
                            "history": clf.history,
                        }
                    )

                    print(f"  Accuracy: {acc:.4f}, Balanced Acc: {bal_acc:.4f}")

                except Exception as e:
                    print(f"  ERROR: Fold {fold_idx + 1} failed - {str(e)}")
                    failed_folds.append((fold_idx, str(e)))

        # Check if any folds succeeded
        if not fold_results:
            print(f"\nWARNING: All folds failed for {model_name}")
            result = {
                "model": model_name,
                "mean_accuracy": 0.0,
                "std_accuracy": 0.0,
                "mean_balanced_accuracy": 0.0,
                "std_balanced_accuracy": 0.0,
                "per_class_accuracy": {},
                "fold_results": [],
                "model_params": model_params,
                "training_overrides": overrides,
                "failed_folds": failed_folds,
                "confusion_matrix": None,
            }
            self.results.append(result)
            return result

        # Aggregate results
        # Compute aggregate per-class accuracy
        agg_per_class_acc = {}
        for fold_result in fold_results:
            for cls_name, acc in fold_result["per_class_accuracy"].items():
                if cls_name not in agg_per_class_acc:
                    agg_per_class_acc[cls_name] = []
                agg_per_class_acc[cls_name].append(acc)

        mean_per_class_acc = {
            cls: np.mean(accs) for cls, accs in agg_per_class_acc.items()
        }

        # Compute confusion matrix
        cm = None
        if all_y_true and all_y_pred:
            cm = confusion_matrix(all_y_true, all_y_pred)

        result = {
            "model": model_name,
            "mean_accuracy": np.mean([r["accuracy"] for r in fold_results]),
            "std_accuracy": np.std([r["accuracy"] for r in fold_results]),
            "mean_balanced_accuracy": np.mean(
                [r["balanced_accuracy"] for r in fold_results]
            ),
            "std_balanced_accuracy": np.std(
                [r["balanced_accuracy"] for r in fold_results]
            ),
            "per_class_accuracy": mean_per_class_acc,
            "fold_results": fold_results,
            "model_params": model_params,
            "training_overrides": overrides,
            "failed_folds": failed_folds,
            "confusion_matrix": cm,
        }

        self.results.append(result)
        return result

    def run_benchmark(
        self, model_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> pd.DataFrame:
        """Run benchmark on all configured models.

        Parameters
        ----------
        model_configs : dict, optional
            Dictionary mapping model names to their architecture parameters.
            Training parameters are handled separately via model_training_overrides.

        Returns
        -------
        pd.DataFrame
            DataFrame containing benchmark results for all models.

        Examples
        --------
        >>> model_configs = {
        ...     "SPDNet": {"subspacedim": 22},
        ...     "TSMNet": {"n_temp_filters": 8},
        ... }
        >>> results_df = benchmark.run_benchmark(model_configs)
        """
        model_configs = model_configs or {}

        for model_name in self.config.models:
            params = model_configs.get(model_name, {})
            try:
                self.evaluate_model(model_name, params)
            except Exception as e:
                print(f"\nERROR: Failed to evaluate {model_name}: {str(e)}")
                # Add a placeholder result
                self.results.append(
                    {
                        "model": model_name,
                        "mean_accuracy": 0.0,
                        "std_accuracy": 0.0,
                        "mean_balanced_accuracy": 0.0,
                        "std_balanced_accuracy": 0.0,
                        "per_class_accuracy": {},
                        "fold_results": [],
                        "model_params": params,
                        "training_overrides": {},
                        "failed_folds": [(0, str(e))],
                        "confusion_matrix": None,
                    }
                )

        return self.get_results_dataframe()

    def get_results_dataframe(self) -> pd.DataFrame:
        """Get results as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with benchmark results including accuracy metrics,
            training configuration, and per-class breakdown.
        """
        records = []
        for r in self.results:
            # Format per-class accuracy
            per_class_str = ", ".join(
                [f"{cls}: {acc:.2f}" for cls, acc in r["per_class_accuracy"].items()]
            )

            # Get training info
            overrides = r.get("training_overrides", {})
            lr = overrides.get("learning_rate", self.config.training.learning_rate)
            optimizer = overrides.get("optimizer", "AdamW")

            records.append(
                {
                    "Model": r["model"],
                    "Accuracy": f"{r['mean_accuracy']:.4f} +/- {r['std_accuracy']:.4f}",
                    "Balanced Accuracy": f"{r['mean_balanced_accuracy']:.4f} +/- {r['std_balanced_accuracy']:.4f}",
                    "Mean Acc": r["mean_accuracy"],
                    "Std Acc": r["std_accuracy"],
                    "Per-Class Acc": per_class_str,
                    "LR": lr,
                    "Optimizer": optimizer,
                    "Failed Folds": len(r.get("failed_folds", [])),
                }
            )
        return pd.DataFrame(records)


######################################################################
# Running the Benchmark
# ---------------------
#
# Now we run the benchmark with our configured models.
# Note the model-specific configurations for optimal performance.
#

# Define model-specific architecture configurations
model_configs = {
    "SPDNet": {
        "subspacedim": n_chans,
        "threshold": 1e-4,
    },
    "TSMNet": {
        # Architecture parameters from plot_tsmnet_domain_adaptation.py
        "n_temp_filters": 8,
        "temp_kernel_length": 50,  # 200ms at 250Hz
        "n_spatiotemp_filters": 32,
        "n_bimap_filters": 16,
    },
    "EEGSPDNet": {
        # Architecture parameters from plot_eegspdnet.py
        "n_filters": 4,
        "bimap_sizes": (2, 2),
        "filter_time_length": 25,  # 100ms at 250Hz
        "spd_drop_prob": 0.0,  # Disable SPD dropout for stability
        "final_layer_drop_prob": 0.5,
    },
}

# Create benchmark instance
benchmark = SPDLearnBenchmark(
    config=config,
    X=X,
    y=y,
    meta=meta,
    device=device,
    label_encoder=le,
)

# Run benchmark
results_df = benchmark.run_benchmark(model_configs)

######################################################################
# Results Summary
# ---------------
#
# We display the benchmark results in a formatted table with per-class
# accuracy breakdown.
#

print("\n" + "=" * 80)
print("Benchmark Results Summary")
print("=" * 80)
print(
    results_df[["Model", "Accuracy", "Balanced Accuracy", "LR", "Optimizer"]].to_string(
        index=False
    )
)

print("\n" + "-" * 80)
print("Per-Class Accuracy Breakdown")
print("-" * 80)
for r in benchmark.results:
    print(f"\n{r['model']}:")
    for cls_name, acc in r["per_class_accuracy"].items():
        print(f"  {cls_name}: {acc:.4f}")

######################################################################
# Visualizing Benchmark Results
# -----------------------------
#
# We create visualizations to compare model performance.
#

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Extract data for plotting
models = [r["model"] for r in benchmark.results]
mean_accs = [r["mean_accuracy"] for r in benchmark.results]
std_accs = [r["std_accuracy"] for r in benchmark.results]
mean_bal_accs = [r["mean_balanced_accuracy"] for r in benchmark.results]
std_bal_accs = [r["std_balanced_accuracy"] for r in benchmark.results]

# Plot 1: Accuracy comparison with error bars
ax1 = axes[0]
x_pos = np.arange(len(models))
bar_width = 0.35

bars1 = ax1.bar(
    x_pos - bar_width / 2,
    mean_accs,
    bar_width,
    yerr=std_accs,
    label="Accuracy",
    color="#3498db",
    edgecolor="black",
    capsize=5,
)
bars2 = ax1.bar(
    x_pos + bar_width / 2,
    mean_bal_accs,
    bar_width,
    yerr=std_bal_accs,
    label="Balanced Accuracy",
    color="#2ecc71",
    edgecolor="black",
    capsize=5,
)

ax1.axhline(y=0.25, color="red", linestyle="--", label="Chance (4 classes)", alpha=0.7)
ax1.set_xlabel("Model", fontsize=12)
ax1.set_ylabel("Score", fontsize=12)
ax1.set_title("Model Performance Comparison", fontsize=14)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(models, rotation=15, ha="right")
ax1.set_ylim([0, 1])
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis="y")

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.02,
        f"{height:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

# Plot 2: Radar chart for multi-dimensional comparison
ax2 = axes[1]

# Create ranking-based scores (normalized)
metrics = ["Accuracy", "Bal. Accuracy", "Training Stability"]
n_metrics = len(metrics)

# Calculate stability as inverse of std (lower std = more stable)
max_std = max(std_accs) if max(std_accs) > 0 else 1
stability_scores = [1 - (s / max_std) if max_std > 0 else 1 for s in std_accs]

# Create data for radar chart
angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
angles += angles[:1]  # Complete the loop

colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12"]

for idx, model in enumerate(models):
    values = [mean_accs[idx], mean_bal_accs[idx], stability_scores[idx]]
    values += values[:1]  # Complete the loop

    ax2.plot(
        angles,
        values,
        "o-",
        linewidth=2,
        label=model,
        color=colors[idx % len(colors)],
    )
    ax2.fill(angles, values, alpha=0.1, color=colors[idx % len(colors)])

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(metrics, fontsize=10)
ax2.set_ylim([0, 1])
ax2.set_title("Multi-Metric Comparison", fontsize=14)
ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 1), fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle("SPD Learn Model Benchmark Results", fontsize=16, y=1.02)
plt.show()

######################################################################
# Per-Fold Results Visualization
# ------------------------------
#
# We can also visualize the per-fold results to understand variance.
#

if not config.use_session_split and len(benchmark.results[0]["fold_results"]) > 1:
    fig, ax = plt.subplots(figsize=(10, 5))

    for idx, result in enumerate(benchmark.results):
        fold_accs = [r["accuracy"] for r in result["fold_results"]]
        ax.plot(
            range(1, len(fold_accs) + 1),
            fold_accs,
            "o-",
            label=result["model"],
            linewidth=2,
            markersize=8,
        )

    ax.set_xlabel("Fold", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Per-Fold Accuracy", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.show()

######################################################################
# Using Hydra from Command Line
# -----------------------------
#
# In a production environment, you would use Hydra's command-line
# interface to manage configurations. Here's how you would structure
# your project:
#
# .. code-block:: text
#
#    project/
#    +-- config/
#    |   +-- config.yaml           # Main config
#    |   +-- training/
#    |   |   +-- default.yaml
#    |   |   +-- fast.yaml
#    |   +-- data/
#    |   |   +-- bnci2014_001.yaml
#    |   |   +-- bnci2014_004.yaml
#    |   +-- model/
#    |       +-- spdnet.yaml
#    |       +-- tsmnet.yaml
#    |       +-- eegspdnet.yaml
#    +-- benchmark.py              # Main script
#
# You would then run experiments like:
#
# .. code-block:: bash
#
#    # Default configuration
#    python benchmark.py
#
#    # Override training parameters
#    python benchmark.py training.max_epochs=200 training.learning_rate=1e-4
#
#    # Use different dataset
#    python benchmark.py data=bnci2014_004
#
#    # Run multiple experiments with multirun
#    python benchmark.py -m training.learning_rate=1e-3,1e-4,1e-5
#
# Example Hydra main script:
#
# .. code-block:: python
#
#    import hydra
#    from omegaconf import DictConfig
#
#    @hydra.main(version_base=None, config_path="config", config_name="config")
#    def main(cfg: DictConfig) -> float:
#        # Your benchmark code here
#        benchmark = SPDLearnBenchmark(cfg, X, y, meta, device)
#        results = benchmark.run_benchmark()
#        return results["mean_accuracy"].mean()
#
#    if __name__ == "__main__":
#        main()
#

######################################################################
# Advanced Configuration: Filter Bank Models
# ------------------------------------------
#
# For filter bank models like TensorCSPNet, we need to use
# FilterBankMotorImagery paradigm. Here's how to configure this:
#

print("\n" + "=" * 60)
print("Filter Bank Configuration Example")
print("=" * 60)

# Define filter bank configuration
filterbank_config = OmegaConf.create(
    {
        "paradigm": "FilterBankMotorImagery",
        "filters": [
            [4, 8],
            [8, 12],
            [12, 16],
            [16, 20],
            [20, 24],
            [24, 28],
            [28, 32],
            [32, 36],
            [36, 40],
        ],
        "model": {
            "name": "TensorCSPNet",
            "n_patches": 4,
            "n_freqs": 9,
            "use_mlp": False,
            "tcn_channels": 16,
        },
    }
)

print("Filter Bank Configuration:")
print(OmegaConf.to_yaml(filterbank_config))

# Note: To actually run TensorCSPNet, you would use:
#
# from moabb.paradigms import FilterBankMotorImagery
#
# fb_paradigm = FilterBankMotorImagery(
#     n_classes=4,
#     filters=filterbank_config.filters,
# )
# X_fb, labels_fb, meta_fb = fb_paradigm.get_data(...)
#
# Then create the model with:
# model = create_model("TensorCSPNet", n_chans=n_chans, n_outputs=n_outputs,
#                      n_freqs=len(filterbank_config.filters))

######################################################################
# Extending the Benchmark with New Models
# ---------------------------------------
#
# To add a new model to this benchmark framework, follow these steps:
#
# 1. **Define a configuration dataclass** for your model:
#
# .. code-block:: python
#
#    @dataclass
#    class MyNewModelConfig(ModelConfig):
#        """Configuration for MyNewModel."""
#        name: str = "MyNewModel"
#        param1: int = 10
#        param2: float = 0.1
#
# 2. **Add the model to create_model()** factory function:
#
# .. code-block:: python
#
#    def create_model(model_name, n_chans, n_outputs, **kwargs):
#        # ... existing models ...
#        elif model_name == "MyNewModel":
#            return MyNewModel(
#                n_chans=n_chans,
#                n_outputs=n_outputs,
#                param1=kwargs.get("param1", 10),
#                param2=kwargs.get("param2", 0.1),
#            )
#
# 3. **Add training overrides** if your model needs special training:
#
# .. code-block:: python
#
#    def get_default_model_training_overrides():
#        return {
#            # ... existing models ...
#            "MyNewModel": {
#                "learning_rate": 5e-4,
#                "max_epochs": 120,
#                "optimizer": "AdamW",
#            },
#        }
#
# 4. **Add the model to the experiment configuration**:
#
# .. code-block:: python
#
#    config = OmegaConf.structured(
#        ExperimentConfig(
#            models=["SPDNet", "TSMNet", "EEGSPDNet", "MyNewModel"],
#            # ...
#        )
#    )
#
# 5. **Provide model architecture parameters**:
#
# .. code-block:: python
#
#    model_configs = {
#        # ... existing models ...
#        "MyNewModel": {
#            "param1": 20,
#            "param2": 0.05,
#        },
#    }
#

######################################################################
# Summary
# -------
#
# In this tutorial, we demonstrated how to:
#
# 1. **Define structured configurations** using dataclasses and OmegaConf
# 2. **Create a model factory** for easy model instantiation
# 3. **Build a benchmarking pipeline** with proper cross-validation
# 4. **Use model-specific training configurations** for optimal performance
# 5. **Implement early stopping and learning rate scheduling**
# 6. **Evaluate multiple SPD Learn models** (SPDNet, TSMNet, EEGSPDNet)
# 7. **Visualize and compare results** across models with per-class breakdown
# 8. **Structure a Hydra-based project** for production use
#
# Key takeaways:
#
# - **Hydra and OmegaConf** provide powerful configuration management
# - **Structured configs** enable type safety and easy modification
# - **Model-specific training** is essential - TSMNet/EEGSPDNet need
#   lower learning rates (1e-4) and more epochs than SPDNet
# - **Proper cross-validation** is essential for reliable benchmarks
# - **SPD Learn models** :cite:p:`wilson2025deep` offer different trade-offs for EEG classification
#
# For production benchmarks, consider:
#
# - Using more subjects and epochs
# - Implementing hyperparameter tuning with Optuna
# - Adding more evaluation metrics (F1, Cohen's kappa)
# - Using Hydra's multirun for hyperparameter sweeps
# - Logging with MLflow or Weights & Biases
#
