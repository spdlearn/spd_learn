# Authors: Bruno Aristimunha
#
# License: BSD-3

import mne
import numpy as np
import pytest
import torch

from braindecode.classifier import EEGClassifier
from braindecode.util import set_random_seeds
from mne.io import concatenate_raws
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from torch.utils.data import TensorDataset

import spd_learn

from spd_learn.models import __all__ as model_list
from spd_learn.models import __filter_bank_models__ as filter_bank_models


mne.set_log_level("ERROR")


@pytest.fixture()
def raw_data():
    subject_id = 1
    # 5, 6, 7, 10, 13, 14 are codes for executed and imagined hands/feet
    event_codes = [5, 6, 9, 10, 13, 14]

    # This will download the files if you don't have them yet,
    # and then return the paths to the files.
    physionet_paths = mne.datasets.eegbci.load_data(
        subject_id, event_codes, update_path=False
    )

    # Load each of the files
    parts = [
        mne.io.read_raw_edf(path, preload=True, stim_channel="auto", verbose="WARNING")
        for path in physionet_paths
    ]

    # Concatenate them
    raw = concatenate_raws(parts)

    # Find the events in this dataset
    events, _ = mne.events_from_annotations(raw)

    # Use only EEG channels
    eeg_channel_inds = mne.pick_types(
        raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
    )

    return raw, events, eeg_channel_inds


@pytest.fixture()
def epoch_data(raw_data):
    raw, events, eeg_channel_inds = raw_data
    # Extract trials, only using EEG channels
    epoched = mne.Epochs(
        raw,
        events,
        dict(hands=2, feet=3),
        tmin=1,
        tmax=4.1,
        proj=False,
        picks=eeg_channel_inds,
        baseline=None,
        preload=True,
    )

    # Convert data from volt to millivolt
    # Pytorch expects float32 for input and int64 for labels.
    X = (epoched.get_data() * 1e6).astype(np.float32)
    y = (epoched.events[:, 2] - 2).astype(np.int64)  # 2,3 -> 0,1
    return X, y


def create_filterbank(raw, freq_bands):
    filterbank = []
    for _, (l_freq, h_freq) in freq_bands.items():
        raw_filtered = raw.copy().filter(
            l_freq,
            h_freq,
            method="fir",  # Finite Impulse Response (zero-phase)
            fir_design="firwin",
            verbose=False,
        )
        filterbank.append(raw_filtered.get_data() * 1e6)
    return np.stack(filterbank, axis=1)


@pytest.fixture()
def filterbank_data(raw_data):
    raw, events, eeg_channel_inds = raw_data
    # Extract trials, only using EEG channels
    epoched = mne.Epochs(
        raw,
        events,
        dict(hands=2, feet=3),
        tmin=1,
        tmax=4.1,
        proj=False,
        picks=eeg_channel_inds,
        baseline=None,
        preload=True,
    )

    # Define frequency bands (customize as needed)
    freq_bands = {f"{i}": (freq, freq + 4) for i, freq in enumerate(range(4, 40, 4))}
    # shape: (90, 9, 64, 497)
    filterbank = create_filterbank(epoched, freq_bands)

    # Convert data from volt to millivolt
    # Pytorch expects float32 for input and int64 for labels.
    filter_X = filterbank.astype(np.float32)
    y = (epoched.events[:, 2] - 2).astype(np.int64)

    return filter_X, y


@pytest.mark.parametrize("model_name", model_list)
def test_correctness_spd_learn(epoch_data, filterbank_data, model_name):
    seed = 42

    set_random_seeds(seed=seed, cuda=False)

    if model_name in filter_bank_models:
        X, y = filterbank_data
    else:
        X, y = epoch_data

    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()

    n_chans = 64
    n_times = 450
    kwargs = dict()

    lr = 0.01

    if model_name == "TensorCSPNet":
        # Use proper dimensions: input -> intermediate -> output
        # dims should reduce from n_chans (64) to a smaller dimension
        kwargs = dict(n_patches=2, dims=(64, 32, 32, 16))
        lr = 1e-3
    elif model_name == "EEGSPDNet":
        lr = 1e-3
    elif model_name == "Green":
        kwargs = dict(bi_out=[16])

    model_class = getattr(spd_learn.models, model_name)

    model = model_class(
        n_outputs=2,
        n_chans=n_chans,
        **kwargs,
    )

    X_train = X[:60, ..., :n_times]
    y_train = y[:60]

    X_valid = X[60:90, ..., :n_times]
    y_valid = y[60:90]

    train_set = TensorDataset(X_train, y_train)
    valid_set = TensorDataset(X_valid, y_valid)
    # I really dont know why this is working with create_from_X_y with 4 epochs,
    # but directly numpy need more epoch...
    # I will try to fix it later
    n_epochs = 20

    clf = EEGClassifier(
        model,
        criterion=torch.nn.CrossEntropyLoss,
        train_split=predefined_split(valid_set),
        batch_size=16,
        classes=[0, 1],
        optimizer=torch.optim.Adam,
        optimizer__lr=lr,
        max_epochs=n_epochs,
        callbacks=[
            "accuracy",
            ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=n_epochs - 1)),
        ],
        verbose=1,
    )
    clf.fit(train_set, y=None)

    train_loss = clf.history[:, "train_loss"]
    valid_loss = clf.history[:, "valid_loss"]
    valid_accuracy = clf.history[:, "valid_accuracy"]
    train_accuracy = clf.history[:, "train_accuracy"]

    assert valid_loss[0] > valid_loss[-1]
    assert train_loss[0] > train_loss[-1]
    assert train_accuracy[0] < train_accuracy[-1]
    assert valid_accuracy[0] < valid_accuracy[-1]
