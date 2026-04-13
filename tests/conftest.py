"""
Shared pytest fixtures for RepProbe tests.

All fixtures generate synthetic data in memory — no model downloads,
no network access required.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from datasets import Dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_dataset(
    n: int = 200,
    k: int = 3,
    hidden_dim: int = 32,
    *,
    include_polarity: bool = False,
    include_objective: bool = False,
    seed: int = 0,
) -> dict:
    """Return a dict suitable for ``Dataset.from_dict``."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, k, size=n).tolist()
    texts = [f"Synthetic statement {i} for class {labels[i]}" for i in range(n)]
    data = {"text": texts, "label": labels}

    # Add activation columns (simulating extraction output)
    for layer in (8, 12, 16):
        # Cluster each class around a different mean
        acts = []
        for lbl in labels:
            mean = np.zeros(hidden_dim)
            mean[lbl] = 2.0         # separable cluster per class
            acts.append((mean + rng.normal(scale=0.3, size=hidden_dim)).tolist())
        data[f"layer_{layer}_acts"] = acts

    if include_polarity:
        data["polarity"] = [1 if l == 1 else -1 for l in labels]
    if include_objective:
        objectives = [f"obj_{i % 5}" for i in range(n)]
        data["objective"] = objectives

    return data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def binary_hf_dataset() -> Dataset:
    """Binary (k=2) dataset with polarity column."""
    raw = _make_synthetic_dataset(n=200, k=2, hidden_dim=32, include_polarity=True)
    return Dataset.from_dict(raw)


@pytest.fixture
def multiclass_hf_dataset() -> Dataset:
    """Multiclass (k=5) dataset."""
    raw = _make_synthetic_dataset(n=300, k=5, hidden_dim=32)
    return Dataset.from_dict(raw)


@pytest.fixture
def multiclass_with_objective_hf_dataset() -> Dataset:
    raw = _make_synthetic_dataset(n=300, k=5, hidden_dim=32, include_objective=True)
    return Dataset.from_dict(raw)


@pytest.fixture
def binary_rep_dataset(binary_hf_dataset):
    from repprobe import RepDataset  # noqa: PLC0415
    return RepDataset(hf_dataset=binary_hf_dataset)


@pytest.fixture
def multiclass_rep_dataset(multiclass_hf_dataset):
    from repprobe import RepDataset  # noqa: PLC0415
    return RepDataset(hf_dataset=multiclass_hf_dataset)


@pytest.fixture
def acts_and_labels_binary(binary_hf_dataset):
    from repprobe import RepDataset  # noqa: PLC0415
    ds = RepDataset(hf_dataset=binary_hf_dataset)
    return ds.get_acts(12), ds.get_labels()


@pytest.fixture
def acts_and_labels_multiclass(multiclass_hf_dataset):
    from repprobe import RepDataset  # noqa: PLC0415
    ds = RepDataset(hf_dataset=multiclass_hf_dataset)
    return ds.get_acts(12), ds.get_labels()
