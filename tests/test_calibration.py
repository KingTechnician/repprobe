"""Tests for calibration module."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from repprobe import RepDataset, LabelSchema
from repprobe.calibration import multiclass_fisher_score, calibrate_best_layer


class TestMulticlassFisherScore:
    def test_returns_positive_for_separable(self):
        """Separable clusters → positive Fisher score."""
        acts = torch.cat([
            torch.randn(50, 16) + torch.tensor([5.0] + [0.0] * 15),
            torch.randn(50, 16) + torch.tensor([-5.0] + [0.0] * 15),
        ])
        labels = torch.cat([torch.zeros(50), torch.ones(50)]).long()
        score = multiclass_fisher_score(acts, labels, k=2)
        assert score > 1.0

    def test_returns_near_zero_for_random(self):
        """Random noise → near-zero score."""
        torch.manual_seed(0)
        acts = torch.randn(100, 16)
        labels = torch.randint(0, 3, (100,))
        score = multiclass_fisher_score(acts, labels, k=3)
        assert score < 10.0  # not explosively high

    def test_multiclass_five(self):
        k = 5
        per_class = 40
        acts = []
        for c in range(k):
            mean = np.zeros(16)
            mean[c] = 10.0
            acts.append(torch.tensor(
                np.random.default_rng(c).normal(loc=mean, scale=0.1, size=(per_class, 16)),
                dtype=torch.float32,
            ))
        acts_t = torch.cat(acts)
        labels_t = torch.tensor([c for c in range(k) for _ in range(per_class)], dtype=torch.long)
        score = multiclass_fisher_score(acts_t, labels_t, k=k)
        assert score > 10.0

    def test_handles_empty_class(self):
        """One empty class should not crash."""
        acts = torch.randn(40, 8)
        labels = torch.tensor([0] * 20 + [1] * 20)
        # Ask for k=3 but only 2 classes present
        score = multiclass_fisher_score(acts, labels, k=3)
        assert isinstance(score, float)


class TestCalibrateBestLayer:
    def test_returns_best_layer(self, multiclass_rep_dataset):
        best, scores = calibrate_best_layer(multiclass_rep_dataset, verbose=False)
        assert best in multiclass_rep_dataset.activation_layers
        assert isinstance(scores, dict)
        assert len(scores) == len(multiclass_rep_dataset.activation_layers)

    def test_subset_of_layers(self, multiclass_rep_dataset):
        best, scores = calibrate_best_layer(
            multiclass_rep_dataset, layers=[8, 12], verbose=False
        )
        assert best in [8, 12]
        assert set(scores.keys()) == {8, 12}

    def test_no_act_cols_raises(self):
        from datasets import Dataset  # noqa: PLC0415
        hf = Dataset.from_dict({"text": ["x", "y"], "label": [0, 1]})
        ds = RepDataset(hf_dataset=hf)
        with pytest.raises(ValueError, match="no activation columns"):
            calibrate_best_layer(ds, verbose=False)

    def test_missing_layer_raises(self, multiclass_rep_dataset):
        with pytest.raises(ValueError, match="not found in dataset"):
            calibrate_best_layer(multiclass_rep_dataset, layers=[99], verbose=False)

    def test_best_layer_has_highest_score(self, multiclass_rep_dataset):
        best, scores = calibrate_best_layer(multiclass_rep_dataset, verbose=False)
        assert scores[best] == max(scores.values())
