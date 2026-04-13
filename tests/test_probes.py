"""Tests for all probe implementations and ProbeFactory."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from repprobe import LabelSchema, LRProbe, LDAProbe, MeanDiffProbe
from repprobe.probes import CCSProbe, ProbeFactory
from repprobe.probes.base import AbstractProbe


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_separable(n: int, k: int, d: int = 32, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate activations with well-separated clusters per class."""
    rng = np.random.default_rng(seed)
    acts, labels = [], []
    per_class = n // k
    for c in range(k):
        mean = np.zeros(d)
        mean[c % d] = 5.0
        a = rng.normal(loc=mean, scale=0.3, size=(per_class, d))
        acts.append(a)
        labels.extend([c] * per_class)
    acts_t = torch.tensor(np.vstack(acts), dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    return acts_t, labels_t


# ---------------------------------------------------------------------------
# LRProbe
# ---------------------------------------------------------------------------


class TestLRProbe:
    def test_binary(self, acts_and_labels_binary):
        acts, labels = acts_and_labels_binary
        schema = LabelSchema.binary()
        probe = LRProbe.from_data(acts, labels, schema)
        assert isinstance(probe, LRProbe)

    def test_multiclass(self, acts_and_labels_multiclass):
        acts, labels = acts_and_labels_multiclass
        schema = LabelSchema(names=[str(i) for i in range(5)])
        probe = LRProbe.from_data(acts, labels, schema)
        preds = probe.pred(acts)
        assert preds.shape == (len(acts),)
        assert preds.dtype == torch.long

    def test_predict_proba_shape(self, acts_and_labels_multiclass):
        acts, labels = acts_and_labels_multiclass
        schema = LabelSchema(names=[str(i) for i in range(5)])
        probe = LRProbe.from_data(acts, labels, schema)
        proba = probe.predict_proba(acts)
        assert proba.shape == (len(acts), 5)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_high_accuracy_separable(self):
        acts, labels = _make_separable(200, k=3, d=32)
        schema = LabelSchema(names=["a", "b", "c"])
        probe = LRProbe.from_data(acts, labels, schema)
        result = probe.evaluate(acts, labels)
        assert result["accuracy"] > 0.90

    def test_evaluate_keys(self, acts_and_labels_multiclass):
        acts, labels = acts_and_labels_multiclass
        schema = LabelSchema(names=[str(i) for i in range(5)])
        probe = LRProbe.from_data(acts, labels, schema)
        result = probe.evaluate(acts, labels)
        assert "accuracy" in result
        assert "macro_f1" in result
        assert "per_class_accuracy" in result
        assert "confusion_matrix" in result

    def test_confusion_matrix_shape(self, acts_and_labels_multiclass):
        acts, labels = acts_and_labels_multiclass
        schema = LabelSchema(names=[str(i) for i in range(5)])
        probe = LRProbe.from_data(acts, labels, schema)
        cm = probe.evaluate(acts, labels)["confusion_matrix"]
        assert len(cm) == 5
        assert all(len(row) == 5 for row in cm)


# ---------------------------------------------------------------------------
# LDAProbe
# ---------------------------------------------------------------------------


class TestLDAProbe:
    def test_from_data_returns_instance(self, acts_and_labels_multiclass):
        acts, labels = acts_and_labels_multiclass
        schema = LabelSchema(names=[str(i) for i in range(5)])
        probe = LDAProbe.from_data(acts, labels, schema)
        assert isinstance(probe, LDAProbe)

    def test_projections_shape(self, acts_and_labels_multiclass):
        acts, labels = acts_and_labels_multiclass
        schema = LabelSchema(names=[str(i) for i in range(5)])
        probe = LDAProbe.from_data(acts, labels, schema)
        # [hidden_dim, K-1]
        assert probe.projections.shape == (acts.shape[1], 4)

    def test_pred_shape(self, acts_and_labels_multiclass):
        acts, labels = acts_and_labels_multiclass
        schema = LabelSchema(names=[str(i) for i in range(5)])
        probe = LDAProbe.from_data(acts, labels, schema)
        preds = probe.pred(acts)
        assert preds.shape == (len(acts),)

    def test_high_accuracy_separable(self):
        acts, labels = _make_separable(300, k=5, d=32)
        schema = LabelSchema(names=["a", "b", "c", "d", "e"])
        probe = LDAProbe.from_data(acts, labels, schema)
        result = probe.evaluate(acts, labels)
        assert result["accuracy"] > 0.85

    def test_binary_single_direction(self, acts_and_labels_binary):
        acts, labels = acts_and_labels_binary
        schema = LabelSchema.binary()
        probe = LDAProbe.from_data(acts, labels, schema)
        # K=2 → K-1=1 direction
        assert probe.projections.shape == (acts.shape[1], 1)

    def test_predict_proba_sums_to_one(self, acts_and_labels_multiclass):
        acts, labels = acts_and_labels_multiclass
        schema = LabelSchema(names=[str(i) for i in range(5)])
        probe = LDAProbe.from_data(acts, labels, schema)
        proba = probe.predict_proba(acts)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# MeanDiffProbe
# ---------------------------------------------------------------------------


class TestMeanDiffProbe:
    def test_from_data(self, acts_and_labels_multiclass):
        acts, labels = acts_and_labels_multiclass
        schema = LabelSchema(names=[str(i) for i in range(5)])
        probe = MeanDiffProbe.from_data(acts, labels, schema)
        assert isinstance(probe, MeanDiffProbe)

    def test_directions_shape(self, acts_and_labels_multiclass):
        acts, labels = acts_and_labels_multiclass
        schema = LabelSchema(names=[str(i) for i in range(5)])
        probe = MeanDiffProbe.from_data(acts, labels, schema)
        # [hidden_dim, K]
        assert probe.directions.shape == (acts.shape[1], 5)

    def test_pred_shape_and_dtype(self, acts_and_labels_binary):
        acts, labels = acts_and_labels_binary
        schema = LabelSchema.binary()
        probe = MeanDiffProbe.from_data(acts, labels, schema)
        preds = probe.pred(acts)
        assert preds.dtype == torch.long
        assert preds.shape == (len(acts),)

    def test_high_accuracy_separable(self):
        acts, labels = _make_separable(200, k=3, d=32)
        schema = LabelSchema(names=["x", "y", "z"])
        probe = MeanDiffProbe.from_data(acts, labels, schema)
        result = probe.evaluate(acts, labels)
        assert result["accuracy"] > 0.85


# ---------------------------------------------------------------------------
# CCSProbe
# ---------------------------------------------------------------------------


class TestCCSProbe:
    def _make_binary_paired(self, n=100, d=16, seed=7):
        rng = np.random.default_rng(seed)
        labels = rng.integers(0, 2, size=n)
        acts = torch.tensor(
            rng.normal(size=(n, d), loc=np.where(labels[:, None] == 1, 2.0, -2.0)),
            dtype=torch.float32,
        )
        neg_acts = torch.tensor(
            rng.normal(size=(n, d), loc=np.where(labels[:, None] == 1, -2.0, 2.0)),
            dtype=torch.float32,
        )
        return acts, neg_acts, torch.tensor(labels, dtype=torch.long)

    def test_from_data_with_neg_acts(self):
        acts, neg_acts, labels = self._make_binary_paired()
        schema = LabelSchema.binary(has_polarity=True)
        probe = CCSProbe.from_data(acts, labels, schema, neg_acts=neg_acts, epochs=50)
        assert isinstance(probe, CCSProbe)

    def test_binary_only_guard(self):
        acts = torch.randn(10, 16)
        labels = torch.randint(0, 3, (10,))
        schema = LabelSchema(names=["a", "b", "c"])
        with pytest.raises(ValueError, match="binary"):
            CCSProbe.from_data(acts, labels, schema, neg_acts=acts)

    def test_neg_acts_required(self):
        acts = torch.randn(10, 16)
        labels = torch.randint(0, 2, (10,))
        schema = LabelSchema.binary(has_polarity=True)
        with pytest.raises(ValueError, match="neg_acts"):
            CCSProbe.from_data(acts, labels, schema)

    def test_direction_property(self):
        acts, neg_acts, labels = self._make_binary_paired()
        schema = LabelSchema.binary(has_polarity=True)
        probe = CCSProbe.from_data(acts, labels, schema, neg_acts=neg_acts, epochs=50)
        assert probe.direction.shape == (16,)

    def test_predict_proba_shape(self):
        acts, neg_acts, labels = self._make_binary_paired(n=50)
        schema = LabelSchema.binary(has_polarity=True)
        probe = CCSProbe.from_data(acts, labels, schema, neg_acts=neg_acts, epochs=50)
        proba = probe.predict_proba(acts)
        assert proba.shape == (50, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# ProbeFactory
# ---------------------------------------------------------------------------


class TestProbeFactory:
    def test_available_probes_multiclass(self):
        schema = LabelSchema(names=["a", "b", "c"])
        avail = ProbeFactory.available_probes(schema)
        assert LRProbe in avail
        assert LDAProbe in avail
        assert MeanDiffProbe in avail
        assert CCSProbe not in avail

    def test_available_probes_binary_no_polarity(self):
        schema = LabelSchema.binary(has_polarity=False)
        avail = ProbeFactory.available_probes(schema)
        assert CCSProbe not in avail

    def test_available_probes_binary_with_polarity(self):
        schema = LabelSchema.binary(has_polarity=True)
        avail = ProbeFactory.available_probes(schema)
        assert CCSProbe in avail

    def test_create_lr(self, acts_and_labels_multiclass):
        acts, labels = acts_and_labels_multiclass
        schema = LabelSchema(names=[str(i) for i in range(5)])
        probe = ProbeFactory.create("lr", acts, labels, schema)
        assert isinstance(probe, LRProbe)

    def test_create_lda(self, acts_and_labels_multiclass):
        acts, labels = acts_and_labels_multiclass
        schema = LabelSchema(names=[str(i) for i in range(5)])
        probe = ProbeFactory.create("lda", acts, labels, schema)
        assert isinstance(probe, LDAProbe)

    def test_create_mean_diff(self, acts_and_labels_multiclass):
        acts, labels = acts_and_labels_multiclass
        schema = LabelSchema(names=[str(i) for i in range(5)])
        probe = ProbeFactory.create("mean_diff", acts, labels, schema)
        assert isinstance(probe, MeanDiffProbe)

    def test_create_unknown_raises(self, acts_and_labels_multiclass):
        acts, labels = acts_and_labels_multiclass
        schema = LabelSchema(names=["a", "b"])
        with pytest.raises(ValueError, match="Unknown probe type"):
            ProbeFactory.create("magic_probe", acts, labels, schema)

    def test_create_ccs_on_multiclass_raises(self, acts_and_labels_multiclass):
        acts, labels = acts_and_labels_multiclass
        schema = LabelSchema(names=[str(i) for i in range(5)])
        with pytest.raises(ValueError, match="not valid for the given schema"):
            ProbeFactory.create("ccs", acts, labels, schema)

    def test_create_case_insensitive(self, acts_and_labels_binary):
        acts, labels = acts_and_labels_binary
        schema = LabelSchema.binary()
        probe = ProbeFactory.create("LR", acts, labels, schema)
        assert isinstance(probe, LRProbe)

    def test_abstract_probe_interface(self):
        """All probes must satisfy the AbstractProbe interface."""
        for probe_cls in [LRProbe, LDAProbe, MeanDiffProbe]:
            assert issubclass(probe_cls, AbstractProbe)
