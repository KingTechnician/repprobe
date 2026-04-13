"""Tests for evaluation utilities."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from repprobe import LabelSchema, LRProbe
from repprobe.eval import evaluate_probe, compare_probes, print_results


def _make_probe_and_data(k=3, n=150, d=16, seed=1):
    rng = np.random.default_rng(seed)
    labels = torch.tensor(rng.integers(0, k, size=n), dtype=torch.long)
    acts = torch.zeros(n, d)
    for i, l in enumerate(labels):
        acts[i, l.item()] = 5.0
    acts += torch.tensor(rng.normal(scale=0.1, size=(n, d)), dtype=torch.float32)
    schema = LabelSchema(names=[str(i) for i in range(k)])
    probe = LRProbe.from_data(acts, labels, schema)
    return probe, acts, labels


class TestEvaluateProbe:
    def test_returns_dict_with_expected_keys(self):
        probe, acts, labels = _make_probe_and_data()
        result = evaluate_probe(probe, acts, labels)
        assert "accuracy" in result
        assert "macro_f1" in result
        assert "per_class_accuracy" in result
        assert "confusion_matrix" in result

    def test_accuracy_in_range(self):
        probe, acts, labels = _make_probe_and_data()
        result = evaluate_probe(probe, acts, labels)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_macro_f1_in_range(self):
        probe, acts, labels = _make_probe_and_data()
        result = evaluate_probe(probe, acts, labels)
        assert 0.0 <= result["macro_f1"] <= 1.0

    def test_per_class_length(self):
        probe, acts, labels = _make_probe_and_data(k=4)
        result = evaluate_probe(probe, acts, labels)
        assert len(result["per_class_accuracy"]) == 4

    def test_perfect_accuracy_separable(self):
        probe, acts, labels = _make_probe_and_data(k=3, n=150, d=16)
        result = evaluate_probe(probe, acts, labels)
        assert result["accuracy"] > 0.95


class TestCompareProbes:
    def test_returns_dict_per_probe(self):
        from repprobe import LDAProbe, MeanDiffProbe  # noqa: PLC0415

        probe_lr, acts, labels = _make_probe_and_data(k=3)
        schema = LabelSchema(names=["a", "b", "c"])
        probe_lda = LDAProbe.from_data(acts, labels, schema)
        probe_md = MeanDiffProbe.from_data(acts, labels, schema)

        results = compare_probes(
            {"lr": probe_lr, "lda": probe_lda, "mean_diff": probe_md},
            acts,
            labels,
        )
        assert set(results.keys()) == {"lr", "lda", "mean_diff"}
        for r in results.values():
            assert "accuracy" in r


class TestPrintResults:
    def test_does_not_crash(self, capsys):
        probe, acts, labels = _make_probe_and_data(k=2)
        result = evaluate_probe(probe, acts, labels)
        print_results(result, title="Test probe")
        captured = capsys.readouterr()
        assert "Accuracy" in captured.out
        assert "Macro F1" in captured.out
