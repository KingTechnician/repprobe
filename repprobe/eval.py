"""
Evaluation utilities for representation probes.

Provides both single-probe evaluation and multi-probe comparison helpers.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from .data import RepDataset
    from .probes.base import AbstractProbe
    from .schema import LabelSchema


# ---------------------------------------------------------------------------
# Per-probe evaluation
# ---------------------------------------------------------------------------


def evaluate_probe(
    probe: "AbstractProbe",
    acts: torch.Tensor,
    labels: torch.Tensor,
) -> dict:
    """
    Run standard evaluation metrics for one probe.

    Returns a dict with:
    - ``accuracy``            (float)
    - ``macro_f1``            (float)
    - ``per_class_accuracy``  (dict[class_name, float])
    - ``confusion_matrix``    (list[list[int]])
    """
    return probe.evaluate(acts, labels)


def evaluate_generalization(
    probe: "AbstractProbe",
    dataset: "RepDataset",
    layer: int,
    split_column: str,
    train_values: list,
    test_values: list,
) -> dict:
    """
    Evaluate generalization by training on some *split_column* values and
    testing on held-out ones.

    The probe is NOT re-trained here — it is assumed to be already fitted on
    the training split.  This function only evaluates on the test split.

    Parameters
    ----------
    probe : AbstractProbe
        An already-trained probe.
    dataset : RepDataset
        Full dataset with activation columns.
    layer : int
        Which activation layer to use.
    split_column : str
        Column to split on (e.g. ``"objective"``).
    train_values : list
        Values that belong to the training partition (for reference reporting).
    test_values : list
        Values to evaluate on.

    Returns
    -------
    dict
        Standard metrics on the test split.
    """
    _, test_ds = dataset.split_by_column(split_column, train_values, test_values)
    test_acts = test_ds.get_acts(layer)
    test_labels = test_ds.get_labels()
    result = probe.evaluate(test_acts, test_labels)
    result["split_column"] = split_column
    result["test_values"] = test_values
    return result


# ---------------------------------------------------------------------------
# Multi-probe comparison
# ---------------------------------------------------------------------------


def compare_probes(
    probes: dict[str, "AbstractProbe"],
    acts: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, dict]:
    """
    Evaluate multiple probes on the same test data.

    Parameters
    ----------
    probes : dict[str, AbstractProbe]
        Mapping of probe name → probe instance.
    acts : Tensor [N, hidden_dim]
    labels : Tensor [N]

    Returns
    -------
    dict[str, dict]
        Mapping of probe name → evaluation dict.
    """
    return {name: probe.evaluate(acts, labels) for name, probe in probes.items()}


def print_results(results: dict, title: str = "Evaluation") -> None:
    """Pretty-print evaluation results for a single probe."""
    print(f"\n{'─' * 40}")
    print(f"  {title}")
    print(f"{'─' * 40}")
    print(f"  Accuracy : {results['accuracy']:.4f}")
    print(f"  Macro F1 : {results['macro_f1']:.4f}")
    print("  Per-class accuracy:")
    for cls_name, acc in results["per_class_accuracy"].items():
        print(f"    {cls_name:20s}: {acc:.4f}" if not np.isnan(acc) else f"    {cls_name:20s}: N/A")
    print(f"{'─' * 40}")
