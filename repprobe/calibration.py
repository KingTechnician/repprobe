"""
Layer calibration: select the best extraction layer via a generalized
Fisher separability criterion.

Binary Fisher ratio (TiU):
    F = ||μ_T - μ_F||² / (σ²_T + σ²_F)

Multiclass extension used here:
    F_multi = trace(S_w⁻¹ · S_b)

where S_b and S_w are the between-class and within-class scatter matrices.
When K=2 this reduces to the binary Fisher ratio.
"""
from __future__ import annotations

from typing import Optional

import torch

from .data import RepDataset


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------


def multiclass_fisher_score(
    acts: torch.Tensor,
    labels: torch.Tensor,
    k: int,
) -> float:
    """
    Compute trace(S_w⁻¹ · S_b) for the given activations and labels.

    For efficiency with large hidden dimensions, the computation is performed
    in 32-bit float.

    Parameters
    ----------
    acts : Tensor [N, hidden_dim]
    labels : Tensor [N]   (long, 0-indexed)
    k : int
        Number of classes.

    Returns
    -------
    float
        Fisher separability score (higher → more separable).
    """
    acts = acts.float()
    n, d = acts.shape
    global_mean = acts.mean(dim=0)                  # [d]

    S_b = torch.zeros(d, d, dtype=torch.float32)
    S_w = torch.zeros(d, d, dtype=torch.float32)

    for c in range(k):
        mask = labels == c
        acts_c = acts[mask]
        n_c = acts_c.size(0)
        if n_c == 0:
            continue
        mu_c = acts_c.mean(dim=0)                   # [d]
        diff = (mu_c - global_mean).unsqueeze(1)    # [d, 1]
        S_b += n_c * (diff @ diff.T)
        centered = acts_c - mu_c                    # [n_c, d]
        S_w += centered.T @ centered                # [d, d]

    # Regularise for numerical stability
    S_w += 1e-6 * torch.eye(d, dtype=torch.float32)

    try:
        score = torch.trace(torch.linalg.solve(S_w, S_b)).item()
    except RuntimeError:
        # Fallback: use pseudo-inverse
        score = torch.trace(torch.linalg.pinv(S_w) @ S_b).item()

    return score


# ---------------------------------------------------------------------------
# Calibration over a set of layers
# ---------------------------------------------------------------------------


def calibrate_best_layer(
    dataset: RepDataset,
    layers: Optional[list[int]] = None,
    verbose: bool = True,
) -> tuple[int, dict[int, float]]:
    """
    Find the extraction layer with the highest multiclass Fisher score.

    The *dataset* must already have activation columns (``layer_{N}_acts``).

    Parameters
    ----------
    dataset : RepDataset
        Dataset with pre-computed activation columns.
    layers : list[int] | None
        Layer indices to evaluate.  If ``None``, all activation columns
        in the dataset are evaluated.
    verbose : bool
        Print per-layer scores.

    Returns
    -------
    best_layer : int
    scores : dict[int, float]
    """
    available = dataset.activation_layers
    if not available:
        raise ValueError(
            "The dataset has no activation columns. "
            "Run extraction first to add 'layer_{N}_acts' columns."
        )

    if layers is None:
        layers = available
    else:
        missing = [l for l in layers if l not in available]  # noqa: E741
        if missing:
            raise ValueError(
                f"Layers {missing} not found in dataset. "
                f"Available: {available}"
            )

    labels = dataset.get_labels()
    schema = dataset.schema
    scores: dict[int, float] = {}

    for layer in layers:
        acts = dataset.get_acts(layer)
        score = multiclass_fisher_score(acts, labels, schema.k)
        scores[layer] = score
        if verbose:
            print(f"  Layer {layer:3d}: Fisher score = {score:.6e}")

    best_layer = max(scores, key=scores.get)
    if verbose:
        print(f"\nBest layer: {best_layer} (score={scores[best_layer]:.6e})")
    return best_layer, scores
