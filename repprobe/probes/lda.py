"""
LDAProbe: Generalized TTPD via Fisher Linear Discriminant Analysis.

Finds the K−1 directions that maximally separate the K classes, then fits a
logistic-regression classifier in that low-dimensional subspace.

When K=2 this is mathematically equivalent to the original TTPD truth-direction
approach (single discriminant direction derived via eigendecomposition rather
than OLS).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from .base import AbstractProbe

if TYPE_CHECKING:
    from ..schema import LabelSchema


def _compute_fisher_directions(
    acts: torch.Tensor,
    labels: torch.Tensor,
    k: int,
) -> np.ndarray:
    """
    Compute the top K−1 Fisher discriminant directions.

    Returns
    -------
    W : ndarray [hidden_dim, K-1]
    """
    n, d = acts.shape
    global_mean = acts.mean(dim=0)

    S_b = torch.zeros(d, d)
    S_w = torch.zeros(d, d)

    for c in range(k):
        mask = labels == c
        acts_c = acts[mask]
        n_c = acts_c.size(0)
        if n_c == 0:
            continue
        mu_c = acts_c.mean(dim=0)
        diff = (mu_c - global_mean).unsqueeze(1)          # [d, 1]
        S_b += n_c * (diff @ diff.T)
        centered = acts_c - mu_c                            # [n_c, d]
        S_w += centered.T @ centered                        # [d, d]

    # Regularise S_w for numerical stability
    S_w += 1e-6 * torch.eye(d)

    # Generalised eigendecomposition: S_b v = λ S_w v
    # ⟺ S_w⁻¹ S_b v = λ v
    try:
        M = torch.linalg.solve(S_w, S_b)          # [d, d]
    except RuntimeError:
        # Fall back to pseudo-inverse if S_w is still singular
        M = torch.linalg.pinv(S_w) @ S_b

    # Top K-1 eigenvectors (real part)
    eigenvalues, eigenvectors = torch.linalg.eig(M)
    eigenvalues_real = eigenvalues.real
    order = torch.argsort(eigenvalues_real, descending=True)
    n_dirs = min(k - 1, d)
    top_dirs = order[:n_dirs]
    W = eigenvectors[:, top_dirs].real.numpy()  # [d, K-1]
    return W


class LDAProbe(AbstractProbe):
    """
    Fisher LDA probe (generalised TTPD).

    Projects activations into the K−1 dimensional discriminant subspace, then
    fits logistic regression in that subspace.
    """

    def __init__(
        self,
        projections: np.ndarray,
        classifier: LogisticRegression,
        schema: "LabelSchema",
    ) -> None:
        self.projections = projections   # [hidden_dim, K-1]
        self.classifier = classifier
        self.schema = schema

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @classmethod
    def from_data(
        cls,
        acts: torch.Tensor,
        labels: torch.Tensor,
        schema: "LabelSchema",
        max_iter: int = 1000,
        **kwargs,
    ) -> "LDAProbe":
        W = _compute_fisher_directions(acts, labels, schema.k)
        acts_proj = acts.numpy() @ W            # [N, K-1]

        clf = LogisticRegression(
            C=1e10,
            fit_intercept=True,
            max_iter=max_iter,
        )
        clf.fit(acts_proj, labels.numpy())
        return cls(projections=W, classifier=clf, schema=schema)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def pred(self, acts: torch.Tensor) -> torch.Tensor:
        acts_proj = acts.numpy() @ self.projections
        return torch.tensor(self.classifier.predict(acts_proj), dtype=torch.long)

    def predict_proba(self, acts: torch.Tensor) -> np.ndarray:
        acts_proj = acts.numpy() @ self.projections
        return self.classifier.predict_proba(acts_proj)
