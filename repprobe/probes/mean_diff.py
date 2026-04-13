"""
MeanDiffProbe: Generalised Mass-Mean (MMProbe) for K classes.

Binary MMProbe computes ``pos_mean − neg_mean`` and fits a 1-D LR on the
projection.  The K-class generalisation computes K deviation vectors
``μ_k − μ_global`` and projects activations onto that K-dimensional subspace
before fitting multiclass LR.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from .base import AbstractProbe

if TYPE_CHECKING:
    from ..schema import LabelSchema


class MeanDiffProbe(AbstractProbe):
    """
    Class-mean deviation probe (generalised MMProbe).

    Parameters
    ----------
    directions : ndarray [hidden_dim, K]
        Each column is ``μ_k − μ_global`` for class k.
    classifier : LogisticRegression
        Fitted in the K-dimensional projected space.
    schema : LabelSchema
    """

    def __init__(
        self,
        directions: np.ndarray,
        classifier: LogisticRegression,
        schema: "LabelSchema",
    ) -> None:
        self.directions = directions    # [hidden_dim, K]
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
    ) -> "MeanDiffProbe":
        """
        Compute class-mean deviation vectors and fit multiclass LR.
        """
        global_mean = acts.mean(dim=0)              # [hidden_dim]
        dirs = []
        for c in range(schema.k):
            mask = labels == c
            if mask.sum() == 0:
                dirs.append(torch.zeros_like(global_mean))
            else:
                mu_c = acts[mask].mean(dim=0)
                dirs.append(mu_c - global_mean)
        directions = torch.stack(dirs, dim=1).numpy()   # [hidden_dim, K]

        acts_proj = acts.numpy() @ directions            # [N, K]

        clf = LogisticRegression(
            C=1e10,
            fit_intercept=True,
            max_iter=max_iter,
        )
        clf.fit(acts_proj, labels.numpy())
        return cls(directions=directions, classifier=clf, schema=schema)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def pred(self, acts: torch.Tensor) -> torch.Tensor:
        acts_proj = acts.numpy() @ self.directions
        return torch.tensor(self.classifier.predict(acts_proj), dtype=torch.long)

    def predict_proba(self, acts: torch.Tensor) -> np.ndarray:
        acts_proj = acts.numpy() @ self.directions
        return self.classifier.predict_proba(acts_proj)
