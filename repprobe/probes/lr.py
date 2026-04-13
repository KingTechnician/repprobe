"""
LRProbe: Logistic Regression probe.

sklearn handles binary and multiclass transparently — no code changes needed
relative to the TiU version.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from .base import AbstractProbe

if TYPE_CHECKING:
    from ..schema import LabelSchema


class LRProbe(AbstractProbe):
    """
    Logistic Regression probe.

    Fits a penaltyless LR classifier directly in the activation space.
    Works for any number of classes.
    """

    def __init__(self, classifier: LogisticRegression, schema: "LabelSchema") -> None:
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
    ) -> "LRProbe":
        """
        Train a penaltyless logistic regression on raw activations.

        Parameters
        ----------
        acts : Tensor [N, hidden_dim]
        labels : Tensor [N]
        schema : LabelSchema
        max_iter : int
        """
        clf = LogisticRegression(
            C=1e10,
            fit_intercept=True,
            max_iter=max_iter,
        )
        clf.fit(acts.numpy(), labels.numpy())
        return cls(clf, schema)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def pred(self, acts: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.classifier.predict(acts.numpy()), dtype=torch.long)

    def predict_proba(self, acts: torch.Tensor) -> np.ndarray:
        return self.classifier.predict_proba(acts.numpy())
