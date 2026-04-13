"""
AbstractProbe: common interface all probe implementations must satisfy.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from ..schema import LabelSchema


class AbstractProbe(ABC):
    """
    Abstract base class for representation probes.

    All probes are trained via the :meth:`from_data` class-method and expose
    :meth:`pred`, :meth:`predict_proba`, and :meth:`evaluate`.
    """

    schema: "LabelSchema"

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def from_data(
        cls,
        acts: torch.Tensor,
        labels: torch.Tensor,
        schema: "LabelSchema",
        **kwargs,
    ) -> "AbstractProbe":
        """
        Train the probe from activation tensors and labels.

        Parameters
        ----------
        acts : Tensor [N, hidden_dim]
        labels : Tensor [N]  (long, 0-indexed)
        schema : LabelSchema
        **kwargs : probe-specific hyper-parameters
        """
        ...

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @abstractmethod
    def pred(self, acts: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices as a ``[N]`` long tensor."""
        ...

    @abstractmethod
    def predict_proba(self, acts: torch.Tensor) -> np.ndarray:
        """Return class probability matrix of shape ``[N, K]``."""
        ...

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self, acts: torch.Tensor, labels: torch.Tensor
    ) -> dict:
        """
        Standard evaluation on a held-out set.

        Returns
        -------
        dict with keys:
            ``accuracy``, ``macro_f1``, ``per_class_accuracy``,
            ``confusion_matrix``
        """
        from sklearn.metrics import (  # noqa: PLC0415
            accuracy_score,
            confusion_matrix,
            f1_score,
        )

        preds = self.pred(acts).numpy()
        true = labels.numpy()

        acc = float(accuracy_score(true, preds))
        macro_f1 = float(f1_score(true, preds, average="macro", zero_division=0))
        cm = confusion_matrix(true, preds, labels=list(range(self.schema.k))).tolist()

        per_class: dict[str, float] = {}
        for idx, name in enumerate(self.schema.names):
            mask = true == idx
            if mask.sum() > 0:
                per_class[name] = float((preds[mask] == idx).mean())
            else:
                per_class[name] = float("nan")

        return {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "per_class_accuracy": per_class,
            "confusion_matrix": cm,
        }
