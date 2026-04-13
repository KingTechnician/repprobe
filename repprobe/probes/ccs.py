"""
CCSProbe: Contrastive Consistency Search — binary-only.

CCS requires paired affirmative/negated statements and a consistency loss that
is inherently binary.  This probe is gated: :class:`~repprobe.probes.ProbeFactory`
will only offer it when ``schema.is_binary and schema.has_polarity``.

The implementation is ported from the original TiU codebase with light
refactoring to match the :class:`~repprobe.probes.base.AbstractProbe` interface.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import torch.nn as nn

from .base import AbstractProbe

if TYPE_CHECKING:
    from ..schema import LabelSchema


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def ccs_loss(probe: "CCSProbe", acts: torch.Tensor, neg_acts: torch.Tensor) -> torch.Tensor:
    """Consistency + confidence loss used during CCS training."""
    p_pos = probe(acts)
    p_neg = probe(neg_acts)
    consistency = (p_pos - (1 - p_neg)) ** 2
    confidence = torch.min(torch.stack([p_pos, p_neg], dim=-1), dim=-1).values ** 2
    return (consistency + confidence).mean()


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


class CCSProbe(AbstractProbe, nn.Module):
    """
    Contrastive Consistency Search probe (binary only).

    Must be trained via :meth:`from_data` which requires both affirmative
    *and* negated activation tensors.

    .. warning::
        The standard :meth:`~repprobe.probes.base.AbstractProbe.from_data`
        signature does not carry ``neg_acts``; call
        :meth:`CCSProbe.from_data_with_neg` for the full API.
        When called through :class:`~repprobe.probes.ProbeFactory`, pass
        ``neg_acts`` as a keyword argument.
    """

    def __init__(self, d_in: int, schema: "LabelSchema") -> None:
        nn.Module.__init__(self)
        self.schema = schema
        self.net = nn.Sequential(
            nn.Linear(d_in, 1, bias=True),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------
    # Forward / pred
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x).squeeze(-1)

    def pred(self, acts: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        with torch.no_grad():
            return self(acts).round().long()

    def predict_proba(self, acts: torch.Tensor) -> np.ndarray:  # type: ignore[override]
        with torch.no_grad():
            p = self(acts).numpy()
        return np.column_stack([1 - p, p])

    # ------------------------------------------------------------------
    # Training — AbstractProbe interface
    # ------------------------------------------------------------------

    @classmethod
    def from_data(
        cls,
        acts: torch.Tensor,
        labels: torch.Tensor,
        schema: "LabelSchema",
        *,
        neg_acts: Optional[torch.Tensor] = None,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        epochs: int = 1000,
        device: str = "cpu",
        **kwargs,
    ) -> "CCSProbe":
        """
        Train the CCS probe.

        Parameters
        ----------
        acts : Tensor [N, hidden_dim]
            Affirmative statement activations.
        labels : Tensor [N]
            Ground-truth binary labels (0/1).
        schema : LabelSchema
            Must be binary.
        neg_acts : Tensor [N, hidden_dim]
            Negated statement activations (required).
        """
        if not schema.is_binary:
            raise ValueError("CCSProbe is only supported for binary schemas.")
        if neg_acts is None:
            raise ValueError(
                "CCSProbe requires 'neg_acts' keyword argument "
                "(activations of negated statements)."
            )

        return cls._train(acts, neg_acts, labels, schema, lr, weight_decay, epochs, device)

    @classmethod
    def _train(
        cls,
        acts: torch.Tensor,
        neg_acts: torch.Tensor,
        labels: Optional[torch.Tensor],
        schema: "LabelSchema",
        lr: float,
        weight_decay: float,
        epochs: int,
        device: str,
    ) -> "CCSProbe":
        acts = acts.to(device).float()
        neg_acts = neg_acts.to(device).float()

        probe = cls(acts.shape[-1], schema).to(device)
        opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)

        for _ in range(epochs):
            opt.zero_grad()
            loss = ccs_loss(probe, acts, neg_acts)
            loss.backward()
            opt.step()

        # Flip direction if needed
        if labels is not None:
            labels = labels.to(device)
            with torch.no_grad():
                acc = (probe.pred(acts) == labels).float().mean()
            if acc < 0.5:
                probe.net[0].weight.data *= -1

        return probe

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def direction(self) -> torch.Tensor:
        """The learned probe direction vector."""
        return self.net[0].weight.data[0]

    @property
    def bias(self) -> torch.Tensor:  # type: ignore[override]
        """The learned bias scalar."""
        return self.net[0].bias.data[0]
