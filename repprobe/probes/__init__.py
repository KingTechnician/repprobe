"""
Probe sub-package: ProbeFactory + all probe classes.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .base import AbstractProbe
from .ccs import CCSProbe
from .lda import LDAProbe
from .lr import LRProbe
from .mean_diff import MeanDiffProbe

if TYPE_CHECKING:
    from ..schema import LabelSchema

import torch

__all__ = [
    "AbstractProbe",
    "LRProbe",
    "LDAProbe",
    "MeanDiffProbe",
    "CCSProbe",
    "ProbeFactory",
]


class ProbeFactory:
    """
    Factory that instantiates and trains probes given a :class:`~repprobe.schema.LabelSchema`.

    Usage::

        probe = ProbeFactory.create("lda", acts, labels, schema)
        results = probe.evaluate(test_acts, test_labels)
    """

    _REGISTRY: dict[str, type[AbstractProbe]] = {
        "lr": LRProbe,
        "lda": LDAProbe,
        "mean_diff": MeanDiffProbe,
        "ccs": CCSProbe,
    }

    @staticmethod
    def available_probes(schema: "LabelSchema") -> list[type[AbstractProbe]]:
        """
        Return the probe types valid for *schema*.

        All schemas get: ``LRProbe``, ``LDAProbe``, ``MeanDiffProbe``.
        Binary schemas with polarity additionally get: ``CCSProbe``.
        """
        probes: list[type[AbstractProbe]] = [LRProbe, LDAProbe, MeanDiffProbe]
        if schema.is_binary and schema.has_polarity:
            probes.append(CCSProbe)
        return probes

    @staticmethod
    def create(
        probe_type: str,
        acts: torch.Tensor,
        labels: torch.Tensor,
        schema: "LabelSchema",
        **kwargs,
    ) -> AbstractProbe:
        """
        Train and return a probe of the requested type.

        Parameters
        ----------
        probe_type : str
            One of ``"lr"``, ``"lda"``, ``"mean_diff"``, ``"ccs"``.
        acts : Tensor [N, hidden_dim]
        labels : Tensor [N]
        schema : LabelSchema
        **kwargs : forwarded to the probe's ``from_data`` class method.

        Raises
        ------
        ValueError
            If *probe_type* is not recognised or not valid for *schema*.
        """
        key = probe_type.lower().replace("-", "_")
        if key not in ProbeFactory._REGISTRY:
            raise ValueError(
                f"Unknown probe type '{probe_type}'. "
                f"Options: {list(ProbeFactory._REGISTRY.keys())}"
            )
        probe_cls = ProbeFactory._REGISTRY[key]
        valid_types = [
            p.__name__.lower().replace("probe", "")
            for p in ProbeFactory.available_probes(schema)
        ]
        # Check availability
        available_classes = ProbeFactory.available_probes(schema)
        if probe_cls not in available_classes:
            raise ValueError(
                f"Probe '{probe_type}' is not valid for the given schema "
                f"(k={schema.k}, is_binary={schema.is_binary}, "
                f"has_polarity={schema.has_polarity}). "
                f"Valid probes: {[c.__name__ for c in available_classes]}"
            )
        return probe_cls.from_data(acts, labels, schema, **kwargs)
