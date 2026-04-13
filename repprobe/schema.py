"""
LabelSchema: Describes the label space of a probe task.

Auto-detected from HuggingFace ClassLabel features or inferred from
integer label values in the dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset  # type: ignore


@dataclass
class LabelSchema:
    """
    Describes the label space for a representation probe task.

    Attributes
    ----------
    names : list[str]
        Human-readable class names (e.g. ``["False", "True"]``).
    k : int
        Number of classes (auto-derived from ``len(names)``).
    ordinal : bool
        Whether labels have a natural ordering (e.g. severity levels).
    has_polarity : bool
        Whether the dataset includes a ``polarity`` column (±1).
        Enables TTPD/CCS probes when *k == 2*.
    """

    names: list[str]
    k: int = field(init=False)
    ordinal: bool = False
    has_polarity: bool = False

    def __post_init__(self) -> None:
        if not self.names:
            raise ValueError("LabelSchema requires at least one class name.")
        object.__setattr__(self, "k", len(self.names))

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_hf_dataset(cls, dataset: "Dataset") -> "LabelSchema":
        """
        Auto-detect schema from a HuggingFace ``Dataset``.

        Priority:
        1. ``ClassLabel`` feature on the ``label`` column (names + count).
        2. Unique integer values in the ``label`` column.
        """
        features = dataset.features

        # ── 1. ClassLabel feature ──────────────────────────────────────
        if "label" in features:
            feat = features["label"]
            # datasets.ClassLabel has .names attribute
            try:
                if hasattr(feat, "names") and feat.names:
                    has_polarity = "polarity" in features
                    return cls(names=list(feat.names), has_polarity=has_polarity)
            except Exception:
                pass  # fall through to inference

        # ── 2. Infer from unique values ────────────────────────────────
        if "label" not in dataset.column_names:
            raise ValueError(
                "Dataset has no 'label' column. "
                "RepProbe requires a 'text' and 'label' column."
            )

        unique_vals = sorted(set(dataset["label"]))
        k = len(unique_vals)
        # Generate synthetic names: "class_0", "class_1", ...
        names = [f"class_{v}" for v in unique_vals]
        has_polarity = "polarity" in dataset.column_names
        return cls(names=names, has_polarity=has_polarity)

    @classmethod
    def binary(
        cls, neg_name: str = "False", pos_name: str = "True", *, has_polarity: bool = False
    ) -> "LabelSchema":
        """Convenience constructor for binary schemas."""
        return cls(names=[neg_name, pos_name], has_polarity=has_polarity)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_binary(self) -> bool:
        """``True`` when exactly two classes."""
        return self.k == 2

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"LabelSchema(names={self.names!r}, k={self.k}, "
            f"ordinal={self.ordinal}, has_polarity={self.has_polarity})"
        )
