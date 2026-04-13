"""
RepDataset: HuggingFace dataset wrapper for probe training.

Contract: the underlying HF dataset must have at least:
  - ``text``  (str)  – the input utterance
  - ``label`` (int)  – integer-encoded class (0-indexed)

Activation columns, if present, are named ``layer_{N}_acts`` and contain
``list[float]`` of shape ``[hidden_dim]``.

DatasetDict support
-------------------
``RepDataset`` accepts a :class:`~datasets.DatasetDict` directly (via the
``hf_dataset`` kwarg or ``from_dataset_dict``).  When built from a
``DatasetDict`` the object exposes each split via ``splits`` and
``get_split(name)``; the default split used by ``get_acts`` / ``get_labels``
/ all probe-training helpers is the first split in the dict (typically
``"train"``).  Pass ``default_split=`` to override.
"""
from __future__ import annotations

import re
from typing import Optional, Union

import torch

from .schema import LabelSchema

try:
    from datasets import Dataset, DatasetDict, load_dataset  # type: ignore
    import pandas as pd  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "RepProbe requires the 'datasets' and 'pandas' packages. "
        "Install them with: pip install datasets pandas"
    ) from exc

_ACT_COL_RE = re.compile(r"^layer_(\d+)_acts$")

# Type alias for what we accept
_DatasetInput = Union[Dataset, DatasetDict]


class RepDataset:
    """
    Wraps a HuggingFace :class:`~datasets.Dataset` **or**
    :class:`~datasets.DatasetDict` for probe training.

    Parameters
    ----------
    dataset_name_or_path : str | None
        HuggingFace Hub dataset ID or a local directory loadable with
        ``datasets.load_from_disk``.
    split : str | None
        Which split to select when loading from a name/path.  ``None``
        loads the whole ``DatasetDict``; if it has a ``"train"`` key that
        becomes the *default split*.
    hf_dataset : Dataset | DatasetDict | None
        Pass an already-loaded object directly (overrides
        ``dataset_name_or_path``).  Both ``Dataset`` and ``DatasetDict``
        are accepted.
    default_split : str | None
        When *hf_dataset* is a ``DatasetDict``, which split to use for
        ``get_acts`` / ``get_labels`` / all probe-training helpers.
        Defaults to ``"train"`` if present, otherwise the first key.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        dataset_name_or_path: Optional[str] = None,
        split: Optional[str] = None,
        *,
        hf_dataset: Optional[_DatasetInput] = None,
        default_split: Optional[str] = None,
    ) -> None:
        if hf_dataset is not None:
            raw = hf_dataset
        elif dataset_name_or_path is not None:
            raw = self._load(dataset_name_or_path, split)
        else:
            raise ValueError(
                "Provide either dataset_name_or_path or hf_dataset keyword argument."
            )

        # ── Normalise into _splits dict ────────────────────────────────
        if isinstance(raw, DatasetDict):
            self._splits: dict[str, Dataset] = dict(raw)
            if default_split is not None:
                if default_split not in self._splits:
                    raise ValueError(
                        f"default_split='{default_split}' not found in DatasetDict. "
                        f"Available splits: {list(self._splits.keys())}"
                    )
                self._default_split: str = default_split
            elif "train" in self._splits:
                self._default_split = "train"
            else:
                self._default_split = next(iter(self._splits))
        else:
            # Plain Dataset — stored as a single-split dict
            self._splits = {"_flat": raw}
            self._default_split = "_flat"

        # Validate every split
        for name, ds in self._splits.items():
            self._validate(ds, split_name=name)

        self._schema: Optional[LabelSchema] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load(name_or_path: str, split: Optional[str]) -> _DatasetInput:
        try:
            result = load_dataset(name_or_path, split=split, trust_remote_code=True)
        except Exception:
            try:
                from datasets import load_from_disk  # noqa: PLC0415
                result = load_from_disk(name_or_path)
                if isinstance(result, DatasetDict) and split:
                    result = result[split]
            except Exception as exc:
                raise ValueError(
                    f"Could not load dataset from '{name_or_path}'. Error: {exc}"
                ) from exc

        # If a specific split was requested and we got a DatasetDict, select it
        if split and isinstance(result, DatasetDict):
            return result[split]
        return result

    @staticmethod
    def _validate(ds: Dataset, split_name: str = "") -> None:
        label = f"split '{split_name}'" if split_name and split_name != "_flat" else "dataset"
        missing = [c for c in ("text", "label") if c not in ds.column_names]
        if missing:
            raise ValueError(
                f"The {label} is missing required columns: {missing}. "
                f"Available columns: {ds.column_names}"
            )

    # ------------------------------------------------------------------
    # Split access
    # ------------------------------------------------------------------

    @property
    def splits(self) -> list[str]:
        """Names of all available splits (excluding the internal ``'_flat'`` key)."""
        return [k for k in self._splits if k != "_flat"]

    @property
    def is_dict(self) -> bool:
        """``True`` when built from a ``DatasetDict``."""
        return "_flat" not in self._splits

    def get_split(self, name: str) -> "RepDataset":
        """Return a single-split :class:`RepDataset` for the named split."""
        if name not in self._splits:
            raise KeyError(
                f"Split '{name}' not found. Available: {list(self._splits.keys())}"
            )
        child = RepDataset(hf_dataset=self._splits[name])
        if self._schema is not None:
            child._schema = self._schema
        return child

    # ------------------------------------------------------------------
    # Active (default-split) dataset
    # ------------------------------------------------------------------

    @property
    def _ds(self) -> Dataset:
        """The active (default-split) :class:`~datasets.Dataset`."""
        return self._splits[self._default_split]

    @property
    def hf_dataset(self) -> Dataset:
        """The active split as a plain :class:`~datasets.Dataset`."""
        return self._ds

    @property
    def hf_dataset_dict(self) -> DatasetDict:
        """
        All splits as a :class:`~datasets.DatasetDict`.

        Raises ``TypeError`` if this ``RepDataset`` was not built from a
        ``DatasetDict`` (use ``hf_dataset`` for flat datasets).
        """
        if not self.is_dict:
            raise TypeError(
                "This RepDataset was not built from a DatasetDict. "
                "Use .hf_dataset to access the underlying Dataset."
            )
        return DatasetDict(self._splits)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    @property
    def schema(self) -> LabelSchema:
        """Auto-detected :class:`LabelSchema` (cached after first call)."""
        if self._schema is None:
            self._schema = LabelSchema.from_hf_dataset(self._ds)
        return self._schema

    # ------------------------------------------------------------------
    # Activation columns  (operate on active split)
    # ------------------------------------------------------------------

    @property
    def activation_layers(self) -> list[int]:
        """Layer indices for which ``layer_{N}_acts`` columns exist in the active split."""
        return sorted(
            int(m.group(1))
            for col in self._ds.column_names
            if (m := _ACT_COL_RE.match(col))
        )

    def get_acts(self, layer: int) -> torch.Tensor:
        """
        Return activation tensor ``[N, hidden_dim]`` for *layer* from the
        active split.
        """
        col = f"layer_{layer}_acts"
        if col not in self._ds.column_names:
            raise KeyError(
                f"No activation column '{col}' found. "
                f"Available layers: {self.activation_layers}"
            )
        import numpy as np  # noqa: PLC0415
        return torch.tensor(np.array(self._ds[col], dtype="float32"))

    def get_labels(self) -> torch.Tensor:
        """Return integer label tensor ``[N]`` from the active split."""
        return torch.tensor(self._ds["label"], dtype=torch.long)

    # ------------------------------------------------------------------
    # Splitting helpers  (operate on active split)
    # ------------------------------------------------------------------

    def stratified_split(
        self, train_frac: float = 0.8, seed: int = 42
    ) -> tuple["RepDataset", "RepDataset"]:
        """
        Stratified train / test split on the active split.

        Uses HF's native stratification for ``ClassLabel`` columns; falls
        back to a manual numpy-based split for plain integer labels.
        """
        from datasets import ClassLabel  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        label_feature = self._ds.features.get("label")
        if isinstance(label_feature, ClassLabel):
            split_result = self._ds.train_test_split(
                train_size=train_frac,
                stratify_by_column="label",
                seed=seed,
            )
            train_hf, test_hf = split_result["train"], split_result["test"]
        else:
            rng = np.random.default_rng(seed)
            labels = np.array(self._ds["label"])
            train_idx, test_idx = [], []
            for cls in np.unique(labels):
                idx = np.where(labels == cls)[0]
                rng.shuffle(idx)
                n_tr = max(1, int(len(idx) * train_frac))
                train_idx.extend(idx[:n_tr].tolist())
                test_idx.extend(idx[n_tr:].tolist())
            train_hf = self._ds.select(train_idx)
            test_hf = self._ds.select(test_idx)

        train_rds = RepDataset(hf_dataset=train_hf)
        test_rds = RepDataset(hf_dataset=test_hf)
        if self._schema is not None:
            train_rds._schema = self._schema
            test_rds._schema = self._schema
        return train_rds, test_rds

    def split_by_column(
        self,
        column: str,
        train_values: list,
        test_values: list,
    ) -> tuple["RepDataset", "RepDataset"]:
        """
        Generalization split — filter the active split by *column* values.
        """
        if column not in self._ds.column_names:
            raise ValueError(
                f"Column '{column}' not found. Available: {self._ds.column_names}"
            )
        train_hf = self._ds.filter(lambda row: row[column] in train_values)
        test_hf = self._ds.filter(lambda row: row[column] in test_values)

        train_rds = RepDataset(hf_dataset=train_hf)
        test_rds = RepDataset(hf_dataset=test_hf)
        if self._schema is not None:
            train_rds._schema = self._schema
            test_rds._schema = self._schema
        return train_rds, test_rds

    # ------------------------------------------------------------------
    # Delegated interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._ds)

    def __repr__(self) -> str:
        if self.is_dict:
            split_info = ", ".join(
                f"{k}({len(v)})" for k, v in self._splits.items()
            )
            return (
                f"RepDataset(splits=[{split_info}], "
                f"default='{self._default_split}', schema={self.schema})"
            )
        return (
            f"RepDataset(rows={len(self)}, columns={self._ds.column_names}, "
            f"schema={self.schema})"
        )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_dataset_dict(
        cls,
        dataset_dict: DatasetDict,
        default_split: Optional[str] = None,
    ) -> "RepDataset":
        """
        Build a ``RepDataset`` from an existing :class:`~datasets.DatasetDict`.

        Parameters
        ----------
        dataset_dict : DatasetDict
            Must have ``"text"`` and ``"label"`` columns in every split.
        default_split : str | None
            Which split becomes the active split for ``get_acts`` /
            ``get_labels``.  Defaults to ``"train"`` when present.
        """
        return cls(hf_dataset=dataset_dict, default_split=default_split)

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        text_col: str = "statement",
        label_col: str = "label",
    ) -> "RepDataset":
        """
        Load a TiU-style CSV and wrap it in the RepDataset interface.

        Backward-compatibility adapter: existing TiU CSVs work without
        modification.
        """
        df = pd.read_csv(csv_path)
        rename_map: dict[str, str] = {}
        if text_col != "text":
            rename_map[text_col] = "text"
        if label_col != "label":
            rename_map[label_col] = "label"
        if rename_map:
            df = df.rename(columns=rename_map)
        return cls(hf_dataset=Dataset.from_pandas(df))
