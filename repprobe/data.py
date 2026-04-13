"""
RepDataset: HuggingFace dataset wrapper for probe training.

Contract: the underlying HF dataset must have at least:
  - ``text``  (str)  – the input utterance
  - ``label`` (int)  – integer-encoded class (0-indexed)

Activation columns, if present, are named ``layer_{N}_acts`` and contain
``list[float]`` of shape ``[hidden_dim]``.
"""
from __future__ import annotations

import re
from typing import Optional

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


class RepDataset:
    """
    Wraps a HuggingFace :class:`datasets.Dataset` for probe training.

    Parameters
    ----------
    dataset_name_or_path : str
        HuggingFace Hub dataset ID (e.g. ``"KingTechnician/my-ds"``) or a
        local directory that can be loaded with ``datasets.load_from_disk``.
    split : str | None
        Which split to load, e.g. ``"train"``.  ``None`` loads the whole
        :class:`DatasetDict` and uses the ``"train"`` split if only one
        split exists, or raises if ambiguous.
    hf_dataset : Dataset | None
        Pass an already-loaded HF Dataset directly (overrides
        ``dataset_name_or_path``).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        dataset_name_or_path: Optional[str] = None,
        split: Optional[str] = None,
        *,
        hf_dataset: Optional[Dataset] = None,
    ) -> None:
        if hf_dataset is not None:
            self._ds: Dataset = hf_dataset
        elif dataset_name_or_path is not None:
            self._ds = self._load(dataset_name_or_path, split)
        else:
            raise ValueError(
                "Provide either dataset_name_or_path or hf_dataset keyword argument."
            )

        self._validate()
        self._schema: Optional[LabelSchema] = None

    @staticmethod
    def _load(name_or_path: str, split: Optional[str]) -> Dataset:
        try:
            result = load_dataset(name_or_path, split=split, trust_remote_code=True)
        except Exception:
            try:
                from datasets import load_from_disk  # type: ignore
                result = load_from_disk(name_or_path)
                if isinstance(result, DatasetDict):
                    if split:
                        result = result[split]
                    elif len(result) == 1:
                        result = next(iter(result.values()))
                    else:
                        result = result["train"]
            except Exception as exc:
                raise ValueError(
                    f"Could not load dataset from '{name_or_path}'. "
                    f"Error: {exc}"
                ) from exc

        if isinstance(result, DatasetDict):
            if split:
                return result[split]
            if len(result) == 1:
                return next(iter(result.values()))
            if "train" in result:
                return result["train"]
            raise ValueError(
                f"Dataset '{name_or_path}' has multiple splits: {list(result.keys())}. "
                "Specify one via the `split` argument."
            )
        return result

    def _validate(self) -> None:
        missing = [c for c in ("text", "label") if c not in self._ds.column_names]
        if missing:
            raise ValueError(
                f"Dataset is missing required columns: {missing}. "
                f"Available columns: {self._ds.column_names}"
            )

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
    # Activation columns
    # ------------------------------------------------------------------

    @property
    def activation_layers(self) -> list[int]:
        """List of layer indices for which activation columns exist."""
        layers = []
        for col in self._ds.column_names:
            m = _ACT_COL_RE.match(col)
            if m:
                layers.append(int(m.group(1)))
        return sorted(layers)

    def get_acts(self, layer: int) -> torch.Tensor:
        """
        Return activation tensor ``[N, hidden_dim]`` for the given layer.

        Requires ``layer_{N}_acts`` columns to be present.
        """
        col = f"layer_{layer}_acts"
        if col not in self._ds.column_names:
            available = self.activation_layers
            raise KeyError(
                f"No activation column '{col}' found. "
                f"Available layers: {available}"
            )
        import numpy as np  # noqa: PLC0415

        raw = self._ds[col]
        arr = np.array(raw, dtype="float32")
        return torch.tensor(arr)

    def get_labels(self) -> torch.Tensor:
        """Return integer label tensor ``[N]``."""
        return torch.tensor(self._ds["label"], dtype=torch.long)

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def stratified_split(
        self, train_frac: float = 0.8, seed: int = 42
    ) -> tuple["RepDataset", "RepDataset"]:
        """
        Standard stratified train / test split.

        Uses HF's native stratified split when the label column is a
        ``ClassLabel`` feature; otherwise falls back to a manual
        numpy-based stratified split that works with plain integer columns.

        Returns
        -------
        train_ds, test_ds : RepDataset
        """
        from datasets import ClassLabel  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        # ── Native stratified split (ClassLabel only) ──────────────────
        label_feature = self._ds.features.get("label")
        if isinstance(label_feature, ClassLabel):
            split_result = self._ds.train_test_split(
                train_size=train_frac,
                stratify_by_column="label",
                seed=seed,
            )
            train_hf = split_result["train"]
            test_hf = split_result["test"]
        else:
            # ── Manual stratified split ────────────────────────────────
            rng = np.random.default_rng(seed)
            labels = np.array(self._ds["label"])
            train_indices, test_indices = [], []
            for cls in np.unique(labels):
                cls_idx = np.where(labels == cls)[0]
                rng.shuffle(cls_idx)
                n_train = max(1, int(len(cls_idx) * train_frac))
                train_indices.extend(cls_idx[:n_train].tolist())
                test_indices.extend(cls_idx[n_train:].tolist())
            train_hf = self._ds.select(train_indices)
            test_hf = self._ds.select(test_indices)

        train_rds = RepDataset(hf_dataset=train_hf)
        test_rds = RepDataset(hf_dataset=test_hf)
        # Propagate cached schema
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
        Generalization split — train on rows where ``column`` is in
        ``train_values``, test on rows where it is in ``test_values``.

        Parameters
        ----------
        column : str
            Column name to split on (e.g. ``"objective"``).
        train_values : list
            Values to include in the training set.
        test_values : list
            Values to include in the test set.
        """
        if column not in self._ds.column_names:
            raise ValueError(
                f"Column '{column}' not found. "
                f"Available: {self._ds.column_names}"
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
    # Delegated dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._ds)

    def __repr__(self) -> str:
        return (
            f"RepDataset(rows={len(self)}, columns={self._ds.column_names}, "
            f"schema={self.schema})"
        )

    @property
    def hf_dataset(self) -> Dataset:
        """The underlying :class:`datasets.Dataset`."""
        return self._ds

    # ------------------------------------------------------------------
    # Backward-compat: CSV adapter
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        text_col: str = "statement",
        label_col: str = "label",
    ) -> "RepDataset":
        """
        Load a TiU-style CSV and wrap it in the RepDataset interface.

        This is a backward-compatibility adapter: existing TiU CSVs work
        without modification.
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
