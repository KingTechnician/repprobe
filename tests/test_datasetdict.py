"""
Tests for DatasetDict support in RepDataset and generate_activations.

These tests cover the exact scenario described:
  - A DatasetDict with 'train' and 'test' splits
  - Each split has a numeric 'label' column and a 'text' column
  - RepDataset wraps it, exposing per-split access
  - generate_activations runs extraction over every split and returns a
    DatasetDict with layer_{N}_acts columns added to each split
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch
from datasets import Dataset, DatasetDict

from repprobe import RepDataset
from repprobe.extraction import generate_activations


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_split(n: int, k: int, seed: int) -> Dataset:
    """Flat Dataset with numeric label + text, NO activation columns."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, k, size=n).tolist()
    texts = [f"Statement {i} label {labels[i]}" for i in range(n)]
    return Dataset.from_dict({"text": texts, "label": labels})


@pytest.fixture
def raw_dd() -> DatasetDict:
    """
    The canonical test fixture: DatasetDict with 'train' (160 rows) and
    'test' (40 rows), numeric label (0/1/2), no activation columns.
    """
    return DatasetDict({
        "train": _make_split(n=160, k=3, seed=0),
        "test":  _make_split(n=40,  k=3, seed=1),
    })


@pytest.fixture
def dd_with_acts() -> DatasetDict:
    """DatasetDict that already has layer_12_acts columns."""
    rng = np.random.default_rng(42)
    k, hidden = 3, 16

    def _split_with_acts(n, seed):
        labels = rng.integers(0, k, size=n).tolist()
        texts = [f"text {i}" for i in range(n)]
        acts = (rng.normal(size=(n, hidden))).tolist()
        return Dataset.from_dict({"text": texts, "label": labels, "layer_12_acts": acts})

    return DatasetDict({
        "train": _split_with_acts(120, 0),
        "test":  _split_with_acts(40,  1),
    })


# ---------------------------------------------------------------------------
# RepDataset — DatasetDict construction
# ---------------------------------------------------------------------------


class TestRepDatasetFromDatasetDict:
    def test_accepts_dataset_dict(self, raw_dd):
        ds = RepDataset(hf_dataset=raw_dd)
        assert ds.is_dict

    def test_splits_property(self, raw_dd):
        ds = RepDataset(hf_dataset=raw_dd)
        assert set(ds.splits) == {"train", "test"}

    def test_default_split_is_train(self, raw_dd):
        ds = RepDataset(hf_dataset=raw_dd)
        assert ds._default_split == "train"
        assert len(ds) == 160          # train has 160 rows

    def test_explicit_default_split(self, raw_dd):
        ds = RepDataset(hf_dataset=raw_dd, default_split="test")
        assert ds._default_split == "test"
        assert len(ds) == 40

    def test_bad_default_split_raises(self, raw_dd):
        with pytest.raises(ValueError, match="default_split='missing'"):
            RepDataset(hf_dataset=raw_dd, default_split="missing")

    def test_from_dataset_dict_classmethod(self, raw_dd):
        ds = RepDataset.from_dataset_dict(raw_dd)
        assert ds.is_dict
        assert set(ds.splits) == {"train", "test"}

    def test_from_dataset_dict_with_default(self, raw_dd):
        ds = RepDataset.from_dataset_dict(raw_dd, default_split="test")
        assert len(ds) == 40

    def test_get_split_returns_child(self, raw_dd):
        ds = RepDataset(hf_dataset=raw_dd)
        train = ds.get_split("train")
        test  = ds.get_split("test")
        assert isinstance(train, RepDataset)
        assert len(train) == 160
        assert len(test)  == 40

    def test_get_split_bad_name_raises(self, raw_dd):
        ds = RepDataset(hf_dataset=raw_dd)
        with pytest.raises(KeyError, match="'nope'"):
            ds.get_split("nope")

    def test_hf_dataset_dict_property(self, raw_dd):
        ds = RepDataset(hf_dataset=raw_dd)
        dd = ds.hf_dataset_dict
        assert isinstance(dd, DatasetDict)
        assert set(dd.keys()) == {"train", "test"}

    def test_hf_dataset_dict_raises_on_flat(self):
        flat = Dataset.from_dict({"text": ["a"], "label": [0]})
        ds = RepDataset(hf_dataset=flat)
        with pytest.raises(TypeError, match="not built from a DatasetDict"):
            _ = ds.hf_dataset_dict

    def test_repr_mentions_splits(self, raw_dd):
        ds = RepDataset(hf_dataset=raw_dd)
        r = repr(ds)
        assert "train" in r
        assert "test" in r


# ---------------------------------------------------------------------------
# Validation: each split must have text + label
# ---------------------------------------------------------------------------


class TestDatasetDictValidation:
    def test_missing_text_in_split_raises(self):
        bad = DatasetDict({
            "train": Dataset.from_dict({"text": ["a"], "label": [0]}),
            "test":  Dataset.from_dict({"label": [1]}),          # no text
        })
        with pytest.raises(ValueError, match="missing required columns"):
            RepDataset(hf_dataset=bad)

    def test_missing_label_in_split_raises(self):
        bad = DatasetDict({
            "train": Dataset.from_dict({"text": ["a"], "label": [0]}),
            "test":  Dataset.from_dict({"text": ["b"]}),          # no label
        })
        with pytest.raises(ValueError, match="missing required columns"):
            RepDataset(hf_dataset=bad)

    def test_numeric_label_accepted(self, raw_dd):
        """Plain int labels (not ClassLabel) must be accepted without error."""
        ds = RepDataset(hf_dataset=raw_dd)
        labels = ds.get_labels()
        assert labels.dtype == torch.long


# ---------------------------------------------------------------------------
# Schema detection through DatasetDict
# ---------------------------------------------------------------------------


class TestSchemaFromDatasetDict:
    def test_schema_k_detected(self, raw_dd):
        ds = RepDataset(hf_dataset=raw_dd)
        assert ds.schema.k == 3

    def test_schema_uses_default_split(self, raw_dd):
        # Train has k=3; schema should reflect the default (train) split
        ds = RepDataset(hf_dataset=raw_dd)
        assert ds.schema.k == 3

    def test_get_split_inherits_schema(self, raw_dd):
        ds = RepDataset(hf_dataset=raw_dd)
        _ = ds.schema          # prime cache
        child = ds.get_split("test")
        assert child._schema is ds._schema   # same cached object


# ---------------------------------------------------------------------------
# Activation access through DatasetDict
# ---------------------------------------------------------------------------


class TestActivationAccessDatasetDict:
    def test_get_acts_from_default_split(self, dd_with_acts):
        ds = RepDataset(hf_dataset=dd_with_acts)
        acts = ds.get_acts(12)
        assert acts.shape == (120, 16)     # train split

    def test_get_acts_from_test_split(self, dd_with_acts):
        ds = RepDataset(hf_dataset=dd_with_acts, default_split="test")
        acts = ds.get_acts(12)
        assert acts.shape == (40, 16)

    def test_get_acts_via_get_split(self, dd_with_acts):
        ds = RepDataset(hf_dataset=dd_with_acts)
        test_child = ds.get_split("test")
        acts = test_child.get_acts(12)
        assert acts.shape == (40, 16)

    def test_activation_layers_on_dict(self, dd_with_acts):
        ds = RepDataset(hf_dataset=dd_with_acts)
        assert 12 in ds.activation_layers

    def test_get_acts_no_col_raises(self, raw_dd):
        ds = RepDataset(hf_dataset=raw_dd)
        with pytest.raises(KeyError, match="No activation column"):
            ds.get_acts(12)

    def test_get_labels_from_default_split(self, raw_dd):
        ds = RepDataset(hf_dataset=raw_dd)
        labels = ds.get_labels()
        assert labels.shape == (160,)
        assert labels.dtype == torch.long

    def test_get_labels_from_test_split(self, raw_dd):
        ds = RepDataset(hf_dataset=raw_dd, default_split="test")
        labels = ds.get_labels()
        assert labels.shape == (40,)


# ---------------------------------------------------------------------------
# generate_activations — DatasetDict end-to-end
# ---------------------------------------------------------------------------


def _make_stub_model_for_extraction(hidden_dim: int = 16, num_layers: int = 4):
    """Minimal stub model that fires forward hooks on each layer."""

    class StubLayer(torch.nn.Module):
        def __init__(self, d):
            super().__init__()
            self._d = d

        def forward(self, x):
            return (torch.randn(x.shape[0], x.shape[1], self._d),)

    class StubInner(torch.nn.Module):
        def __init__(self, d, n):
            super().__init__()
            self.layers = torch.nn.ModuleList([StubLayer(d) for _ in range(n)])
            self._d = d

        def forward(self, input_ids):
            x = torch.randn(input_ids.shape[0], input_ids.shape[1], self._d)
            for layer in self.layers:
                x = layer(x)[0]
            return x

    class StubLM(torch.nn.Module):
        def __init__(self, d, n):
            super().__init__()
            self.model = StubInner(d, n)

        def forward(self, input_ids):
            return self.model(input_ids)

    return StubLM(hidden_dim, num_layers)


class TestGenerateActivationsDatasetDict:
    def _patched_load_model(self, hidden_dim=16):
        """Patch load_model so no real weights are needed."""
        model = _make_stub_model_for_extraction(hidden_dim=hidden_dim, num_layers=4)
        tokenizer = MagicMock()
        tokenizer.encode = lambda text, return_tensors="pt": torch.randint(0, 100, (1, 5))
        return tokenizer, model

    def test_returns_dataset_dict_when_given_dataset_dict(self, raw_dd):
        """Core contract: DatasetDict in → DatasetDict out."""
        tok, mdl = self._patched_load_model()
        with patch("repprobe.extraction.load_model", return_value=(tok, mdl)):
            result = generate_activations(
                raw_dd,
                model_family="Llama3",
                model_size="8B",
                model_type="instruct",
                layers=[1],
            )
        assert isinstance(result, DatasetDict)

    def test_both_splits_present_in_result(self, raw_dd):
        tok, mdl = self._patched_load_model()
        with patch("repprobe.extraction.load_model", return_value=(tok, mdl)):
            result = generate_activations(
                raw_dd,
                model_family="Llama3", model_size="8B", model_type="instruct",
                layers=[1],
            )
        assert "train" in result
        assert "test"  in result

    def test_act_columns_added_to_every_split(self, raw_dd):
        tok, mdl = self._patched_load_model()
        with patch("repprobe.extraction.load_model", return_value=(tok, mdl)):
            result = generate_activations(
                raw_dd,
                model_family="Llama3", model_size="8B", model_type="instruct",
                layers=[1],
            )
        assert "layer_1_acts" in result["train"].column_names
        assert "layer_1_acts" in result["test"].column_names

    def test_original_columns_preserved(self, raw_dd):
        tok, mdl = self._patched_load_model()
        with patch("repprobe.extraction.load_model", return_value=(tok, mdl)):
            result = generate_activations(
                raw_dd,
                model_family="Llama3", model_size="8B", model_type="instruct",
                layers=[1],
            )
        for split in ("train", "test"):
            assert "text"  in result[split].column_names
            assert "label" in result[split].column_names

    def test_row_counts_preserved(self, raw_dd):
        tok, mdl = self._patched_load_model()
        with patch("repprobe.extraction.load_model", return_value=(tok, mdl)):
            result = generate_activations(
                raw_dd,
                model_family="Llama3", model_size="8B", model_type="instruct",
                layers=[1],
            )
        assert len(result["train"]) == 160
        assert len(result["test"])  == 40

    def test_act_tensor_shape(self, raw_dd):
        """Each activation row must be a vector of length hidden_dim."""
        hidden_dim = 16
        tok, mdl = self._patched_load_model(hidden_dim=hidden_dim)
        with patch("repprobe.extraction.load_model", return_value=(tok, mdl)):
            result = generate_activations(
                raw_dd,
                model_family="Llama3", model_size="8B", model_type="instruct",
                layers=[1],
            )
        row_0_acts = result["train"]["layer_1_acts"][0]
        assert len(row_0_acts) == hidden_dim

    def test_multiple_layers(self, raw_dd):
        tok, mdl = self._patched_load_model()
        with patch("repprobe.extraction.load_model", return_value=(tok, mdl)):
            result = generate_activations(
                raw_dd,
                model_family="Llama3", model_size="8B", model_type="instruct",
                layers=[0, 1, 2],
            )
        for layer in (0, 1, 2):
            assert f"layer_{layer}_acts" in result["train"].column_names
            assert f"layer_{layer}_acts" in result["test"].column_names

    def test_returns_flat_dataset_when_given_flat_dataset(self):
        """Flat Dataset in → flat Dataset out (no regression)."""
        flat = _make_split(n=20, k=2, seed=5)
        tok, mdl = self._patched_load_model()
        with patch("repprobe.extraction.load_model", return_value=(tok, mdl)):
            result = generate_activations(
                flat,
                model_family="Llama3", model_size="8B", model_type="instruct",
                layers=[1],
            )
        assert isinstance(result, Dataset)
        assert "layer_1_acts" in result.column_names

    def test_result_usable_as_repdataset(self, raw_dd):
        """
        Full end-to-end: DatasetDict → generate_activations → RepDataset →
        get_acts / get_labels works on both splits.
        """
        tok, mdl = self._patched_load_model(hidden_dim=16)
        with patch("repprobe.extraction.load_model", return_value=(tok, mdl)):
            acts_dd = generate_activations(
                raw_dd,
                model_family="Llama3", model_size="8B", model_type="instruct",
                layers=[1],
            )

        rep = RepDataset(hf_dataset=acts_dd)

        # Default split (train)
        train_acts   = rep.get_acts(1)
        train_labels = rep.get_labels()
        assert train_acts.shape   == (160, 16)
        assert train_labels.shape == (160,)

        # Explicitly switch to test split
        test_rep    = rep.get_split("test")
        test_acts   = test_rep.get_acts(1)
        test_labels = test_rep.get_labels()
        assert test_acts.shape   == (40, 16)
        assert test_labels.shape == (40,)
