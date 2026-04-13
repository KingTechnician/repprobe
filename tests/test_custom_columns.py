"""
Tests for RepDataset custom text_col / label_col support.

Covers:
- Flat Dataset with custom text_col only
- Flat Dataset with custom label_col only
- Flat Dataset with both custom columns
- DatasetDict with custom text_col only
- DatasetDict with custom label_col only
- DatasetDict with both custom columns (train + test splits)
- Default column names still work (no rename needed)
- Error when specified text_col is missing
- Error when specified label_col is missing
- from_dataset_dict classmethod propagates text_col / label_col
- get_labels() and get_acts() work correctly after renaming
- stratified_split() works correctly after renaming
"""
from __future__ import annotations

import pytest
import torch
from datasets import Dataset, DatasetDict

from repprobe import RepDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_flat(*, text_col="text", label_col="label", n=20):
    """Return a flat Dataset with the given column names."""
    return Dataset.from_dict(
        {
            text_col: [f"sentence {i}" for i in range(n)],
            label_col: [i % 2 for i in range(n)],
        }
    )


def _make_dict(*, text_col="text", label_col="label", n_train=20, n_test=10):
    """Return a DatasetDict with train/test splits and the given column names."""
    return DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    text_col: [f"train sentence {i}" for i in range(n_train)],
                    label_col: [i % 2 for i in range(n_train)],
                }
            ),
            "test": Dataset.from_dict(
                {
                    text_col: [f"test sentence {i}" for i in range(n_test)],
                    label_col: [i % 2 for i in range(n_test)],
                }
            ),
        }
    )


# ---------------------------------------------------------------------------
# Flat Dataset — custom columns
# ---------------------------------------------------------------------------

class TestFlatCustomColumns:
    def test_default_columns_no_rename(self):
        """Default text/label names work without any rename."""
        ds = _make_flat(text_col="text", label_col="label")
        rds = RepDataset(hf_dataset=ds)
        assert "text" in rds.hf_dataset.column_names
        assert "label" in rds.hf_dataset.column_names

    def test_custom_text_col(self):
        """Custom text column is renamed to 'text' internally."""
        ds = _make_flat(text_col="sentence", label_col="label")
        rds = RepDataset(hf_dataset=ds, text_col="sentence")
        assert "text" in rds.hf_dataset.column_names
        # original name should no longer be present
        assert "sentence" not in rds.hf_dataset.column_names

    def test_custom_label_col(self):
        """Custom label column is renamed to 'label' internally."""
        ds = _make_flat(text_col="text", label_col="target")
        rds = RepDataset(hf_dataset=ds, label_col="target")
        assert "label" in rds.hf_dataset.column_names
        assert "target" not in rds.hf_dataset.column_names

    def test_both_custom_columns(self):
        """Both custom columns are renamed correctly."""
        ds = _make_flat(text_col="utterance", label_col="category")
        rds = RepDataset(hf_dataset=ds, text_col="utterance", label_col="category")
        assert "text" in rds.hf_dataset.column_names
        assert "label" in rds.hf_dataset.column_names
        assert "utterance" not in rds.hf_dataset.column_names
        assert "category" not in rds.hf_dataset.column_names

    def test_custom_columns_row_count_preserved(self):
        """Renaming does not change the row count."""
        ds = _make_flat(text_col="utterance", label_col="category", n=30)
        rds = RepDataset(hf_dataset=ds, text_col="utterance", label_col="category")
        assert len(rds) == 30

    def test_custom_columns_text_values_preserved(self):
        """Text values are preserved after renaming."""
        ds = _make_flat(text_col="utterance", label_col="category", n=5)
        rds = RepDataset(hf_dataset=ds, text_col="utterance", label_col="category")
        assert rds.hf_dataset["text"] == [f"sentence {i}" for i in range(5)]

    def test_custom_columns_label_values_preserved(self):
        """Label values are preserved after renaming."""
        ds = _make_flat(text_col="utterance", label_col="category", n=6)
        rds = RepDataset(hf_dataset=ds, text_col="utterance", label_col="category")
        expected = torch.tensor([i % 2 for i in range(6)], dtype=torch.long)
        assert torch.equal(rds.get_labels(), expected)

    def test_missing_text_col_raises(self):
        """Passing a text_col that doesn't exist should raise ValueError."""
        ds = _make_flat(text_col="text", label_col="label")
        with pytest.raises(ValueError, match="missing required columns"):
            RepDataset(hf_dataset=ds, text_col="nonexistent")

    def test_missing_label_col_raises(self):
        """Passing a label_col that doesn't exist should raise ValueError."""
        ds = _make_flat(text_col="text", label_col="label")
        with pytest.raises(ValueError, match="missing required columns"):
            RepDataset(hf_dataset=ds, label_col="nonexistent")

    def test_stratified_split_after_rename(self):
        """stratified_split works correctly after column renaming."""
        ds = _make_flat(text_col="utterance", label_col="category", n=20)
        rds = RepDataset(hf_dataset=ds, text_col="utterance", label_col="category")
        train_rds, test_rds = rds.stratified_split(train_frac=0.8, seed=0)
        assert len(train_rds) + len(test_rds) == 20
        # Both sub-datasets must also have the canonical column names
        assert "text" in train_rds.hf_dataset.column_names
        assert "label" in test_rds.hf_dataset.column_names


# ---------------------------------------------------------------------------
# DatasetDict — custom columns
# ---------------------------------------------------------------------------

class TestDatasetDictCustomColumns:
    def test_default_columns_dict(self):
        """DatasetDict with default column names works without rename."""
        dsd = _make_dict()
        rds = RepDataset(hf_dataset=dsd)
        for split_name in ["train", "test"]:
            split_ds = rds.get_split(split_name).hf_dataset
            assert "text" in split_ds.column_names
            assert "label" in split_ds.column_names

    def test_custom_text_col_dict(self):
        """DatasetDict: custom text column is renamed in every split."""
        dsd = _make_dict(text_col="sentence", label_col="label")
        rds = RepDataset(hf_dataset=dsd, text_col="sentence")
        for split_name in ["train", "test"]:
            split_ds = rds.get_split(split_name).hf_dataset
            assert "text" in split_ds.column_names
            assert "sentence" not in split_ds.column_names

    def test_custom_label_col_dict(self):
        """DatasetDict: custom label column is renamed in every split."""
        dsd = _make_dict(text_col="text", label_col="target")
        rds = RepDataset(hf_dataset=dsd, label_col="target")
        for split_name in ["train", "test"]:
            split_ds = rds.get_split(split_name).hf_dataset
            assert "label" in split_ds.column_names
            assert "target" not in split_ds.column_names

    def test_both_custom_columns_dict(self):
        """DatasetDict: both custom columns renamed in every split."""
        dsd = _make_dict(text_col="utterance", label_col="category")
        rds = RepDataset(
            hf_dataset=dsd, text_col="utterance", label_col="category"
        )
        assert rds.is_dict
        assert set(rds.splits) == {"train", "test"}
        for split_name in ["train", "test"]:
            split_ds = rds.get_split(split_name).hf_dataset
            assert "text" in split_ds.column_names
            assert "label" in split_ds.column_names
            assert "utterance" not in split_ds.column_names
            assert "category" not in split_ds.column_names

    def test_dict_row_counts_preserved(self):
        """Row counts are preserved after renaming in DatasetDict."""
        dsd = _make_dict(
            text_col="utterance", label_col="category", n_train=20, n_test=10
        )
        rds = RepDataset(
            hf_dataset=dsd, text_col="utterance", label_col="category"
        )
        assert len(rds.get_split("train")) == 20
        assert len(rds.get_split("test")) == 10

    def test_dict_text_values_preserved(self):
        """Text content is preserved across splits after rename."""
        dsd = _make_dict(text_col="utterance", label_col="category", n_train=3, n_test=2)
        rds = RepDataset(
            hf_dataset=dsd, text_col="utterance", label_col="category"
        )
        assert rds.get_split("train").hf_dataset["text"] == [
            f"train sentence {i}" for i in range(3)
        ]
        assert rds.get_split("test").hf_dataset["text"] == [
            f"test sentence {i}" for i in range(2)
        ]

    def test_dict_label_values_preserved(self):
        """Label values are preserved across splits after rename."""
        dsd = _make_dict(text_col="utterance", label_col="category", n_train=4, n_test=4)
        rds = RepDataset(
            hf_dataset=dsd, text_col="utterance", label_col="category"
        )
        # Default split is 'train'
        expected = torch.tensor([i % 2 for i in range(4)], dtype=torch.long)
        assert torch.equal(rds.get_labels(), expected)

    def test_dict_get_labels_on_test_split(self):
        """get_labels() on the test split works after rename."""
        dsd = _make_dict(text_col="utterance", label_col="category", n_train=4, n_test=4)
        rds = RepDataset(
            hf_dataset=dsd, text_col="utterance", label_col="category"
        )
        test_rds = rds.get_split("test")
        expected = torch.tensor([i % 2 for i in range(4)], dtype=torch.long)
        assert torch.equal(test_rds.get_labels(), expected)

    def test_missing_text_col_dict_raises(self):
        """DatasetDict: passing nonexistent text_col raises ValueError."""
        dsd = _make_dict()
        with pytest.raises(ValueError, match="missing required columns"):
            RepDataset(hf_dataset=dsd, text_col="nonexistent")

    def test_missing_label_col_dict_raises(self):
        """DatasetDict: passing nonexistent label_col raises ValueError."""
        dsd = _make_dict()
        with pytest.raises(ValueError, match="missing required columns"):
            RepDataset(hf_dataset=dsd, label_col="nonexistent")

    def test_from_dataset_dict_custom_columns(self):
        """from_dataset_dict classmethod propagates text_col / label_col."""
        dsd = _make_dict(text_col="utterance", label_col="category")
        rds = RepDataset.from_dataset_dict(
            dsd, text_col="utterance", label_col="category"
        )
        assert rds.is_dict
        for split_name in ["train", "test"]:
            cols = rds.get_split(split_name).hf_dataset.column_names
            assert "text" in cols
            assert "label" in cols

    def test_from_dataset_dict_default_split_preserved(self):
        """from_dataset_dict: default_split kwarg is honoured alongside custom cols."""
        dsd = _make_dict(text_col="utterance", label_col="category")
        rds = RepDataset.from_dataset_dict(
            dsd,
            default_split="test",
            text_col="utterance",
            label_col="category",
        )
        assert len(rds) == 10  # test split has 10 rows
