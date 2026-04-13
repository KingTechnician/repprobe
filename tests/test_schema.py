"""Tests for LabelSchema."""
from __future__ import annotations

import pytest
from datasets import Dataset, ClassLabel, Features, Value, Sequence

from repprobe.schema import LabelSchema


class TestLabelSchemaConstruction:
    def test_basic_names(self):
        schema = LabelSchema(names=["False", "True"])
        assert schema.k == 2
        assert schema.is_binary is True

    def test_multiclass(self):
        names = ["A", "B", "C", "D", "E"]
        schema = LabelSchema(names=names)
        assert schema.k == 5
        assert schema.is_binary is False

    def test_single_class_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            LabelSchema(names=[])

    def test_ordinal_flag(self):
        schema = LabelSchema(names=["low", "mid", "high"], ordinal=True)
        assert schema.ordinal is True

    def test_polarity_flag(self):
        schema = LabelSchema(names=["neg", "pos"], has_polarity=True)
        assert schema.has_polarity is True

    def test_binary_convenience(self):
        schema = LabelSchema.binary()
        assert schema.names == ["False", "True"]
        assert schema.is_binary

    def test_binary_convenience_custom_names(self):
        schema = LabelSchema.binary("bad", "good", has_polarity=True)
        assert schema.names == ["bad", "good"]
        assert schema.has_polarity

    def test_repr(self):
        schema = LabelSchema(names=["a", "b"])
        r = repr(schema)
        assert "LabelSchema" in r
        assert "k=2" in r


class TestLabelSchemaFromHFDataset:
    def test_infer_from_ints(self):
        ds = Dataset.from_dict({"text": ["x"] * 6, "label": [0, 1, 2, 0, 1, 2]})
        schema = LabelSchema.from_hf_dataset(ds)
        assert schema.k == 3

    def test_from_class_label_feature(self):
        feat = Features({"text": Value("string"), "label": ClassLabel(names=["cat", "dog"])})
        ds = Dataset.from_dict({"text": ["a", "b"], "label": [0, 1]}, features=feat)
        schema = LabelSchema.from_hf_dataset(ds)
        assert schema.names == ["cat", "dog"]
        assert schema.k == 2

    def test_polarity_detected(self):
        ds = Dataset.from_dict({
            "text": ["x", "y"],
            "label": [0, 1],
            "polarity": [1, -1],
        })
        schema = LabelSchema.from_hf_dataset(ds)
        assert schema.has_polarity is True

    def test_no_polarity_without_column(self):
        ds = Dataset.from_dict({"text": ["x"], "label": [0]})
        schema = LabelSchema.from_hf_dataset(ds)
        assert schema.has_polarity is False

    def test_missing_label_col_raises(self):
        ds = Dataset.from_dict({"text": ["x", "y"]})
        with pytest.raises(ValueError, match="no 'label' column"):
            LabelSchema.from_hf_dataset(ds)

    def test_binary_inferred(self):
        ds = Dataset.from_dict({"text": ["a", "b", "c", "d"], "label": [0, 1, 0, 1]})
        schema = LabelSchema.from_hf_dataset(ds)
        assert schema.is_binary

    def test_k_matches_unique_labels(self):
        ds = Dataset.from_dict({"text": list("abcde"), "label": [0, 0, 1, 2, 2]})
        schema = LabelSchema.from_hf_dataset(ds)
        assert schema.k == 3
