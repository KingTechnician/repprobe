"""Tests for RepDataset."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch
from datasets import Dataset

from repprobe import RepDataset, LabelSchema


class TestRepDatasetConstruction:
    def test_from_hf_dataset(self, multiclass_hf_dataset):
        ds = RepDataset(hf_dataset=multiclass_hf_dataset)
        assert len(ds) == 300

    def test_missing_both_args_raises(self):
        with pytest.raises(ValueError, match="Provide either"):
            RepDataset()

    def test_missing_text_col_raises(self):
        hf = Dataset.from_dict({"label": [0, 1]})
        with pytest.raises(ValueError, match="missing required columns"):
            RepDataset(hf_dataset=hf)

    def test_missing_label_col_raises(self):
        hf = Dataset.from_dict({"text": ["a", "b"]})
        with pytest.raises(ValueError, match="missing required columns"):
            RepDataset(hf_dataset=hf)

    def test_repr(self, multiclass_rep_dataset):
        r = repr(multiclass_rep_dataset)
        assert "RepDataset" in r
        assert "rows=300" in r


class TestRepDatasetSchema:
    def test_schema_auto_detected(self, multiclass_rep_dataset):
        schema = multiclass_rep_dataset.schema
        assert isinstance(schema, LabelSchema)
        assert schema.k == 5

    def test_schema_cached(self, multiclass_rep_dataset):
        s1 = multiclass_rep_dataset.schema
        s2 = multiclass_rep_dataset.schema
        assert s1 is s2  # same object (cached)


class TestRepDatasetActivations:
    def test_activation_layers_detected(self, multiclass_rep_dataset):
        layers = multiclass_rep_dataset.activation_layers
        assert 8 in layers
        assert 12 in layers
        assert 16 in layers

    def test_get_acts_shape(self, multiclass_rep_dataset):
        acts = multiclass_rep_dataset.get_acts(12)
        assert acts.ndim == 2
        assert acts.shape[0] == 300
        assert acts.shape[1] == 32

    def test_get_acts_missing_layer_raises(self, multiclass_rep_dataset):
        with pytest.raises(KeyError, match="No activation column"):
            multiclass_rep_dataset.get_acts(99)

    def test_get_labels_shape(self, multiclass_rep_dataset):
        labels = multiclass_rep_dataset.get_labels()
        assert labels.shape == (300,)
        assert labels.dtype == torch.long

    def test_get_labels_values_in_range(self, multiclass_rep_dataset):
        labels = multiclass_rep_dataset.get_labels()
        assert labels.min() >= 0
        assert labels.max() <= 4


class TestRepDatasetSplits:
    def test_stratified_split_sizes(self, multiclass_rep_dataset):
        train, test = multiclass_rep_dataset.stratified_split(train_frac=0.8)
        assert len(train) + len(test) == 300
        assert len(train) > len(test)

    def test_stratified_split_schema_propagated(self, multiclass_rep_dataset):
        # Prime the cache first
        _ = multiclass_rep_dataset.schema
        train, test = multiclass_rep_dataset.stratified_split()
        assert train.schema.k == 5
        assert test.schema.k == 5

    def test_split_by_column(self, multiclass_with_objective_hf_dataset):
        ds = RepDataset(hf_dataset=multiclass_with_objective_hf_dataset)
        train_vals = ["obj_0", "obj_1", "obj_2"]
        test_vals = ["obj_3", "obj_4"]
        train, test = ds.split_by_column("objective", train_vals, test_vals)
        assert len(train) > 0
        assert len(test) > 0

    def test_split_by_missing_column_raises(self, multiclass_rep_dataset):
        with pytest.raises(ValueError, match="Column 'nonexistent'"):
            multiclass_rep_dataset.split_by_column("nonexistent", ["a"], ["b"])


class TestRepDatasetFromCSV:
    def test_from_csv_basic(self, tmp_path):
        csv = tmp_path / "data.csv"
        df = pd.DataFrame({
            "statement": ["Alice is tall", "Bob is short"],
            "label": [1, 0],
        })
        df.to_csv(csv, index=False)
        ds = RepDataset.from_csv(str(csv), text_col="statement")
        assert len(ds) == 2
        assert "text" in ds.hf_dataset.column_names

    def test_from_csv_already_named_text(self, tmp_path):
        csv = tmp_path / "data.csv"
        df = pd.DataFrame({"text": ["a", "b"], "label": [0, 1]})
        df.to_csv(csv, index=False)
        ds = RepDataset.from_csv(str(csv))
        assert len(ds) == 2
