"""
Tests for the repprobe CLI — argument parsing and _cmd_generate column renaming.

Strategy: we don't run a real model (no weights available in CI).  Instead we
patch ``generate_activations`` and ``load_dataset`` so we can verify:

- ``--text_col`` / ``--label_col`` are parsed and propagated correctly
- The dataset passed to generate_activations already has canonical column names
- DatasetDict is passed through intact (no concatenation) when --split is omitted
- A single split is selected when --split is given
- Defaults (text_col="text", label_col="label") leave columns unchanged
- Missing column raises a clean ValueError before the model is touched
"""
from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset, DatasetDict

from repprobe.cli import build_parser, _cmd_generate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_dict(*, text_col="text", label_col="label", n=10):
    """Simulate what load_dataset returns — a DatasetDict with custom cols."""
    split_data = {
        text_col: [f"sentence {i}" for i in range(n)],
        label_col: [i % 2 for i in range(n)],
    }
    return DatasetDict(
        {
            "train": Dataset.from_dict(split_data),
            "test": Dataset.from_dict({
                text_col: [f"test {i}" for i in range(n // 2)],
                label_col: [i % 2 for i in range(n // 2)],
            }),
        }
    )


def _make_args(**kwargs) -> argparse.Namespace:
    """Return a Namespace with all generate-subcommand fields set to safe defaults."""
    defaults = dict(
        dataset="fake/dataset",
        split=None,
        model_family="Llama3",
        model_size="8B",
        model_type="instruct",
        layers=["8"],
        device="cpu",
        batch_size=4,
        config=None,
        push_to_hub=None,
        output_dir=None,
        text_col="text",
        label_col="label",
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _fake_generate_activations(**kwargs):
    """Stub that returns the dataset unchanged (adds no act columns)."""
    ds = kwargs["dataset"]
    if isinstance(ds, DatasetDict):
        return ds
    return ds


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

class TestParser:
    def test_generate_text_col_default(self):
        parser = build_parser()
        args = parser.parse_args([
            "generate",
            "--dataset", "org/ds",
            "--model_family", "Llama3",
            "--model_size", "8B",
            "--model_type", "instruct",
        ])
        assert args.text_col == "text"
        assert args.label_col == "label"

    def test_generate_text_col_custom(self):
        parser = build_parser()
        args = parser.parse_args([
            "generate",
            "--dataset", "org/ds",
            "--model_family", "Llama3",
            "--model_size", "8B",
            "--model_type", "instruct",
            "--text_col", "utterance",
            "--label_col", "category",
        ])
        assert args.text_col == "utterance"
        assert args.label_col == "category"

    def test_calibrate_has_no_text_col(self):
        """calibrate sub-command intentionally has no --text_col."""
        parser = build_parser()
        args = parser.parse_args([
            "calibrate",
            "--dataset", "org/ds",
        ])
        assert not hasattr(args, "text_col")

    def test_probe_has_no_text_col(self):
        """probe sub-command intentionally has no --text_col."""
        parser = build_parser()
        args = parser.parse_args([
            "probe",
            "--dataset", "org/ds",
            "--layer", "8",
        ])
        assert not hasattr(args, "text_col")


# ---------------------------------------------------------------------------
# _cmd_generate — column renaming
# ---------------------------------------------------------------------------

class TestCmdGenerateColumnRenaming:
    def _run(self, raw_dict, args):
        """Patch load_dataset + generate_activations and run _cmd_generate."""
        captured = {}

        def fake_ga(**kwargs):
            captured["dataset"] = kwargs["dataset"]
            ds = kwargs["dataset"]
            return ds if isinstance(ds, DatasetDict) else ds

        # Both symbols are imported lazily inside _cmd_generate, so patch at source
        with patch("datasets.load_dataset", return_value=raw_dict), \
             patch("repprobe.extraction.generate_activations", side_effect=fake_ga), \
             patch("repprobe.cli.generate_activations", side_effect=fake_ga, create=True):
            _cmd_generate(args)

        return captured["dataset"]

    def test_default_cols_unchanged(self):
        """Default text/label columns pass through without modification."""
        raw = _make_raw_dict(text_col="text", label_col="label")
        args = _make_args()
        ds = self._run(raw, args)
        assert isinstance(ds, DatasetDict)
        assert "text" in ds["train"].column_names
        assert "label" in ds["train"].column_names

    def test_custom_text_col_renamed(self):
        """--text_col renames the column before generate_activations sees it."""
        raw = _make_raw_dict(text_col="utterance", label_col="label")
        args = _make_args(text_col="utterance")
        ds = self._run(raw, args)
        assert "text" in ds["train"].column_names
        assert "utterance" not in ds["train"].column_names

    def test_custom_label_col_renamed(self):
        """--label_col renames the column before generate_activations sees it."""
        raw = _make_raw_dict(text_col="text", label_col="category")
        args = _make_args(label_col="category")
        ds = self._run(raw, args)
        assert "label" in ds["train"].column_names
        assert "category" not in ds["train"].column_names

    def test_both_custom_cols_renamed(self):
        """Both custom columns are renamed correctly."""
        raw = _make_raw_dict(text_col="utterance", label_col="category")
        args = _make_args(text_col="utterance", label_col="category")
        ds = self._run(raw, args)
        for split in ("train", "test"):
            assert "text" in ds[split].column_names
            assert "label" in ds[split].column_names
            assert "utterance" not in ds[split].column_names
            assert "category" not in ds[split].column_names

    def test_datasetdict_passed_intact_without_split(self):
        """Without --split the full DatasetDict reaches generate_activations."""
        raw = _make_raw_dict()
        args = _make_args(split=None)
        ds = self._run(raw, args)
        assert isinstance(ds, DatasetDict)
        assert set(ds.keys()) == {"train", "test"}

    def test_single_split_selected(self):
        """--split selects a single Dataset (not a DatasetDict)."""
        raw = _make_raw_dict()
        args = _make_args(split="train")
        ds = self._run(raw, args)
        assert isinstance(ds, Dataset)

    def test_missing_text_col_raises(self):
        """generate raises cleanly when --text_col is not in the dataset."""
        raw = _make_raw_dict(text_col="text", label_col="label")
        args = _make_args(text_col="nonexistent")
        with patch("datasets.load_dataset", return_value=raw):
            with pytest.raises(ValueError, match="missing required columns"):
                _cmd_generate(args)

    def test_missing_label_col_raises(self):
        """generate raises cleanly when --label_col is not in the dataset."""
        raw = _make_raw_dict(text_col="text", label_col="label")
        args = _make_args(label_col="nonexistent")
        with patch("datasets.load_dataset", return_value=raw):
            with pytest.raises(ValueError, match="missing required columns"):
                _cmd_generate(args)
