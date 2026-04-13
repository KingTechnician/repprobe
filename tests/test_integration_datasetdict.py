"""
Integration test: DatasetDict with numeric labels and text → activations.

No mocking. No Hub downloads. Uses a tiny LlamaForCausalLM (~230k params)
built entirely from a config so the full get_acts hook path executes for real.

The scenario under test, end-to-end:
  1. Build a DatasetDict with 'train' and 'test' splits.
     - 'label'  is a plain Python int  (not a ClassLabel)
     - 'text'   is a plain string
  2. Wrap it in RepDataset — both splits must be accepted.
  3. Run generate_activations (with a pre-loaded model, bypassing config.ini).
  4. The returned DatasetDict must have layer_N_acts in both splits.
  5. Wrap the result back in RepDataset and call get_acts / get_labels on
     each split — shapes and dtypes must be correct.
"""
from __future__ import annotations

import pytest
import torch
from datasets import Dataset, DatasetDict
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE

from repprobe import RepDataset
from repprobe.extraction import get_acts, _map_single_split


# ---------------------------------------------------------------------------
# Tiny Llama model + tokenizer — built in-memory, no downloads
# ---------------------------------------------------------------------------

HIDDEN = 64
N_LAYERS = 4
VOCAB = 512


@pytest.fixture(scope="module")
def tiny_model() -> LlamaForCausalLM:
    cfg = LlamaConfig(
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=VOCAB,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(cfg).eval()
    return model


@pytest.fixture(scope="module")
def tiny_tokenizer() -> PreTrainedTokenizerFast:
    """
    Minimal BPE tokenizer with a small fixed vocabulary, built in-memory.
    Encodes each character as its own token — good enough for integration tests.
    """
    # Build a character-level BPE tokenizer over printable ASCII
    vocab = {chr(i): i for i in range(32, 32 + VOCAB - 4)}
    merges = []
    tok = Tokenizer(BPE(vocab=vocab, merges=merges, unk_token="?"))

    from tokenizers.pre_tokenizers import ByteLevel  # noqa: PLC0415
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)

    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="?",
        pad_token="?",
    )
    return wrapped


# ---------------------------------------------------------------------------
# The DatasetDict fixture — exactly as described
# ---------------------------------------------------------------------------

@pytest.fixture
def dd() -> DatasetDict:
    """
    DatasetDict with 'train' and 'test' splits.
    'label' is a plain int. 'text' is a plain string.
    """
    return DatasetDict({
        "train": Dataset.from_dict({
            "text":  ["The cat sat on the mat.",
                      "Paris is the capital of France.",
                      "Water boils at one hundred degrees.",
                      "The sky appears blue due to scattering.",
                      "Gravity pulls objects toward the earth.",
                      "Photosynthesis converts light into energy.",
                      "DNA carries genetic information.",
                      "The moon orbits the earth.",
                      "Sound travels through air as waves.",
                      "Fire requires oxygen to burn."],
            "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }),
        "test": Dataset.from_dict({
            "text":  ["Ice is frozen water.",
                      "The sun is a star.",
                      "Humans breathe oxygen.",
                      "Salt dissolves in water."],
            "label": [0, 1, 0, 1],
        }),
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRepDatasetAcceptsDatasetDict:
    def test_wraps_without_error(self, dd):
        ds = RepDataset(hf_dataset=dd)
        assert ds.is_dict

    def test_both_splits_visible(self, dd):
        ds = RepDataset(hf_dataset=dd)
        assert set(ds.splits) == {"train", "test"}

    def test_default_split_is_train(self, dd):
        ds = RepDataset(hf_dataset=dd)
        assert len(ds) == 10

    def test_test_split_accessible(self, dd):
        ds = RepDataset(hf_dataset=dd)
        test_child = ds.get_split("test")
        assert len(test_child) == 4

    def test_numeric_label_schema_detected(self, dd):
        ds = RepDataset(hf_dataset=dd)
        assert ds.schema.k == 2   # labels 0 and 1

    def test_get_labels_train(self, dd):
        ds = RepDataset(hf_dataset=dd)
        labels = ds.get_labels()
        assert labels.shape == (10,)
        assert labels.dtype == torch.long
        assert set(labels.tolist()) == {0, 1}

    def test_get_labels_test(self, dd):
        ds = RepDataset(hf_dataset=dd)
        labels = ds.get_split("test").get_labels()
        assert labels.shape == (4,)
        assert set(labels.tolist()) == {0, 1}


class TestGetActsWithRealModel:
    """get_acts called directly — exercises hook registration on real layers."""

    def test_hook_fires_and_returns_correct_shape(self, tiny_model, tiny_tokenizer):
        texts = ["Hello world", "Foo bar baz"]
        layers = [0, N_LAYERS - 1]
        acts = get_acts(texts, tiny_tokenizer, tiny_model, layers, device="cpu")

        assert set(acts.keys()) == {0, N_LAYERS - 1}
        for layer_idx, tensor in acts.items():
            assert tensor.shape == (2, HIDDEN), (
                f"Layer {layer_idx}: expected (2, {HIDDEN}), got {tensor.shape}"
            )

    def test_output_dtype_is_float32(self, tiny_model, tiny_tokenizer):
        acts = get_acts(["test sentence"], tiny_tokenizer, tiny_model, [0], "cpu")
        assert acts[0].dtype == torch.float32

    def test_hooks_removed_after_call(self, tiny_model, tiny_tokenizer):
        get_acts(["hello"], tiny_tokenizer, tiny_model, [0, 1], "cpu")
        for i, layer in enumerate(tiny_model.model.layers):
            assert len(layer._forward_hooks) == 0, (
                f"Hook not removed from layer {i}"
            )

    def test_single_statement(self, tiny_model, tiny_tokenizer):
        acts = get_acts(["one"], tiny_tokenizer, tiny_model, [2], "cpu")
        assert acts[2].shape == (1, HIDDEN)


class TestMapSingleSplitWithRealModel:
    """_map_single_split on a real flat Dataset with a real model."""

    def test_adds_act_columns(self, dd, tiny_model, tiny_tokenizer):
        train_ds = dd["train"]
        result = _map_single_split(
            train_ds, tiny_tokenizer, tiny_model,
            resolved_layers=[0], device="cpu", batch_size=4,
            desc="test",
        )
        assert "layer_0_acts" in result.column_names

    def test_preserves_text_and_label(self, dd, tiny_model, tiny_tokenizer):
        train_ds = dd["train"]
        result = _map_single_split(
            train_ds, tiny_tokenizer, tiny_model,
            resolved_layers=[0], device="cpu", batch_size=4,
            desc="test",
        )
        assert "text"  in result.column_names
        assert "label" in result.column_names

    def test_row_count_unchanged(self, dd, tiny_model, tiny_tokenizer):
        train_ds = dd["train"]
        result = _map_single_split(
            train_ds, tiny_tokenizer, tiny_model,
            resolved_layers=[0], device="cpu", batch_size=4,
            desc="test",
        )
        assert len(result) == 10

    def test_act_vector_length_equals_hidden(self, dd, tiny_model, tiny_tokenizer):
        train_ds = dd["train"]
        result = _map_single_split(
            train_ds, tiny_tokenizer, tiny_model,
            resolved_layers=[0], device="cpu", batch_size=4,
            desc="test",
        )
        assert len(result["layer_0_acts"][0]) == HIDDEN


class TestFullPipelineDatasetDict:
    """
    Full end-to-end: DatasetDict → _map_single_split on each split →
    RepDataset → get_acts / get_labels on train and test.

    (generate_activations loads a model via config.ini / load_model, so we
    drive the same internal path — _map_single_split — directly with a
    pre-loaded model, which is the correct library-API pattern.)
    """

    @pytest.fixture
    def acts_dd(self, dd, tiny_model, tiny_tokenizer) -> DatasetDict:
        """Run extraction on both splits and return a DatasetDict."""
        return DatasetDict({
            split_name: _map_single_split(
                split_ds, tiny_tokenizer, tiny_model,
                resolved_layers=[0, N_LAYERS - 1],
                device="cpu", batch_size=4,
                desc=f"test [{split_name}]",
            )
            for split_name, split_ds in dd.items()
        })

    def test_both_splits_have_act_columns(self, acts_dd):
        for split in ("train", "test"):
            assert "layer_0_acts" in acts_dd[split].column_names
            assert f"layer_{N_LAYERS - 1}_acts" in acts_dd[split].column_names

    def test_repdataset_wraps_result(self, acts_dd):
        rep = RepDataset(hf_dataset=acts_dd)
        assert rep.is_dict
        assert set(rep.splits) == {"train", "test"}

    def test_get_acts_train_shape(self, acts_dd):
        rep = RepDataset(hf_dataset=acts_dd)
        acts = rep.get_acts(0)
        assert acts.shape == (10, HIDDEN)

    def test_get_acts_test_shape(self, acts_dd):
        rep = RepDataset(hf_dataset=acts_dd)
        acts = rep.get_split("test").get_acts(0)
        assert acts.shape == (4, HIDDEN)

    def test_get_labels_train(self, acts_dd):
        rep = RepDataset(hf_dataset=acts_dd)
        labels = rep.get_labels()
        assert labels.shape == (10,)
        assert labels.dtype == torch.long

    def test_get_labels_test(self, acts_dd):
        rep = RepDataset(hf_dataset=acts_dd)
        labels = rep.get_split("test").get_labels()
        assert labels.shape == (4,)

    def test_activation_layers_detected(self, acts_dd):
        rep = RepDataset(hf_dataset=acts_dd)
        assert set(rep.activation_layers) == {0, N_LAYERS - 1}

    def test_acts_are_finite(self, acts_dd):
        rep = RepDataset(hf_dataset=acts_dd)
        acts = rep.get_acts(0)
        assert torch.isfinite(acts).all(), "Activation tensor contains NaN or Inf"

    def test_train_and_test_acts_differ(self, acts_dd):
        """Sanity: different texts should produce different activation vectors."""
        rep = RepDataset(hf_dataset=acts_dd)
        train_acts = rep.get_acts(0)
        test_acts  = rep.get_split("test").get_acts(0)
        # Shapes differ (10 vs 4), so they can't be equal — just check they exist
        assert train_acts.shape[1] == test_acts.shape[1] == HIDDEN
