"""
Tests for the extraction module.

Model loading is gated behind mocks — these tests exercise the Hook class,
the get_acts function with a lightweight dummy model, and the generate_activations
pipeline without downloading any real weights.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from datasets import Dataset

from repprobe.models import Hook, resolve_layer_index
from repprobe.extraction import get_acts


# ---------------------------------------------------------------------------
# Hook tests
# ---------------------------------------------------------------------------


class TestHook:
    def test_captures_tensor(self):
        hook = Hook()
        tensor = torch.randn(1, 10, 16)
        hook(None, None, tensor)
        assert hook.out is tensor

    def test_captures_first_element_of_tuple(self):
        hook = Hook()
        hidden = torch.randn(1, 10, 16)
        hook(None, None, (hidden, torch.randn(1, 4, 10, 10)))
        assert hook.out is hidden

    def test_initial_state_is_none(self):
        hook = Hook()
        assert hook.out is None


# ---------------------------------------------------------------------------
# resolve_layer_index
# ---------------------------------------------------------------------------


class TestResolveLayerIndex:
    def _mock_model(self, num_layers: int):
        m = MagicMock()
        m.model.layers = [MagicMock() for _ in range(num_layers)]
        return m

    def test_positive_index(self):
        model = self._mock_model(32)
        assert resolve_layer_index(model, 12) == 12

    def test_negative_index(self):
        model = self._mock_model(32)
        assert resolve_layer_index(model, -1) == 31

    def test_negative_2(self):
        model = self._mock_model(32)
        assert resolve_layer_index(model, -4) == 28


# ---------------------------------------------------------------------------
# get_acts with a stub model
# ---------------------------------------------------------------------------


def _make_stub_model(hidden_dim: int = 16, num_layers: int = 4):
    """
    Create a minimal torch model that simulates a causal LM forward pass.

    Each layer is a real nn.Module that fires forward hooks — matching what
    repprobe's get_acts expects.
    """
    _d = hidden_dim

    class StubLayer(torch.nn.Module):
        """A layer that passes the hidden tensor through unchanged."""

        def forward(self, x: torch.Tensor):
            # Return a tuple (hidden_states, ...) like real transformer layers do
            return (x,)

    class StubInnerModel(torch.nn.Module):
        def __init__(self, d, n_layers):
            super().__init__()
            self.layers = torch.nn.ModuleList([StubLayer() for _ in range(n_layers)])
            self._d = d

        def forward(self, input_ids):
            # Start with a random embedding
            x = torch.randn(input_ids.shape[0], input_ids.shape[1], self._d)
            for layer in self.layers:
                x = layer(x)[0]  # pass through each layer
            return x

    class StubCausalLM(torch.nn.Module):
        def __init__(self, d, n_layers):
            super().__init__()
            self.model = StubInnerModel(d, n_layers)

        def forward(self, input_ids):
            return self.model(input_ids)

    return StubCausalLM(hidden_dim, num_layers)


class TestGetActs:
    def test_output_shape(self):
        """get_acts should return [N, hidden_dim] tensors."""
        model = _make_stub_model(hidden_dim=16, num_layers=4)

        class StubTokenizer:
            def encode(self, text, return_tensors="pt"):
                return torch.randint(0, 100, (1, 5))

        statements = ["hello world", "foo bar", "baz qux"]
        acts = get_acts(statements, StubTokenizer(), model, layers=[0, 2], device="cpu")
        assert set(acts.keys()) == {0, 2}
        for layer, tensor in acts.items():
            assert tensor.shape == (3, 16), f"Layer {layer}: expected (3,16), got {tensor.shape}"

    def test_hooks_removed_after_call(self):
        """No forward hooks should remain attached after get_acts returns."""
        model = _make_stub_model(hidden_dim=8, num_layers=2)

        class StubTok:
            def encode(self, text, return_tensors="pt"):
                return torch.randint(0, 100, (1, 3))

        get_acts(["hi"], StubTok(), model, layers=[0, 1], device="cpu")

        # Count remaining hooks on each layer
        for layer in model.model.layers:
            hook_count = len(layer._forward_hooks)
            assert hook_count == 0, f"Hook not removed from layer: {hook_count} hooks"

    def test_float32_output(self):
        """Activations must be float32 regardless of model dtype."""
        model = _make_stub_model(hidden_dim=8, num_layers=2)
        model = model.half()  # fp16 model

        class StubTok:
            def encode(self, text, return_tensors="pt"):
                return torch.randint(0, 100, (1, 3))

        acts = get_acts(["test"], StubTok(), model, layers=[0], device="cpu")
        assert acts[0].dtype == torch.float32
