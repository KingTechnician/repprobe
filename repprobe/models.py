"""
Model loading and forward-hook utilities.

Supports any HuggingFace CausalLM that stores transformer layers under
``model.model.layers`` (Llama, Mistral, Qwen, Gemma …).
"""
from __future__ import annotations

import configparser
import os
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = Path(__file__).parent.parent / "config.ini"


def _read_config(config_path: Optional[str] = None) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    path = config_path or str(_DEFAULT_CONFIG)
    cfg.read(path)
    return cfg


# ---------------------------------------------------------------------------
# Hook
# ---------------------------------------------------------------------------


class Hook:
    """
    Forward hook that captures the hidden-state tensor output of a layer.

    Handles both models that return a plain ``Tensor`` and those that return
    a ``tuple`` (e.g. ``(hidden_states, attention_weights, …)``).
    """

    def __init__(self) -> None:
        self.out: Optional[torch.Tensor] = None

    def __call__(self, module, module_inputs, module_outputs) -> None:  # noqa: ANN001
        if isinstance(module_outputs, tuple):
            self.out = module_outputs[0]
        else:
            self.out = module_outputs


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(
    model_family: str,
    model_size: str,
    model_type: str,
    device: str = "cpu",
    config_path: Optional[str] = None,
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """
    Load a model and tokenizer from local weights defined in ``config.ini``.

    Parameters
    ----------
    model_family : str
        Section name in ``config.ini``, e.g. ``"Llama3"``.
    model_size : str
        Size token, e.g. ``"8B"``.
    model_type : str
        ``"base"`` or ``"instruct"``.
    device : str
        PyTorch device string, e.g. ``"cpu"``, ``"cuda:0"``.
    config_path : str | None
        Override path to ``config.ini``.  Uses the package-level file by default.

    Returns
    -------
    tokenizer, model
    """
    cfg = _read_config(config_path)

    if model_family not in cfg:
        raise KeyError(
            f"Model family '{model_family}' not found in config. "
            f"Available: {list(cfg.sections())}"
        )

    weights_dir = cfg[model_family].get("weights_directory", "")
    subdir_key = f"{model_size}_{model_type}_subdir"
    if subdir_key not in cfg[model_family]:
        raise KeyError(
            f"Key '{subdir_key}' not found under [{model_family}] in config."
        )
    subdir = cfg[model_family][subdir_key]
    model_path = os.path.join(weights_dir, subdir) if weights_dir else subdir

    try:
        if model_family == "Llama2":
            tokenizer: PreTrainedTokenizerBase = LlamaTokenizer.from_pretrained(model_path)
            model: PreTrainedModel = LlamaForCausalLM.from_pretrained(model_path)
            tokenizer.bos_token = "<s>"
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)

        if model_family == "Gemma2":
            model = model.to(torch.bfloat16)
        else:
            model = model.half()  # fp16 for memory efficiency

        return tokenizer, model.to(device)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model '{model_path}': {exc}") from exc


def resolve_layer_index(model: PreTrainedModel, layer: int) -> int:
    """Resolve a possibly-negative layer index (e.g. -1 → last layer)."""
    num_layers = len(model.model.layers)
    if layer < 0:
        return num_layers + layer
    return layer
