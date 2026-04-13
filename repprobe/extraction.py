"""
Activation extraction from HuggingFace-compatible causal language models.

Core primitives
---------------
* :func:`get_acts`  – extract last-token hidden states for a list of strings
* :func:`generate_activations`  – map activations over a full HF dataset and
  optionally push the result to the Hub
"""
from __future__ import annotations

import gc
from typing import Optional

import torch
from tqdm import tqdm

from .models import Hook, load_model, resolve_layer_index

try:
    from datasets import Dataset  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install the 'datasets' package.") from exc


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


def get_acts(
    statements: list[str],
    tokenizer,
    model,
    layers: list[int],
    device: str = "cpu",
) -> dict[int, torch.Tensor]:
    """
    Extract last-token hidden-state activations for a list of statements.

    Parameters
    ----------
    statements : list[str]
        Input text strings.
    tokenizer :
        HuggingFace tokenizer.
    model :
        HuggingFace causal LM.
    layers : list[int]
        Layer indices to hook (non-negative, already resolved).
    device : str
        PyTorch device string.

    Returns
    -------
    dict[int, Tensor]
        ``{layer_index: tensor of shape [N, hidden_dim]}``
    """
    # Attach hooks
    hooks: list[Hook] = []
    handles = []
    for layer in layers:
        hook = Hook()
        handle = model.model.layers[layer].register_forward_hook(hook)
        hooks.append(hook)
        handles.append(handle)

    acts: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}

    with torch.no_grad():
        for statement in tqdm(statements, leave=False):
            input_ids = tokenizer.encode(statement, return_tensors="pt").to(device)
            model(input_ids)
            for layer, hook in zip(layers, hooks):
                # Last token, float32 for downstream numerics
                acts[layer].append(hook.out[0, -1].float().cpu())

    # Remove hooks
    for handle in handles:
        handle.remove()

    return {layer: torch.stack(act) for layer, act in acts.items()}


# ---------------------------------------------------------------------------
# HF dataset mapping
# ---------------------------------------------------------------------------


def generate_activations(
    dataset: Dataset,
    model_family: str,
    model_size: str,
    model_type: str,
    layers: list[int],
    device: str = "cpu",
    batch_size: int = 16,
    config_path: Optional[str] = None,
    push_to_hub: Optional[str] = None,
    local_save_path: Optional[str] = None,
) -> Dataset:
    """
    Compute activations for every row in *dataset* and return a new dataset
    with ``layer_{N}_acts`` columns appended.

    Parameters
    ----------
    dataset : datasets.Dataset
        Input dataset with ``text`` column.
    model_family, model_size, model_type : str
        Passed to :func:`~repprobe.models.load_model`.
    layers : list[int]
        Layer indices.  Negative values (e.g. ``-1``) are resolved against the
        actual model depth.
    device : str
        E.g. ``"cuda:0"`` or ``"cpu"``.
    batch_size : int
        Number of statements to process per call to :func:`get_acts`.
    config_path : str | None
        Override path to ``config.ini``.
    push_to_hub : str | None
        If given (e.g. ``"username/my-acts-dataset"``), push the resulting
        dataset to the HuggingFace Hub.
    local_save_path : str | None
        If given, save the dataset to this local directory.

    Returns
    -------
    datasets.Dataset
        Original dataset with activation columns added.
    """
    torch.set_grad_enabled(False)
    tokenizer, model = load_model(model_family, model_size, model_type, device, config_path)

    # Resolve layer indices (handle -1 etc.)
    resolved = [resolve_layer_index(model, l) for l in layers]  # noqa: E741

    def _extract_batch(batch):
        texts = batch["text"]
        acts_dict = get_acts(texts, tokenizer, model, resolved, device)
        out = {}
        for layer in resolved:
            out[f"layer_{layer}_acts"] = acts_dict[layer].numpy().tolist()
        return out

    result = dataset.map(
        _extract_batch,
        batched=True,
        batch_size=batch_size,
        desc=f"Extracting {model_family}-{model_size}-{model_type}",
    )

    if push_to_hub:
        result.push_to_hub(push_to_hub)

    if local_save_path:
        result.save_to_disk(local_save_path)

    # Cleanup
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return result
