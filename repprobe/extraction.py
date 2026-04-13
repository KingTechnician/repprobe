"""
Activation extraction from HuggingFace-compatible causal language models.

Core primitives
---------------
* :func:`get_acts`          – extract last-token hidden states for a list of
                              strings (model already loaded)
* :func:`generate_activations` – map activations over a HF ``Dataset`` **or**
                              ``DatasetDict`` and optionally push to the Hub

DatasetDict support
-------------------
When *dataset* is a :class:`~datasets.DatasetDict`, activations are extracted
for every split and the function returns a ``DatasetDict`` with the same split
names, each augmented with ``layer_{N}_acts`` columns.
"""
from __future__ import annotations

import gc
from typing import Optional, Union

import torch
from tqdm import tqdm

from .models import Hook, load_model, resolve_layer_index

try:
    from datasets import Dataset, DatasetDict  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install the 'datasets' package.") from exc

_DatasetInput = Union[Dataset, DatasetDict]
_DatasetOutput = Union[Dataset, DatasetDict]


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
                acts[layer].append(hook.out[0, -1].float().cpu())

    for handle in handles:
        handle.remove()

    return {layer: torch.stack(act) for layer, act in acts.items()}


# ---------------------------------------------------------------------------
# HF dataset mapping
# ---------------------------------------------------------------------------


def _map_single_split(
    ds: Dataset,
    tokenizer,
    model,
    resolved_layers: list[int],
    device: str,
    batch_size: int,
    desc: str,
) -> Dataset:
    """Apply activation extraction to one flat Dataset."""

    def _extract_batch(batch):
        texts = batch["text"]
        acts_dict = get_acts(texts, tokenizer, model, resolved_layers, device)
        return {
            f"layer_{layer}_acts": acts_dict[layer].numpy().tolist()
            for layer in resolved_layers
        }

    return ds.map(
        _extract_batch,
        batched=True,
        batch_size=batch_size,
        desc=desc,
    )


def generate_activations(
    dataset: _DatasetInput,
    model_family: str,
    model_size: str,
    model_type: str,
    layers: list[int],
    device: str = "cpu",
    batch_size: int = 16,
    config_path: Optional[str] = None,
    push_to_hub: Optional[str] = None,
    local_save_path: Optional[str] = None,
) -> _DatasetOutput:
    """
    Compute activations for every row in *dataset* and return the dataset
    with ``layer_{N}_acts`` columns appended.

    Accepts both a flat :class:`~datasets.Dataset` **and** a
    :class:`~datasets.DatasetDict` (e.g. one with ``"train"`` and ``"test"``
    splits).  The return type mirrors the input type.

    Parameters
    ----------
    dataset : Dataset | DatasetDict
        Input dataset(s).  Every split must have a ``"text"`` column.
        A ``"label"`` column is preserved but not required for extraction
        itself.
    model_family, model_size, model_type : str
        Passed to :func:`~repprobe.models.load_model`.
    layers : list[int]
        Layer indices to extract.  Negative values (e.g. ``-1``) are
        resolved to absolute indices from the loaded model.
    device : str
        PyTorch device string, e.g. ``"cuda:0"`` or ``"cpu"``.
    batch_size : int
        Rows processed per call to :func:`get_acts`.
    config_path : str | None
        Override path to ``config.ini``.
    push_to_hub : str | None
        HuggingFace Hub repo ID to push the result to (e.g.
        ``"username/my-acts"``).  Works for both ``Dataset`` and
        ``DatasetDict``.
    local_save_path : str | None
        Local directory to save the result to.

    Returns
    -------
    Dataset | DatasetDict
        Same type as *dataset*, with ``layer_{N}_acts`` columns added to
        every split.
    """
    prev_grad = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    tokenizer, model = load_model(model_family, model_size, model_type, device, config_path)

    resolved = [resolve_layer_index(model, l) for l in layers]  # noqa: E741
    tag = f"{model_family}-{model_size}-{model_type}"

    # ── Dispatch on input type ─────────────────────────────────────────
    if isinstance(dataset, DatasetDict):
        result_splits: dict[str, Dataset] = {}
        for split_name, split_ds in dataset.items():
            result_splits[split_name] = _map_single_split(
                split_ds, tokenizer, model, resolved, device, batch_size,
                desc=f"Extracting {tag} [{split_name}]",
            )
        result: _DatasetOutput = DatasetDict(result_splits)
    else:
        result = _map_single_split(
            dataset, tokenizer, model, resolved, device, batch_size,
            desc=f"Extracting {tag}",
        )

    # ── Persist ────────────────────────────────────────────────────────
    if push_to_hub:
        result.push_to_hub(push_to_hub)

    if local_save_path:
        result.save_to_disk(local_save_path)

    # ── Cleanup ────────────────────────────────────────────────────────
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    torch.set_grad_enabled(prev_grad)   # restore grad state for caller

    return result
