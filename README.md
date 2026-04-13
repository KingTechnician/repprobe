# RepProbe

**RepProbe** is a general-purpose linear representation probe factory, forked from [Truth is Universal](https://github.com/sciai-lab/Truth_is_Universal).

Where TiU is hardcoded for binary truth detection, RepProbe supports arbitrary label schemas — binary, multiclass, or ordinal — across any HuggingFace-compatible dataset.

**The only contract:** your dataset has a `text` column and a `label` column. RepProbe handles the rest.

---

## Installation

```bash
pip install -e .
```

Or from PyPI (once published):
```bash
pip install repprobe
```

---

## Quick Start (library API)

```python
from repprobe import RepDataset, ProbeFactory

# Load a dataset with pre-computed activation columns
ds = RepDataset("your-org/your-acts-dataset")

# Split into train/test
train_ds, test_ds = ds.stratified_split(train_frac=0.8)

# Train a probe
probe = ProbeFactory.create(
    "lda",
    train_ds.get_acts(layer=12),
    train_ds.get_labels(),
    train_ds.schema,
)

# Evaluate
results = probe.evaluate(test_ds.get_acts(12), test_ds.get_labels())
print(results)
# {"accuracy": 0.91, "macro_f1": 0.89, "per_class_accuracy": {...}, "confusion_matrix": [...]}
```

---

## CLI

```bash
# 1. Extract activations
repprobe generate \
  --dataset your-org/your-dataset \
  --model_family Llama3 \
  --model_size 8B \
  --model_type instruct \
  --layers 8 12 16 \
  --device cuda:0 \
  --push_to_hub your-org/your-dataset-acts

# 2. Find the best extraction layer
repprobe calibrate \
  --dataset your-org/your-dataset-acts \
  --layers 8 12 16

# 3. Train and evaluate a probe
repprobe probe \
  --dataset your-org/your-dataset-acts \
  --layer 12 \
  --probe lda \
  --split_key objective
```

---

## Probes

| Probe | Class | Notes |
|-------|-------|-------|
| `lr` | `LRProbe` | Penaltyless Logistic Regression, any K |
| `lda` | `LDAProbe` | Fisher LDA — generalized TTPD, K−1 discriminant directions |
| `mean_diff` | `MeanDiffProbe` | Class-mean deviation probe — generalized MMProbe |
| `ccs` | `CCSProbe` | Contrastive Consistency Search — **binary + polarity only** |

`ProbeFactory.available_probes(schema)` returns the valid set for your schema automatically.

---

## Dataset Contract

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `text` | str | ✅ | Input utterance |
| `label` | int / ClassLabel | ✅ | Integer-encoded class (0-indexed) |
| `layer_{N}_acts` | list[float] | After extraction | Activation vector at layer N |
| `polarity` | int ±1 | Optional | Enables CCS/TTPD (binary only) |
| `objective` | str | Optional | Grouping key for generalization splits |

---

## Backward Compatibility

Legacy TiU CSV datasets work without modification:

```python
ds = RepDataset.from_csv("datasets/cities.csv", text_col="statement")
```

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
