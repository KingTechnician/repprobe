"""
RepProbe — general-purpose linear representation probe factory.

Quick start::

    from repprobe import RepDataset, ProbeFactory

    ds = RepDataset("your-hf-org/your-acts-dataset")
    train_ds, test_ds = ds.stratified_split(train_frac=0.8)

    probe = ProbeFactory.create(
        "lda",
        train_ds.get_acts(layer=12),
        train_ds.get_labels(),
        train_ds.schema,
    )

    results = probe.evaluate(test_ds.get_acts(12), test_ds.get_labels())
    print(results["accuracy"])
"""

from .calibration import calibrate_best_layer, multiclass_fisher_score
from .data import RepDataset
from .eval import compare_probes, evaluate_generalization, evaluate_probe, print_results
from .extraction import generate_activations
from .probes import CCSProbe, LDAProbe, LRProbe, MeanDiffProbe, ProbeFactory
from .schema import LabelSchema

__all__ = [
    # Core data
    "RepDataset",
    "LabelSchema",
    # Extraction
    "generate_activations",
    # Probes
    "ProbeFactory",
    "LRProbe",
    "LDAProbe",
    "MeanDiffProbe",
    "CCSProbe",
    # Calibration
    "calibrate_best_layer",
    "multiclass_fisher_score",
    # Evaluation
    "evaluate_probe",
    "evaluate_generalization",
    "compare_probes",
    "print_results",
]

__version__ = "0.1.0"
