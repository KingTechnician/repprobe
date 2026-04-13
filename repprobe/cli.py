"""
RepProbe CLI — three sub-commands:

  repprobe generate   — extract activations and save / push to Hub
  repprobe calibrate  — find the best layer via Fisher criterion
  repprobe probe      — train and evaluate a probe
"""
from __future__ import annotations

import argparse
import json
import sys


def _cmd_generate(args: argparse.Namespace) -> None:
    from datasets import load_dataset  # noqa: PLC0415

    from .extraction import generate_activations  # noqa: PLC0415

    print(f"Loading dataset '{args.dataset}' ...")
    ds_dict = load_dataset(args.dataset, trust_remote_code=True)

    # Merge all splits or pick one
    if args.split:
        ds = ds_dict[args.split]
    else:
        from datasets import concatenate_datasets  # noqa: PLC0415

        ds = concatenate_datasets(list(ds_dict.values()))

    layers = [int(l) for l in args.layers]  # noqa: E741

    print(f"Extracting layers {layers} using {args.model_family}-{args.model_size}-{args.model_type} ...")
    result = generate_activations(
        dataset=ds,
        model_family=args.model_family,
        model_size=args.model_size,
        model_type=args.model_type,
        layers=layers,
        device=args.device,
        batch_size=args.batch_size,
        config_path=args.config,
        push_to_hub=args.push_to_hub if args.push_to_hub else None,
        local_save_path=args.output_dir if args.output_dir else None,
    )
    print(f"Done. Dataset has {len(result)} rows and columns: {result.column_names}")


def _cmd_calibrate(args: argparse.Namespace) -> None:
    from .calibration import calibrate_best_layer  # noqa: PLC0415
    from .data import RepDataset  # noqa: PLC0415

    print(f"Loading dataset '{args.dataset}' ...")
    ds = RepDataset(args.dataset, split=args.split)

    layers = [int(l) for l in args.layers] if args.layers else None  # noqa: E741
    best, scores = calibrate_best_layer(ds, layers=layers, verbose=True)
    print(f"\nBest layer: {best}")
    print("Scores:", json.dumps({str(k): round(v, 6) for k, v in scores.items()}, indent=2))


def _cmd_probe(args: argparse.Namespace) -> None:
    from .data import RepDataset  # noqa: PLC0415
    from .eval import print_results  # noqa: PLC0415
    from .probes import ProbeFactory  # noqa: PLC0415

    print(f"Loading dataset '{args.dataset}' ...")
    ds = RepDataset(args.dataset, split=args.split)

    if args.split_key:
        all_vals = sorted(set(ds.hf_dataset[args.split_key]))
        split_idx = int(len(all_vals) * 0.8)
        train_vals, test_vals = all_vals[:split_idx], all_vals[split_idx:]
        train_ds, test_ds = ds.split_by_column(args.split_key, train_vals, test_vals)
        print(f"Generalization split on '{args.split_key}': "
              f"train={train_vals}, test={test_vals}")
    else:
        train_ds, test_ds = ds.stratified_split(train_frac=0.8)

    layer = args.layer
    train_acts = train_ds.get_acts(layer)
    train_labels = train_ds.get_labels()
    test_acts = test_ds.get_acts(layer)
    test_labels = test_ds.get_labels()

    print(f"Training {args.probe.upper()} probe on layer {layer} ...")
    probe = ProbeFactory.create(args.probe, train_acts, train_labels, ds.schema)

    results = probe.evaluate(test_acts, test_labels)
    print_results(results, title=f"{args.probe.upper()} probe — layer {layer}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repprobe",
        description="RepProbe: linear representation probe factory.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── generate ──────────────────────────────────────────────────────
    gen = sub.add_parser("generate", help="Extract and save activations")
    gen.add_argument("--dataset", required=True, help="HF dataset name or path")
    gen.add_argument("--split", default=None, help="Which split to process")
    gen.add_argument("--model_family", required=True)
    gen.add_argument("--model_size", required=True)
    gen.add_argument("--model_type", required=True)
    gen.add_argument("--layers", nargs="+", default=["-1"])
    gen.add_argument("--device", default="cpu")
    gen.add_argument("--batch_size", type=int, default=16)
    gen.add_argument("--config", default=None, help="Path to config.ini")
    gen.add_argument("--push_to_hub", default=None, help="HF repo to push to")
    gen.add_argument("--output_dir", default=None, help="Local path to save dataset")

    # ── calibrate ─────────────────────────────────────────────────────
    cal = sub.add_parser("calibrate", help="Select best extraction layer")
    cal.add_argument("--dataset", required=True)
    cal.add_argument("--split", default=None)
    cal.add_argument("--layers", nargs="+", default=None)

    # ── probe ─────────────────────────────────────────────────────────
    prb = sub.add_parser("probe", help="Train and evaluate a probe")
    prb.add_argument("--dataset", required=True)
    prb.add_argument("--split", default=None)
    prb.add_argument("--layer", type=int, required=True)
    prb.add_argument("--probe", default="lda",
                     choices=["lr", "lda", "mean_diff", "ccs"])
    prb.add_argument("--split_key", default=None,
                     help="Column for generalization split")
    prb.add_argument("--output", default=None,
                     help="Save results JSON to this path")

    return parser


def main() -> None:  # pragma: no cover
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "generate": _cmd_generate,
        "calibrate": _cmd_calibrate,
        "probe": _cmd_probe,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
