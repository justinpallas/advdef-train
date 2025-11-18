"""CLI to precompute defended datasets for reuse."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List

from .config import ExperimentConfig, load_experiment_config
from .data import DatasetSplits, Sample, sample_dataset
from .data_prep import ensure_imagenet_trainset
from .preprocess import prepare_defended_splits
from .train import resolve_imagenet_paths, _set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute defended dataset cache")
    parser.add_argument(
        "--experiment-config",
        required=True,
        help="Path to the YAML file describing the dataset + defenses",
    )
    parser.add_argument(
        "--imagenet-root",
        help="Base directory that contains the ImageNet train/val/devkit splits",
    )
    parser.add_argument(
        "--imagenet-train-root",
        help="Directory that contains the ILSVRC2012 train split",
    )
    parser.add_argument(
        "--imagenet-val-root",
        help="Optional directory for the validation split (unused but kept for parity)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the defended dataset should be materialized",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing defended cache if the output directory already exists",
    )
    parser.add_argument(
        "--keep-workspace",
        action="store_true",
        help="Keep intermediate defense artifacts instead of deleting the workspace",
    )
    return parser.parse_args()


def _flatten_splits(splits: DatasetSplits) -> List[Sample]:
    combined: List[Sample] = []
    combined.extend(splits.train)
    combined.extend(splits.val)
    combined.extend(splits.test)
    return combined


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.experiment_config)
    _set_global_seed(cfg.seed)

    # Reuse the same path resolution helper as training.train.
    imagenet_train_root, _ = resolve_imagenet_paths(args)
    ensure_imagenet_trainset(imagenet_train_root, cfg.dataset.downloads)

    output_root = Path(args.output_dir).expanduser().resolve()
    data_root = output_root / "data"
    workspace_root = output_root / "workspace"

    if data_root.exists():
        if args.force:
            shutil.rmtree(data_root)
        else:
            raise FileExistsError(f"Defended cache already exists at {data_root}. Pass --force to overwrite.")
    if workspace_root.exists() and not args.force:
        raise FileExistsError(
            f"Workspace directory {workspace_root} already exists. Remove it or pass --force to continue."
        )

    data_root.mkdir(parents=True, exist_ok=True)
    workspace_root.mkdir(parents=True, exist_ok=True)

    dataset_splits = sample_dataset(cfg.dataset, imagenet_train_root)
    combined_samples = _flatten_splits(dataset_splits)
    if not combined_samples:
        raise RuntimeError("Sampled dataset is empty; check dataset.total_images and splits.")

    cache_splits = DatasetSplits(
        train=combined_samples,
        val=[],
        test=[],
        class_to_idx=dataset_splits.class_to_idx,
    )

    defended = prepare_defended_splits(cache_splits, cfg.defenses.stack, workspace_root)
    defended_samples = defended.train
    if len(defended_samples) != len(combined_samples):
        raise RuntimeError("Mismatch between original samples and defended outputs; cache generation failed.")

    for original, defended_sample in zip(combined_samples, defended_samples):
        try:
            relative = original.path.relative_to(imagenet_train_root)
        except ValueError as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                f"Sample {original.path} is not under {imagenet_train_root}. "
                "Ensure the cache is built from the same dataset root."
            ) from exc
        destination = data_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(defended_sample.path, destination)

    metadata = {
        "experiment": cfg.name,
        "description": cfg.description,
        "seed": cfg.seed,
        "dataset": {
            "total_images": cfg.dataset.total_images,
            "splits": {
                "train": len(dataset_splits.train),
                "val": len(dataset_splits.val),
                "test": len(dataset_splits.test),
            },
            "per_class_limit": cfg.dataset.per_class_limit,
            "selection_seed": cfg.dataset.selection_seed,
        },
        "defenses": [
            {"type": item.type, "name": item.name, "params": item.params}
            for item in cfg.defenses.stack
        ],
        "source_root": str(imagenet_train_root),
    }
    with (output_root / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    if not args.keep_workspace:
        shutil.rmtree(workspace_root, ignore_errors=True)

    print(f"[info] Defended dataset cached under {data_root}. Set dataset.defended_root to this path for reuse.")


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
