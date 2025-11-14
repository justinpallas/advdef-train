"""CLI entry-point for launching transfer-learning experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

from .config import ExperimentConfig, load_experiment_config
from .data_prep import ensure_imagenet_trainset

ENV_TRAIN_ROOT = "IMAGENET_TRAIN_ROOT"
ENV_VAL_ROOT = "IMAGENET_VAL_ROOT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transfer-learning launcher")
    parser.add_argument(
        "--experiment-config",
        required=True,
        help="Path to the YAML file that describes the experiment",
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
        help="Optional directory for the validation split (defaults to train root)",
    )
    parser.add_argument(
        "--output-dir",
        help="Overrides the output.base_dir from the YAML if provided",
    )
    parser.add_argument(
        "--defense-config",
        help="Override the defenses.config_path entry without touching the YAML file",
    )
    return parser.parse_args()


def _resolve_optional_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def _first_resolved(*values: Path | str | None) -> Path | None:
    for value in values:
        resolved = _resolve_optional_path(value)
        if resolved is not None:
            return resolved
    return None


def resolve_imagenet_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    base_root = _resolve_optional_path(args.imagenet_root)
    train_root = _first_resolved(
        args.imagenet_train_root,
        os.environ.get(ENV_TRAIN_ROOT),
        base_root / "train" if base_root else None,
    )
    if train_root is None:
        raise ValueError(
            "Unable to locate ImageNet train split. Specify --imagenet-train-root, --imagenet-root, or set "
            f"{ENV_TRAIN_ROOT}."
        )

    val_root = _first_resolved(
        args.imagenet_val_root,
        os.environ.get(ENV_VAL_ROOT),
        base_root / "val" if base_root else None,
    )
    if val_root is None:
        val_root = train_root

    return train_root, val_root


def materialize_run_directory(cfg: ExperimentConfig, override: str | None) -> Path:
    base_dir = Path(override) if override else cfg.output.base_dir
    run_dir = base_dir / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def export_plan(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.experiment_config)

    if args.defense_config:
        cfg.defenses.config_path = Path(args.defense_config)

    try:
        imagenet_train_root, imagenet_val_root = resolve_imagenet_paths(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if not imagenet_train_root.exists():
        raise FileNotFoundError(
            f"ImageNet train root {imagenet_train_root} does not exist."
        )

    ensure_imagenet_trainset(imagenet_train_root, cfg.dataset.downloads)

    split_counts = cfg.dataset.split_counts()
    run_dir = materialize_run_directory(cfg, args.output_dir)

    plan = {
        "experiment": cfg.name,
        "description": cfg.description,
        "seed": cfg.seed,
        "dataset": {
            "name": cfg.dataset.name,
            "total_images": cfg.dataset.total_images,
            "splits": split_counts,
            "imagenet_train_root": str(imagenet_train_root),
            "imagenet_val_root": str(imagenet_val_root),
        },
        "model": {
            "architecture": cfg.model.architecture,
            "pretrained": cfg.model.pretrained,
            "num_classes": cfg.model.num_classes,
            "checkpoint": str(cfg.model.checkpoint) if cfg.model.checkpoint else None,
        },
        "training": {
            "epochs": cfg.training.epochs,
            "batch_size": cfg.training.batch_size,
            "num_workers": cfg.training.num_workers,
            "mixed_precision": cfg.training.mixed_precision,
            "finetune": {
                "freeze_backbone": cfg.training.finetune.freeze_backbone,
                "trainable_layers": cfg.training.finetune.trainable_layers,
                "reset_classifier": cfg.training.finetune.reset_classifier,
                "freeze_batchnorm": cfg.training.finetune.freeze_batchnorm,
            },
            "optimizer": {
                "name": cfg.training.optimizer.name,
                "params": cfg.training.optimizer.params,
            },
            "scheduler": (
                {
                    "name": cfg.training.scheduler.name,
                    "params": cfg.training.scheduler.params,
                }
                if cfg.training.scheduler
                else None
            ),
            "eval_interval": cfg.training.eval_interval,
        },
        "defenses": {
            "config_path": str(cfg.defenses.config_path) if cfg.defenses.config_path else None,
            "stack": [
                {"type": item.type, "name": item.name, "params": item.params}
                for item in cfg.defenses.stack
            ],
        },
        "output": str(run_dir),
    }

    export_plan(run_dir / "plan.json", plan)

    # Real training logic will consume ``plan``; wiring the config in place now
    # ensures we can focus on data/defense plumbing next.
    print(f"Experiment plan written to {run_dir / 'plan.json'}")


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
