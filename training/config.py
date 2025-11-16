"""YAML-backed experiment configuration helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

CONFIG_VERSION = 1
DEFAULT_IMAGENET_TRAIN_URL = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"


def _resolve_path_field(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


@dataclass
class DatasetDownloadConfig:
    auto_download: bool = True
    download_dir: Path = field(default_factory=lambda: _resolve_path_field("downloads"))
    train_archive: Path = field(
        default_factory=lambda: _resolve_path_field("downloads/ILSVRC2012_img_train.tar")
    )
    train_url: str = DEFAULT_IMAGENET_TRAIN_URL

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetDownloadConfig":
        if not data:
            return cls()
        download_dir = _resolve_path_field(data.get("download_dir", "downloads"))
        train_archive_raw = data.get("train_archive")
        train_archive = (
            _resolve_path_field(train_archive_raw)
            if train_archive_raw
            else (download_dir / "ILSVRC2012_img_train.tar")
        )
        return cls(
            auto_download=bool(data.get("auto_download", True)),
            download_dir=download_dir,
            train_archive=train_archive.resolve(),
            train_url=str(data.get("train_url", DEFAULT_IMAGENET_TRAIN_URL)),
        )


@dataclass
class DatasetSplits:
    """Train/val/test ratios that sum (approximately) to 1.0."""

    train: float
    val: float
    test: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetSplits":
        missing = {key for key in ("train", "val", "test") if key not in data}
        if missing:
            raise ValueError(f"dataset.splits missing fields: {sorted(missing)}")
        return cls(train=float(data["train"]), val=float(data["val"]), test=float(data["test"]))

    def validate(self) -> None:
        total = self.train + self.val + self.test
        if total <= 0:
            raise ValueError("dataset.splits must have a positive sum")

    def materialize(self, total_images: int) -> Dict[str, int]:
        """Return per-split counts that respect rounding."""

        if total_images <= 0:
            raise ValueError("dataset.total_images must be positive")
        self.validate()
        total_ratio = self.train + self.val + self.test
        normalized = [self.train / total_ratio, self.val / total_ratio, self.test / total_ratio]
        desired = [ratio * total_images for ratio in normalized]
        counts = [math.floor(val) for val in desired]
        remainder = total_images - sum(counts)
        fractional = [val - math.floor(val) for val in desired]
        order = sorted(range(3), key=lambda idx: fractional[idx], reverse=True)
        for idx in order:
            if remainder <= 0:
                break
            counts[idx] += 1
            remainder -= 1
        return {"train": counts[0], "val": counts[1], "test": counts[2]}


@dataclass
class DatasetConfig:
    name: str
    total_images: int
    splits: DatasetSplits
    shuffle: bool = True
    selection_seed: Optional[int] = None
    class_filter: Optional[List[str]] = None
    per_class_limit: Optional[int] = None
    downloads: DatasetDownloadConfig = field(default_factory=DatasetDownloadConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetConfig":
        if "name" not in data:
            raise ValueError("dataset.name is required")
        if "total_images" not in data:
            raise ValueError("dataset.total_images is required")
        splits = DatasetSplits.from_dict(data.get("splits", {}))
        downloads = DatasetDownloadConfig.from_dict(data.get("downloads", {}))
        selection_seed = data.get("selection_seed")
        class_filter = data.get("class_filter")
        per_class_limit = data.get("per_class_limit")
        per_class_limit_int = (
            int(per_class_limit) if per_class_limit is not None else None
        )
        if per_class_limit_int is not None and per_class_limit_int <= 0:
            raise ValueError("dataset.per_class_limit must be positive when provided")
        return cls(
            name=str(data["name"]),
            total_images=int(data["total_images"]),
            splits=splits,
            shuffle=bool(data.get("shuffle", True)),
            selection_seed=int(selection_seed) if selection_seed is not None else None,
            class_filter=list(class_filter) if class_filter is not None else None,
            per_class_limit=per_class_limit_int,
            downloads=downloads,
        )

    def split_counts(self) -> Dict[str, int]:
        return self.splits.materialize(self.total_images)


@dataclass
class DefenseSpec:
    type: str
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DefenseSpec":
        if "type" not in data:
            raise ValueError("defense entry must include 'type'")
        if "name" not in data:
            raise ValueError("defense entry must include 'name'")
        params = data.get("params", {}) or {}
        return cls(type=str(data["type"]), name=str(data["name"]), params=dict(params))


@dataclass
class DefenseConfig:
    stack: List[DefenseSpec] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DefenseConfig":
        stack = [DefenseSpec.from_dict(item) for item in data.get("stack", [])]
        return cls(stack=stack)


@dataclass
class ModelConfig:
    architecture: str
    pretrained: bool = True
    num_classes: int = 1000
    checkpoint: Optional[Path] = None
    head_dropout: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        checkpoint = data.get("checkpoint")
        head_dropout = float(data.get("head_dropout", 0.0))
        if head_dropout < 0.0 or head_dropout >= 1.0:
            raise ValueError("model.head_dropout must be between 0.0 and 1.0 (exclusive).")
        return cls(
            architecture=str(data.get("architecture", "resnet50")),
            pretrained=bool(data.get("pretrained", True)),
            num_classes=int(data.get("num_classes", 1000)),
            checkpoint=Path(checkpoint) if checkpoint else None,
            head_dropout=head_dropout,
        )


@dataclass
class OptimizerConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizerConfig":
        if "name" not in data:
            raise ValueError("optimizer.name is required")
        return cls(name=str(data["name"]), params=dict(data.get("params", {}) or {}))


@dataclass
class SchedulerConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchedulerConfig":
        if "name" not in data:
            raise ValueError("scheduler.name is required")
        return cls(name=str(data["name"]), params=dict(data.get("params", {}) or {}))


@dataclass
class FinetuneConfig:
    freeze_backbone: bool = True
    trainable_layers: List[str] = field(default_factory=lambda: ["fc"])
    reset_classifier: bool = True
    freeze_batchnorm: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FinetuneConfig":
        return cls(
            freeze_backbone=bool(data.get("freeze_backbone", True)),
            trainable_layers=list(data.get("trainable_layers", ["fc"])),
            reset_classifier=bool(data.get("reset_classifier", True)),
            freeze_batchnorm=bool(data.get("freeze_batchnorm", True)),
        )


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    num_workers: int
    mixed_precision: bool
    optimizer: OptimizerConfig
    scheduler: Optional[SchedulerConfig]
    finetune: FinetuneConfig
    eval_interval: int
    label_smoothing: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        optimizer = OptimizerConfig.from_dict(data.get("optimizer", {}))
        scheduler_cfg = data.get("scheduler")
        scheduler = SchedulerConfig.from_dict(scheduler_cfg) if scheduler_cfg else None
        finetune = FinetuneConfig.from_dict(data.get("finetune", {}))
        smoothing = float(data.get("label_smoothing", 0.0))
        if smoothing < 0.0 or smoothing >= 1.0:
            raise ValueError("training.label_smoothing must be between 0.0 and 1.0 (exclusive).")
        return cls(
            epochs=int(data.get("epochs", 10)),
            batch_size=int(data.get("batch_size", 64)),
            num_workers=int(data.get("num_workers", 8)),
            mixed_precision=bool(data.get("mixed_precision", True)),
            optimizer=optimizer,
            scheduler=scheduler,
            finetune=finetune,
            eval_interval=int(data.get("eval_interval", 1)),
            label_smoothing=smoothing,
        )


@dataclass
class OutputConfig:
    base_dir: Path
    save_every: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutputConfig":
        return cls(
            base_dir=Path(data.get("base_dir", "runs")),
            save_every=int(data.get("save_every", 1)),
        )


@dataclass
class ExperimentConfig:
    name: str
    description: str
    seed: int
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    defenses: DefenseConfig
    output: OutputConfig
    version: int = CONFIG_VERSION

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        dataset = DatasetConfig.from_dict(data.get("dataset", {}))
        model = ModelConfig.from_dict(data.get("model", {}))
        training = TrainingConfig.from_dict(data.get("training", {}))
        defenses = DefenseConfig.from_dict(data.get("defenses", {}))
        output = OutputConfig.from_dict(data.get("output", {}))
        return cls(
            name=str(data.get("name", "experiment")),
            description=str(data.get("description", "")),
            seed=int(data.get("seed", 0)),
            dataset=dataset,
            model=model,
            training=training,
            defenses=defenses,
            output=output,
            version=int(data.get("version", CONFIG_VERSION)),
        )


def load_experiment_config(path: Path | str) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    cfg = ExperimentConfig.from_dict(data)
    cfg.dataset.splits.validate()
    return cfg
