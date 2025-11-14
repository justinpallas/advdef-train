"""Dataset sampling and DataLoader helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .config import DatasetConfig
from .transforms import build_transform

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp"}


@dataclass
class Sample:
    path: Path
    label: int
    label_name: str


@dataclass
class DatasetSplits:
    train: List[Sample]
    val: List[Sample]
    test: List[Sample]
    class_to_idx: Dict[str, int]

    def as_dict(self) -> Dict[str, List[Sample]]:
        return {"train": self.train, "val": self.val, "test": self.test}


class ImageNetSubset(Dataset):
    """Minimal dataset backed by an explicit list of samples."""

    def __init__(self, samples: Sequence[Sample], transform) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    def __getitem__(self, index: int):  # pragma: no cover - I/O heavy
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, sample.label


def _discover_class_folders(root: Path, class_filter: Sequence[str] | None) -> Dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(f"ImageNet root {root} does not exist.")

    all_dirs = [entry for entry in sorted(root.iterdir()) if entry.is_dir()]
    if class_filter:
        class_set = set(class_filter)
        class_dirs = [entry for entry in all_dirs if entry.name in class_set]
    else:
        class_dirs = all_dirs

    if not class_dirs:
        raise RuntimeError(
            f"No class folders found under {root}. Expected ImageNet-style structure with synset directories."
        )

    return {entry.name: entry for entry in class_dirs}


def _collect_images(class_dirs: Dict[str, Path]) -> tuple[List[Sample], Dict[str, int]]:
    samples: List[Sample] = []
    class_to_idx = {name: idx for idx, name in enumerate(sorted(class_dirs))}
    for name, folder in class_dirs.items():
        label = class_to_idx[name]
        for file in folder.rglob("*"):
            if not file.is_file():
                continue
            if file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            samples.append(Sample(path=file, label=label, label_name=name))
    if not samples:
        raise RuntimeError("No images discovered. Check that the ImageNet train split is extracted correctly.")
    return samples, class_to_idx


def sample_dataset(cfg: DatasetConfig, root: Path) -> DatasetSplits:
    class_dirs = _discover_class_folders(root, cfg.class_filter)
    all_samples, class_to_idx = _collect_images(class_dirs)

    rng = random.Random(cfg.selection_seed if cfg.selection_seed is not None else cfg.total_images)
    if cfg.shuffle:
        rng.shuffle(all_samples)

    if cfg.total_images > len(all_samples):
        raise ValueError(
            f"Requested {cfg.total_images} images but only {len(all_samples)} are available under {root}."
        )

    subset = all_samples[: cfg.total_images]
    counts = cfg.split_counts()

    train_end = counts["train"]
    val_end = train_end + counts["val"]

    train_samples = subset[:train_end]
    val_samples = subset[train_end:val_end]
    test_samples = subset[val_end : val_end + counts["test"]]

    return DatasetSplits(
        train=train_samples,
        val=val_samples,
        test=test_samples,
        class_to_idx=class_to_idx,
    )


def build_dataloaders(
    samples: DatasetSplits,
    batch_size: int,
    num_workers: int,
):
    loaders = {}
    for split_name in ("train", "val", "test"):
        split_samples = getattr(samples, split_name)
        if not split_samples:
            loaders[split_name] = None
            continue
        transform = build_transform(split_name)
        dataset = ImageNetSubset(split_samples, transform)
        shuffle = split_name == "train"
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
    return loaders
