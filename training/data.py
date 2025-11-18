"""Dataset sampling and DataLoader helpers."""

from __future__ import annotations

import math
import random
from collections import defaultdict
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
    preprocessed: bool = False

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

    per_class_limit = cfg.per_class_limit
    per_class_selected: Dict[int, List[Sample]] | None = None
    if per_class_limit is not None:
        samples_by_class = defaultdict(list)
        for sample in all_samples:
            samples_by_class[sample.label].append(sample)

        available = sum(min(per_class_limit, len(entries)) for entries in samples_by_class.values())
        if cfg.total_images > available:
            raise ValueError(
                f"Requested {cfg.total_images} images but dataset.per_class_limit={per_class_limit} "
                f"only yields {available} samples. Reduce total_images or increase the limit."
            )

        per_class_selected = {}
        total_selected = 0
        for label in sorted(samples_by_class):
            class_samples = list(samples_by_class[label])
            if cfg.shuffle:
                rng.shuffle(class_samples)
            chosen = class_samples[:per_class_limit]
            per_class_selected[label] = chosen
            total_selected += len(chosen)

        target_total = min(cfg.total_images, total_selected)
        if target_total < total_selected:
            _trim_per_class_samples(per_class_selected, target_total, rng)
    else:
        if cfg.total_images > len(all_samples):
            raise ValueError(
                f"Requested {cfg.total_images} images but only {len(all_samples)} are available under {root}."
            )
        subset = list(all_samples[: cfg.total_images])

    counts = cfg.split_counts()

    if per_class_selected is not None:
        train_samples, val_samples, test_samples = _split_by_class(
            per_class_selected,
            counts,
            rng if cfg.shuffle else None,
        )
    else:
        subset = subset[: cfg.total_images]
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
        transform = build_transform(split_name, samples.preprocessed)
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


def _trim_per_class_samples(
    samples_by_class: Dict[int, List[Sample]],
    target_total: int,
    rng: random.Random,
) -> None:
    labels = list(samples_by_class.keys())
    if rng is not None:
        rng.shuffle(labels)
    current_total = sum(len(entries) for entries in samples_by_class.values())
    idx = 0
    while current_total > target_total and labels:
        label = labels[idx % len(labels)]
        if samples_by_class[label]:
            samples_by_class[label].pop()
            current_total -= 1
        idx += 1


def _split_by_class(
    samples_by_class: Dict[int, List[Sample]],
    desired_counts: Dict[str, int],
    rng: random.Random | None,
) -> tuple[List[Sample], List[Sample], List[Sample]]:
    splits = {key: [] for key in ("train", "val", "test")}
    remaining = {key: desired_counts.get(key, 0) for key in ("train", "val", "test")}
    total_remaining = sum(remaining.values())

    if total_remaining <= 0:
        return [], [], []

    for label in sorted(samples_by_class):
        class_samples = list(samples_by_class[label])
        if not class_samples:
            continue
        if rng is not None:
            rng.shuffle(class_samples)
        class_total = len(class_samples)

        if total_remaining <= 0:
            break

        ratios = {
            split: (remaining[split] / total_remaining) if total_remaining > 0 else 0.0
            for split in ("train", "val", "test")
        }
        allocation = _materialize_class_counts(class_total, ratios)
        assigned = 0
        for split in allocation:
            take = min(allocation[split], remaining[split])
            allocation[split] = take
            assigned += take

        leftover = class_total - assigned
        if leftover > 0:
            for split in ("train", "val", "test"):
                if leftover <= 0:
                    break
                capacity = remaining[split] - allocation[split]
                if capacity <= 0:
                    continue
                take = min(capacity, leftover)
                allocation[split] += take
                leftover -= take

        offset = 0
        for split in ("train", "val", "test"):
            take = allocation[split]
            if take <= 0:
                continue
            splits[split].extend(class_samples[offset : offset + take])
            offset += take
            remaining[split] = max(0, remaining[split] - take)
        total_remaining = max(0, total_remaining - class_total)

    return splits["train"], splits["val"], splits["test"]


def _materialize_class_counts(total: int, ratios: Dict[str, float]) -> Dict[str, int]:
    allocation = {split: 0 for split in ("train", "val", "test")}
    if total <= 0:
        return allocation

    normalized = sum(ratios.values())
    if normalized <= 0:
        allocation["train"] = total
        return allocation

    desired = {split: ratios[split] / normalized * total for split in ("train", "val", "test")}
    for split in allocation:
        allocation[split] = int(math.floor(desired[split]))
    assigned = sum(allocation.values())
    remainder = total - assigned
    order = sorted(
        ("train", "val", "test"),
        key=lambda split: desired[split] - allocation[split],
        reverse=True,
    )
    for split in order:
        if remainder <= 0:
            break
        allocation[split] += 1
        remainder -= 1
    return allocation
