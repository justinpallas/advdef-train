"""Defense preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple
from tqdm import tqdm

from advdef.core.pipeline import DatasetVariant

from .config import DefenseSpec
from .data import DatasetSplits, Sample
from .defenses import build_advdef_defense, ensure_supported
from PIL import Image

DEFENSE_RESIZE_SHORT_SIDE = 256
DEFENSE_CROP_SIZE = 224


def prepare_defended_splits(
    splits: DatasetSplits,
    defense_stack: Iterable[DefenseSpec],
    output_root: Path,
) -> DatasetSplits:
    output_root.mkdir(parents=True, exist_ok=True)
    current = splits
    for index, spec in enumerate(defense_stack):
        ensure_supported(spec)
        step_root = output_root / f"{index:02d}_{spec.type}"
        current = _apply_advdef_defense(current, spec, step_root)
    return current


def _apply_advdef_defense(splits: DatasetSplits, spec: DefenseSpec, step_root: Path) -> DatasetSplits:
    new_train = _run_defense_on_split(splits.train, "train", spec, step_root)
    new_val = _run_defense_on_split(splits.val, "val", spec, step_root)
    new_test = _run_defense_on_split(splits.test, "test", spec, step_root)
    return DatasetSplits(
        train=new_train,
        val=new_val,
        test=new_test,
        class_to_idx=splits.class_to_idx,
        preprocessed=True,
    )


def _run_defense_on_split(
    samples: List[Sample],
    split_name: str,
    spec: DefenseSpec,
    step_root: Path,
) -> List[Sample]:
    if not samples:
        return []

    defense = build_advdef_defense(spec)
    inputs_dir = step_root / split_name / "inputs"
    artifacts_dir = step_root / split_name / "artifacts"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    alias_map = _materialize_inputs(samples, inputs_dir)

    variant = DatasetVariant(name=f"{split_name}_{spec.name or spec.type}", data_dir=str(inputs_dir))
    context = SimpleNamespace(artifacts_dir=artifacts_dir)
    if hasattr(defense, "initialize"):
        defense.initialize(context, [variant])
    result_variant = defense.run(context, variant)
    if hasattr(defense, "finalize"):
        defense.finalize()

    produced = _index_outputs(Path(result_variant.data_dir))
    defended_samples: List[Sample] = []
    for alias, sample in tqdm(alias_map, desc=f"{split_name}:{spec.type}", leave=False):
        produced_path = produced.get(alias)
        if produced_path is None:
            raise FileNotFoundError(
                f"Defense '{spec.type}' did not produce output for alias '{alias}' in split '{split_name}'."
            )
        defended_samples.append(Sample(path=produced_path, label=sample.label, label_name=sample.label_name))

    return defended_samples


def _materialize_inputs(samples: List[Sample], inputs_dir: Path) -> List[Tuple[str, Sample]]:
    alias_map: List[Tuple[str, Sample]] = []
    for index, sample in enumerate(samples):
        alias = _sanitize_alias(f"{sample.label_name}_{index:06d}")
        extension = sample.path.suffix.lower() or ".jpg"
        destination = inputs_dir / sample.label_name / f"{alias}{extension}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        _prepare_for_defense(sample.path, destination)
        alias_map.append((alias, sample))
    return alias_map


def _prepare_for_defense(src: Path, dst: Path) -> None:
    with Image.open(src) as image:
        image = image.convert("RGB")
        image = _resize_shortest_side(image, DEFENSE_RESIZE_SHORT_SIDE)
        image = _center_crop(image, DEFENSE_CROP_SIZE)
        _save_image(image, dst)


def _resize_shortest_side(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    if width == 0 or height == 0:
        return image
    short, long = (width, height) if width <= height else (height, width)
    if short == size:
        return image
    if width < height:
        new_width = size
        new_height = int(round(size * height / width))
    else:
        new_height = size
        new_width = int(round(size * width / height))
    return image.resize((new_width, new_height), Image.BILINEAR)


def _center_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    if width == size and height == size:
        return image
    left = int(round((width - size) / 2.0))
    top = int(round((height - size) / 2.0))
    right = left + size
    bottom = top + size
    return image.crop((left, top, right, bottom))


def _save_image(image: Image.Image, dst: Path) -> None:
    extension = dst.suffix.lower()
    save_kwargs = {}
    if extension in {".png"}:
        fmt = "PNG"
    elif extension in {".bmp"}:
        fmt = "BMP"
    elif extension in {".tiff", ".tif"}:
        fmt = "TIFF"
    else:
        fmt = "JPEG"
        save_kwargs = {"quality": 95}
        if extension not in {".jpg", ".jpeg"}:
            dst = dst.with_suffix(".jpg")
    image.save(dst, format=fmt, **save_kwargs)


def _index_outputs(root: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file():
            mapping[path.stem] = path
    return mapping


def _sanitize_alias(text: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in text)
