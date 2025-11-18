"""Torchvision transform factories."""

from __future__ import annotations

from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RESNET_SCALE_JITTER_SIZES = [256, 288, 320, 352, 384, 416, 448, 480]


def _resnet_style_train_ops(preprocessed: bool) -> list:
    if preprocessed:
        # Keep block artifacts intact; only minimal spatial shift + flip to avoid memorizing single crops.
        return [
            T.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05), fill=0),
            T.RandomHorizontalFlip(),
        ]

    scale_choices = [T.Resize(size) for size in RESNET_SCALE_JITTER_SIZES]
    return [
        T.RandomChoice(scale_choices),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(),
    ]


def build_transform(stage: str, preprocessed: bool = False):
    if stage == "train":
        ops = _resnet_style_train_ops(preprocessed)
    else:
        ops = [] if preprocessed else [T.Resize(256), T.CenterCrop(224)]

    ops.extend([T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    return T.Compose(ops)
