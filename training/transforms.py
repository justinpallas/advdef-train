"""Torchvision transform factories."""

from __future__ import annotations

from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(stage: str):
    ops: list = []
    if stage == "train":
        ops.extend([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
        ])
    else:
        ops.extend([
            T.Resize(256),
            T.CenterCrop(224),
        ])

    ops.extend([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return T.Compose(ops)
