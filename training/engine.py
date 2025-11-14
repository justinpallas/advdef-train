"""Training loop and model helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from .config import ExperimentConfig, FinetuneConfig, ModelConfig, TrainingConfig


@dataclass
class TrainMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: Optional[float]
    val_acc: Optional[float]

    def to_dict(self) -> Dict[str, float | int | None]:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "train_acc": self.train_acc,
            "val_loss": self.val_loss,
            "val_acc": self.val_acc,
        }


def build_model(cfg: ModelConfig, finetune: FinetuneConfig) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if cfg.pretrained else None
    model = models.resnet50(weights=weights)

    if cfg.num_classes != 1000 or finetune.reset_classifier:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, cfg.num_classes)

    if cfg.checkpoint:
        state_dict = torch.load(str(cfg.checkpoint), map_location="cpu")
        model.load_state_dict(state_dict)

    apply_finetune_controls(model, finetune)
    return model


def apply_finetune_controls(model: nn.Module, finetune: FinetuneConfig) -> None:
    if finetune.freeze_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = False

        for layer_name in finetune.trainable_layers:
            module = _resolve_module(model, layer_name)
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = True

    if finetune.freeze_batchnorm:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False


def _resolve_module(model: nn.Module, path: str) -> nn.Module | None:
    target: nn.Module | None = model
    for part in path.split("."):
        if not hasattr(target, part):
            return None
        next_module = getattr(target, part)
        if not isinstance(next_module, nn.Module):
            return None
        target = next_module
    return target


def create_optimizer(training_cfg: TrainingConfig, model: nn.Module):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters remain after applying finetune configuration.")

    name = training_cfg.optimizer.name.lower()
    opts = training_cfg.optimizer.params
    if name == "adamw":
        return AdamW(params, lr=float(opts.get("lr", 5e-5)), weight_decay=float(opts.get("weight_decay", 0.01)))
    if name == "sgd":
        return SGD(
            params,
            lr=float(opts.get("lr", 0.01)),
            momentum=float(opts.get("momentum", 0.9)),
            weight_decay=float(opts.get("weight_decay", 0.0)),
            nesterov=bool(opts.get("nesterov", False)),
        )
    raise ValueError(f"Unsupported optimizer '{training_cfg.optimizer.name}'.")


def create_scheduler(training_cfg: TrainingConfig, optimizer):
    if training_cfg.scheduler is None:
        return None
    name = training_cfg.scheduler.name.lower()
    params = training_cfg.scheduler.params
    if name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=training_cfg.epochs,
            eta_min=float(params.get("min_lr", 0.0)),
        )
    raise ValueError(f"Unsupported scheduler '{training_cfg.scheduler.name}'.")


def train_model(
    cfg: ExperimentConfig,
    dataloaders: Dict[str, Optional[DataLoader]],
    run_dir: Path,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.model, cfg.training.finetune)
    model.to(device)

    optimizer = create_optimizer(cfg.training, model)
    scheduler = create_scheduler(cfg.training, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.mixed_precision and device.type == "cuda")
    criterion = nn.CrossEntropyLoss()

    history: list[TrainMetrics] = []
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.training.epochs + 1):
        train_loader = dataloaders.get("train")
        if train_loader is None:
            raise RuntimeError("Training split is empty; adjust dataset.total_images or splits.")

        train_loss, train_acc = _run_train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            criterion,
            device,
            mixed_precision=cfg.training.mixed_precision,
        )

        val_loader = dataloaders.get("val")
        val_loss = val_acc = None
        if val_loader is not None:
            val_loss, val_acc = _evaluate(model, val_loader, criterion, device)

        history.append(TrainMetrics(epoch=epoch, train_loss=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc))

        if scheduler is not None:
            scheduler.step()

        if epoch % cfg.output.save_every == 0:
            checkpoint_path = checkpoints_dir / f"epoch_{epoch:03d}.pt"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, checkpoint_path)

    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump([entry.to_dict() for entry in history], handle, indent=2)

    test_loader = dataloaders.get("test")
    if test_loader is not None:
        test_loss, test_acc = _evaluate(model, test_loader, criterion, device)
        summary_path = run_dir / "metrics_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump({"test_loss": test_loss, "test_acc": test_acc}, handle, indent=2)


def _run_train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer,
    scaler,
    criterion,
    device: torch.device,
    mixed_precision: bool,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(dataloader, desc="train", leave=False)
    for inputs, targets in progress:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=mixed_precision and device.type == "cuda"):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * inputs.size(0)
        predictions = outputs.argmax(dim=1)
        total_correct += (predictions == targets).sum().item()
        total_samples += inputs.size(0)

    avg_loss = total_loss / max(1, total_samples)
    accuracy = total_correct / max(1, total_samples)
    return avg_loss, accuracy


def _evaluate(model: nn.Module, dataloader: DataLoader, criterion, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="eval", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == targets).sum().item()
            total_samples += inputs.size(0)

    avg_loss = total_loss / max(1, total_samples)
    accuracy = total_correct / max(1, total_samples)
    return avg_loss, accuracy
