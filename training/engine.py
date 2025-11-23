"""Training loop and model helpers."""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass
import textwrap
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
from torch import nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, StepLR
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from torch.distributions.beta import Beta

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

    if cfg.head_dropout > 0.0:
        model.fc = nn.Sequential(nn.Dropout(p=cfg.head_dropout), model.fc)

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
    warmup_epochs = max(0, int(params.get("warmup_epochs", 0)))
    warmup_start = float(params.get("warmup_start_factor", 0.1))

    def _attach_warmup(main_scheduler):
        if warmup_epochs <= 0:
            return main_scheduler
        total_iters = min(warmup_epochs, training_cfg.epochs)
        start_factor = max(1e-3, warmup_start)
        warmup = LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=max(1, total_iters),
        )
        if training_cfg.epochs <= warmup_epochs or main_scheduler is None:
            return warmup
        return SequentialLR(
            optimizer,
            schedulers=[warmup, main_scheduler],
            milestones=[total_iters],
        )

    if name == "cosine":
        main_epochs = max(0, training_cfg.epochs - warmup_epochs)
        scheduler = None
        if main_epochs > 0:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=main_epochs,
                eta_min=float(params.get("min_lr", 0.0)),
            )
        return _attach_warmup(scheduler)
    if name == "step":
        scheduler = StepLR(
            optimizer,
            step_size=int(params.get("step_size", 30)),
            gamma=float(params.get("gamma", 0.1)),
        )
        return _attach_warmup(scheduler)
    raise ValueError(f"Unsupported scheduler '{training_cfg.scheduler.name}'.")


def run_baseline_resnet50_inference(val_loader: Optional[DataLoader], run_dir: Path) -> None:
    """Evaluate the standard ResNet-50 on the defended validation split."""
    if val_loader is None:
        print("[warn] Skipping baseline inference because the validation split is empty.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc = _evaluate(model, val_loader, criterion, device)

    metrics = {"loss": val_loss, "accuracy": val_acc}
    output_path = run_dir / "baseline_resnet50.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(
        f"Baseline ResNet-50 on defended val split — loss: {val_loss:.4f}, accuracy: {val_acc:.4%}. "
        f"Metrics saved to {output_path}"
    )


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
    use_cuda_amp = cfg.training.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_cuda_amp else None
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing)

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
            use_amp=use_cuda_amp,
            mixup_alpha=cfg.training.mixup_alpha,
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

    _plot_history(history, run_dir, cfg, dataloaders)

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
    use_amp: bool,
    mixup_alpha: float,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(dataloader, desc="train", leave=False)
    for inputs, targets in progress:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        inputs, targets_a, targets_b, lam = _apply_mixup(inputs, targets, mixup_alpha)
        mixup_active = lam != 1.0

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            autocast_ctx = torch.amp.autocast("cuda")
        else:
            autocast_ctx = nullcontext()
        with autocast_ctx:
            outputs = model(inputs)
            if mixup_active:
                loss = lam * criterion(outputs, targets_a) + (1.0 - lam) * criterion(outputs, targets_b)
            else:
                loss = criterion(outputs, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        predictions = outputs.argmax(dim=1)
        if mixup_active:
            correct = lam * (predictions == targets_a).float()
            correct += (1.0 - lam) * (predictions == targets_b).float()
            total_correct += correct.sum().item()
        else:
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


def _plot_history(
    history: Sequence[TrainMetrics],
    run_dir: Path,
    cfg: ExperimentConfig,
    dataloaders: Dict[str, Optional[DataLoader]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - fallback if matplotlib missing
        print("[warn] matplotlib not available; skipping metric plots.")
        return

    if not history:
        return

    epochs = [entry.epoch for entry in history]
    train_loss = [entry.train_loss for entry in history]
    train_acc = [entry.train_acc for entry in history]

    val_epochs_loss, val_loss = _filter_series(history, lambda m: m.val_loss)
    val_epochs_acc, val_acc = _filter_series(history, lambda m: m.val_acc)

    plt.style.use("seaborn-v0_8")
    fig = plt.figure(figsize=(12.5, 6))
    grid = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.25, wspace=0.25)

    ax_loss = fig.add_subplot(grid[0, 0])
    ax_acc = fig.add_subplot(grid[0, 1])
    ax_cfg = fig.add_subplot(grid[1, :])

    ax_loss.plot(epochs, train_loss, label="train")
    if val_loss:
        ax_loss.plot(val_epochs_loss, val_loss, label="val")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-Entropy")
    ax_loss.legend()

    ax_acc.plot(epochs, train_acc, label="train")
    if val_acc:
        ax_acc.plot(val_epochs_acc, val_acc, label="val")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Top-1 Accuracy")
    ax_acc.legend()

    summary_lines = _format_config_summary(cfg, dataloaders)
    ax_cfg.axis("off")
    ax_cfg.set_title("Settings", loc="left")
    ax_cfg.text(
        0,
        1,
        "\n".join(summary_lines),
        ha="left",
        va="top",
        fontsize=9,
        fontfamily="monospace",
    )

    fig.tight_layout()
    plot_path = run_dir / "metrics.png"
    fig.savefig(plot_path)
    plt.close(fig)


def _filter_series(history: Sequence[TrainMetrics], selector):
    xs = []
    ys = []
    for entry in history:
        value = selector(entry)
        if value is not None:
            xs.append(entry.epoch)
            ys.append(value)
    return xs, ys


def _format_config_summary(cfg: ExperimentConfig, dataloaders: Dict[str, Optional[DataLoader]]) -> list[str]:
    split_sizes = {name: len(loader.dataset) if loader is not None else 0 for name, loader in dataloaders.items()}
    defenses = _format_defenses(cfg.defenses.stack)
    optimizer_details = _format_named_params(cfg.training.optimizer.name, cfg.training.optimizer.params)
    scheduler_details = (
        _format_named_params(cfg.training.scheduler.name, cfg.training.scheduler.params)
        if cfg.training.scheduler
        else "scheduler: none"
    )
    finetune = cfg.training.finetune
    model_bits = (
        f"model={cfg.model.architecture} pretrained={cfg.model.pretrained} classes={cfg.model.num_classes}"
    )
    if cfg.model.head_dropout > 0:
        model_bits += f" head_dropout={cfg.model.head_dropout}"
    lines = [
        f"seed={cfg.seed} dataset={cfg.dataset.name} total={cfg.dataset.total_images} splits t/v/te="
        f"{split_sizes.get('train', 0)}/{split_sizes.get('val', 0)}/{split_sizes.get('test', 0)}",
        f"defenses: {defenses}",
        model_bits,
        f"training: epochs={cfg.training.epochs} batch={cfg.training.batch_size} mp={cfg.training.mixed_precision}",
        f"optimizer: {optimizer_details}",
        f"scheduler: {scheduler_details}",
        (
            "finetune: "
            f"freeze_backbone={finetune.freeze_backbone} "
            f"trainable={','.join(finetune.trainable_layers)} "
            f"reset_cls={finetune.reset_classifier}"
        ),
        f"regularization: label_smoothing={cfg.training.label_smoothing} mixup_alpha={cfg.training.mixup_alpha}",
    ]
    if cfg.dataset.defended_root:
        root = textwrap.shorten(str(cfg.dataset.defended_root), width=95, placeholder="…")
        lines.append(f"defended_root: {root}")
    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=110) or [""])
    return wrapped


def _format_defenses(defenses: Sequence) -> str:
    if not defenses:
        return "none"
    parts = []
    for defense in defenses:
        params = _format_params(defense.params)
        label = f"{defense.type}:{defense.name}"
        if params:
            label = f"{label} ({params})"
        parts.append(label)
    return "; ".join(parts)


def _format_named_params(name: str, params: Dict[str, Any]) -> str:
    prefix = f"{name.lower()}" if name else ""
    param_str = _format_params(params)
    if param_str:
        return f"{prefix}: {param_str}"
    return prefix or "none"


def _format_params(params: Dict[str, Any]) -> str:
    if not params:
        return ""
    joined = ", ".join(f"{key}={params[key]}" for key in sorted(params))
    if len(joined) > 90:
        joined = textwrap.shorten(joined, width=90, placeholder="…")
    return joined


def _apply_mixup(inputs: torch.Tensor, targets: torch.Tensor, alpha: float):
    if alpha <= 0.0 or inputs.size(0) == 1:
        return inputs, targets, targets, 1.0
    beta_dist = Beta(alpha, alpha)
    lam = float(beta_dist.sample().item())
    index = torch.randperm(inputs.size(0), device=inputs.device)
    mixed_inputs = lam * inputs + (1.0 - lam) * inputs[index]
    targets_a = targets
    targets_b = targets[index]
    return mixed_inputs, targets_a, targets_b, lam
