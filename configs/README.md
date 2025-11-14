# Experiment Configs

Each experiment is described via a YAML file consumed by `python -m training.train`.
The schema mirrors the adv-it-defenses style so you can swap defenses or change
the dataset split without touching the code.

## Top-Level Fields

- `name` / `description` / `seed`: identify the run and make the output path deterministic.
- `dataset`: controls how many images to sample from ImageNet and how they are split.
  - `total_images`: absolute budget pulled from the ILSVRC2012 train split.
  - `splits`: fractional ratios (train/val/test) that will be materialized into counts.
  - `shuffle` + `selection_seed`: toggle deterministic subset sampling.
  - `downloads`: optional block; when `auto_download` is true the CLI will fetch
    and unpack the official `ILSVRC2012_img_train.tar` into the requested root if
    it is missing (default cache lives under `downloads/`).
- `model`: architecture + optional checkpoint to warm start.
- `training`: optimizer, scheduler, loader knobs, and a `finetune` block to declare
  which layers stay trainable (e.g., freeze the backbone and only update `layer4`
  + `fc`).
- `defenses`: points at an adv-it-defenses YAML or declares an inline defense stack.
- `output`: base directory plus checkpoint/metadata cadence.

## Sample Launch

```bash
python -m training.train \
  --experiment-config configs/resnet50_transfer.yaml \
  --imagenet-root "$IMAGENET_TRAIN_ROOT/.." \
  --output-dir runs/$(date +%Y%m%d_%H%M)
```

If no CLI flags or env vars are set, the CLI falls back to `datasets/imagenet`
under the current working directory (mirroring the advdef behavior) and expects
its `train/` + `val/` children there, downloading the train tarball if needed.
Use `--imagenet-root` or `IMAGENET_ROOT` to pin a different base directory, and
`--imagenet-train-root` / `--imagenet-val-root` (or the matching
`IMAGENET_*_ROOT` env vars) to override individual directories when they are not
nested under the inferred base.

Override the defense config on the fly with `--defense-config` to reuse the
same dataset/model split with a different advdef pipeline.
