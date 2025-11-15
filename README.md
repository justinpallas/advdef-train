# advdef-train

This Repository is used to fine-tune a pretrained ResNet-50 Model with the defenses applied in `adv-it-defenses`.

## Prerequisites

- Python 3.10+ (pyenv or system python both work)
- GPU w/ CUDA recommended but not required
- Access to the ILSVRC2012 train split (untarred or the original TAR)
- Git submodule for `external/adv-it-defenses` already checked out

## Initial Setup

```bash
git clone <this repo>
cd advdef-train
git submodule update --init --recursive  # pulls adv-it-defenses (and its submodules)

python -m venv .venv
source .venv/bin/activate

# Install this training package (brings torch, torchvision, etc.)
pip install -e .

# Install the defenses repo so the `advdef` CLI is available
pip install -e external/adv-it-defenses[dev]

# One-time setup for heavy defenses (run from the adv-it-defenses directory)
cd external/adv-it-defenses
advdef setup r-smoe      # if you plan to use R-SMoE
advdef setup bm3d-gpu    # if you plan to use BM3D
cd ../..
```

## Dataset Layout

The CLI defaults to `datasets/imagenet/train` under the repo root. Either:

- Untar `ILSVRC2012_img_train.tar` into that folder, or
- Point the CLI at your existing extraction with `--imagenet-root` or
  `IMAGENET_ROOT` env vars (it infers `train/` underneath).

If the train directory is missing and `dataset.downloads.auto_download: true`,
`python -m training.train` will download/extract the official ImageNet train tar
for you (this is ~150 GB and takes a long time).

## Running an Experiment

```bash
source .venv/bin/activate

# Set dataset roots via flags or env vars (optional if using defaults)
export IMAGENET_ROOT=/path/to/ILSVRC2012

python -m training.train \
  --experiment-config configs/resnet50_transfer.yaml
```

This command will:

1. Sample the requested number of images/splits from the train set.
2. Sequentially apply each defense in `defenses.stack` using the advdef
   implementations (outputs land under `runs/<exp>/defended/`).
3. Launch the fine-tuning loop (ResNet-50 by default) with the optimizer /
   scheduler / finetune settings in the YAML.
4. Write `plan.json`, `dataset_split.json`, per-epoch metrics/plots, and checkpoints under
   `runs/<experiment-name>/`.

## Configuration Notes

- `configs/resnet50_transfer.yaml` is a good starting point. Duplicate it to
  describe new experiments (different datasets, defense stacks, optimizers, etc.).
- Defenses listed in the YAML must exist in `adv-it-defenses`. Heavy defenses like
  R-SMOE run via the advdef pipeline, so make sure their setup commands have been
  executed and the paths (e.g., `external/r-smoe`) exist.
- The CLI enforces deterministic sampling when `selection_seed` is set, making it
  easy to reproduce runs.

## Troubleshooting

- `advdef` command not found → ensure `pip install -e external/adv-it-defenses[dev]`
  ran inside your venv and rehash pyenv if needed.
- R-SMOE missing assets → rerun `advdef setup r-smoe` and confirm the `root` path
  in the YAML points to the prepared directory.
- Dataset not found → either create `datasets/imagenet/train` and untar the train
  set there or pass `--imagenet-root /existing/ILSVRC2012` on the CLI.
