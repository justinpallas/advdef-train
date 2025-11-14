"""Helpers for instantiating advdef defenses."""

from __future__ import annotations

import advdef.plugins  # noqa: F401 - registers defense implementations
from pathlib import Path

from advdef.config.defense import DefenseConfig as AdvDefenseConfig
from advdef.core.registry import DEFENSES

from .config import DefenseSpec


def ensure_supported(spec: DefenseSpec) -> None:
    try:
        DEFENSES.get(spec.type)
    except KeyError as exc:
        available = ", ".join(sorted(name for name, _ in DEFENSES.items()))
        raise ValueError(
            f"Defense '{spec.type}' is not registered in adv-it-defenses."
            f" Available defenses: {available}"
        ) from exc


def build_advdef_defense(spec: DefenseSpec):
    ensure_supported(spec)
    params = _resolve_params(spec)
    config = AdvDefenseConfig(type=spec.type, name=spec.name, params=params)
    defense_cls = DEFENSES.get(spec.type)
    return defense_cls(config)


def _resolve_params(spec: DefenseSpec) -> dict:
    params = dict(spec.params or {})
    if spec.type == "r-smoe" and "root" not in params:
        candidates = [
            Path("external/r-smoe"),
            Path("external/adv-it-defenses/external/r-smoe"),
        ]
        for candidate in candidates:
            if candidate.exists():
                params["root"] = str(candidate)
                break
        else:
            raise FileNotFoundError(
                "R-SMOE defense requires the 'root' directory. Neither 'external/r-smoe' nor "
                "'external/adv-it-defenses/external/r-smoe' exists. Clone or symlink the submodule and rerun."
            )
    if spec.type == "bm3d" and "cli_binary" not in params and "binary_path" not in params:
        candidates = [
            Path("external/bm3d-gpu/build/bm3d"),
            Path("external/adv-it-defenses/external/bm3d-gpu/build/bm3d"),
        ]
        for candidate in candidates:
            if candidate.exists():
                params["cli_binary"] = str(candidate)
                break
    return params
    return params
