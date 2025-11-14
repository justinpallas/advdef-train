"""Helpers for instantiating advdef defenses."""

from __future__ import annotations

import advdef.plugins  # noqa: F401 - registers defense implementations
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
    config = AdvDefenseConfig(type=spec.type, name=spec.name, params=spec.params or {})
    defense_cls = DEFENSES.get(spec.type)
    return defense_cls(config)
