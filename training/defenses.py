"""Helpers for instantiating advdef defenses."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import advdef.plugins  # noqa: F401 - registers defense implementations

from advdef.config.defense import DefenseConfig as AdvDefenseConfig
from advdef.core.registry import DEFENSES

from .config import DefenseSpec

_TRAINING_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TRAINING_DIR.parent
BM3D_WRAPPER = _TRAINING_DIR / "bin" / "bm3d_cli_wrapper.py"
BM3D_LOCK_PATH = (_PROJECT_ROOT / ".bm3d_cli.lock").resolve()
BM3D_BINARY_FLAG = "--bm3d-binary"
BM3D_LOCK_FLAG = "--bm3d-lock"
_BM3D_WRAPPER_NOTICE_SHOWN = False


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
    if spec.type == "bm3d":
        requested_binary = params.get("cli_binary") or params.get("binary_path")
        candidates = []
        if requested_binary:
            candidates.append(Path(requested_binary))
        candidates.extend(
            [
                Path("external/bm3d-gpu/build/bm3d"),
                Path("external/adv-it-defenses/external/bm3d-gpu/build/bm3d"),
            ]
        )
        resolved_binary = None
        for candidate in candidates:
            if candidate.exists():
                resolved_binary = candidate
                break
        if resolved_binary is not None:
            params["cli_binary"] = str(resolved_binary)
            params.pop("binary_path", None)
            _wrap_bm3d_cli(params, resolved_binary)
        else:
            search_paths = ", ".join(str(path) for path in candidates)
            raise FileNotFoundError(
                "BM3D defense requires the 'bm3d' CLI binary. "
                f"Checked the following locations: {search_paths}. "
                "Build bm3d-gpu via `advdef setup bm3d-gpu` or point 'cli_binary' to an existing executable."
            )
    return params


def _wrap_bm3d_cli(params: dict, binary: Path) -> None:
    """Force bm3d CLI invocations through a wrapper that serializes GPU access."""
    if not BM3D_WRAPPER.exists():
        return
    global _BM3D_WRAPPER_NOTICE_SHOWN
    if not _BM3D_WRAPPER_NOTICE_SHOWN:
        print(
            "[info] Enabling BM3D CLI wrapper to serialize GPU calls. "
            f"Actual binary: {binary}"
        )
        _BM3D_WRAPPER_NOTICE_SHOWN = True
    params["cli_binary"] = str(BM3D_WRAPPER)
    extra_args = _normalize_extra_args(params.get("cli_extra_args"))

    if not _has_flag(extra_args, BM3D_BINARY_FLAG):
        extra_args = [BM3D_BINARY_FLAG, str(binary), *extra_args]

    if not _has_flag(extra_args, BM3D_LOCK_FLAG):
        lock_path = _resolve_lock_path(params.get("cli_lock_path"))
        extra_args = [BM3D_LOCK_FLAG, str(lock_path), *extra_args]

    params["cli_extra_args"] = extra_args


def _normalize_extra_args(args: object | Iterable[object] | None) -> list[str]:
    if args is None:
        return []
    if isinstance(args, (list, tuple)):
        return [str(item) for item in args]
    return [str(args)]


def _has_flag(args: list[str], flag: str) -> bool:
    for value in args:
        if value == flag:
            return True
        if value.startswith(f"{flag}="):
            return True
    return False


def _resolve_lock_path(explicit: object | None) -> Path:
    if explicit:
        return Path(str(explicit)).expanduser().resolve()
    return BM3D_LOCK_PATH
