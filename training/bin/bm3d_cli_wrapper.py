#!/usr/bin/env python3
"""Serialize bm3d-gpu CLI invocations to keep GPU runs stable."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX systems
    fcntl = None  # type: ignore[assignment]


BINARY_FLAG = "--bm3d-binary"
LOCK_FLAG = "--bm3d-lock"
DEFAULT_LOCK_NAME = ".bm3d_cli.lock"


def _consume_flag(
    iterator: List[str],
    current: str,
    flag: str,
) -> Tuple[bool, str | None]:
    """Return (matched, value) for flags of the form '--flag value' or '--flag=value'."""
    if current == flag:
        if not iterator:
            raise ValueError(f"{flag} requires a value.")
        return True, iterator.pop(0)
    if current.startswith(f"{flag}="):
        return True, current.split("=", 1)[1]
    return False, None


def _parse_args(argv: List[str]) -> Tuple[str, Path, List[str]]:
    """Extract wrapper flags and return (binary_path, lock_path, forward_args)."""
    remaining = list(argv)
    binary_path: str | None = None
    lock_path: Path | None = None
    forwarded: List[str] = []

    while remaining:
        current = remaining.pop(0)
        matched, value = _consume_flag(remaining, current, BINARY_FLAG)
        if matched:
            if not value:
                raise ValueError(f"{BINARY_FLAG} must include a value.")
            binary_path = value
            continue
        matched, value = _consume_flag(remaining, current, LOCK_FLAG)
        if matched:
            if not value:
                raise ValueError(f"{LOCK_FLAG} must include a value.")
            lock_path = Path(value)
            continue
        forwarded.append(current)

    if not binary_path:
        raise ValueError(
            f"{BINARY_FLAG} was not provided. "
            "Pass '--bm3d-binary /path/to/bm3d' via the defense configuration."
        )

    resolved_lock = lock_path or Path.cwd() / DEFAULT_LOCK_NAME
    return binary_path, resolved_lock.expanduser().resolve(), forwarded


def _acquire_lock(path: Path):
    """Return an open file handle with an exclusive lock applied (if supported)."""
    if fcntl is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def main(argv: List[str]) -> int:
    try:
        real_binary, lock_path, forwarded = _parse_args(argv)
    except ValueError as exc:
        print(f"[bm3d-wrapper] {exc}", file=sys.stderr)
        return 2

    lock_handle = _acquire_lock(lock_path)
    try:
        result = subprocess.run([real_binary, *forwarded], check=False)
        return result.returncode
    finally:
        if lock_handle is not None:
            lock_handle.close()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
