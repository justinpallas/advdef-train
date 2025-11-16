#!/usr/bin/env python3
"""Serialize bm3d-gpu CLI invocations and capture failures."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX systems
    fcntl = None  # type: ignore[assignment]

BINARY_FLAG = "--bm3d-binary"
LOCK_FLAG = "--bm3d-lock"
LOG_DIR_FLAG = "--bm3d-log-dir"
PASSTHRU_FLAG = "--bm3d-passthru"
DEFAULT_LOCK_NAME = ".bm3d_cli.lock"
DEFAULT_LOG_DIR = ".bm3d_cli_logs"


def _consume_flag(iterator: List[str], current: str, flag: str) -> Tuple[bool, str | None]:
    """Return (matched, value) for flags of the form '--flag value' or '--flag=value'."""
    if current == flag:
        if not iterator:
            raise ValueError(f"{flag} requires a value.")
        return True, iterator.pop(0)
    if current.startswith(f"{flag}="):
        return True, current.split("=", 1)[1]
    return False, None


def _parse_args(argv: List[str]) -> Tuple[str, Path, Path, bool, List[str]]:
    """Extract wrapper flags and return (binary_path, lock_path, log_dir, passthru, forward_args)."""
    remaining = list(argv)
    binary_path: str | None = None
    lock_path: Path | None = None
    log_dir: Path | None = None
    passthru = False
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
        matched, value = _consume_flag(remaining, current, LOG_DIR_FLAG)
        if matched:
            if not value:
                raise ValueError(f"{LOG_DIR_FLAG} must include a value.")
            log_dir = Path(value)
            continue
        if current == PASSTHRU_FLAG:
            passthru = True
            continue
        forwarded.append(current)

    if not binary_path:
        raise ValueError(
            f"{BINARY_FLAG} was not provided. "
            "Pass '--bm3d-binary /path/to/bm3d' via the defense configuration."
        )

    resolved_lock = lock_path or Path.cwd() / DEFAULT_LOCK_NAME
    resolved_log_dir = log_dir or Path.cwd() / DEFAULT_LOG_DIR
    return (
        binary_path,
        resolved_lock.expanduser().resolve(),
        resolved_log_dir.expanduser().resolve(),
        passthru,
        forwarded,
    )


def _acquire_lock(path: Path):
    if fcntl is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def _log_failure(log_dir: Path, args: List[str], output: str, returncode: int) -> None:
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"bm3d_failure_{timestamp}_{abs(returncode)}.log"
    try:
        with log_path.open("w") as handle:
            handle.write(f"bm3d CLI failed with code {returncode}\n")
            handle.write("Command:\n")
            handle.write("bm3d " + " ".join(args) + "\n\n")
            handle.write("Output:\n")
            handle.write(output or "<no output>")
            handle.write("\n")
    except Exception:
        return
    print(f"[bm3d-wrapper] Command failed (code {returncode}). See {log_path}", file=sys.stderr)


def main(argv: List[str]) -> int:
    try:
        real_binary, lock_path, log_dir, passthru, forwarded = _parse_args(argv)
    except ValueError as exc:
        print(f"[bm3d-wrapper] {exc}", file=sys.stderr)
        return 2

    lock_handle = _acquire_lock(lock_path)
    try:
        run_kwargs = {}
        if not passthru:
            run_kwargs = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "text": True}
        result = subprocess.run([real_binary, *forwarded], check=False, **run_kwargs)
        if result.returncode != 0 and not passthru:
            _log_failure(log_dir, forwarded, result.stdout or "", result.returncode)
        return result.returncode
    finally:
        if lock_handle is not None:
            lock_handle.close()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
