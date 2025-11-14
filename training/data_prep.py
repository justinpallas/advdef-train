"""Utilities to fetch the ImageNet train split when missing."""

from __future__ import annotations

import shutil
import tarfile
import urllib.request
from pathlib import Path

from .config import DatasetDownloadConfig


def ensure_imagenet_trainset(train_root: Path, downloads: DatasetDownloadConfig) -> None:
    """Ensure the ILSVRC2012 train split exists, downloading if permitted."""

    if _directory_has_contents(train_root):
        return

    if not downloads.auto_download:
        raise FileNotFoundError(
            f"ImageNet train directory {train_root} not found and auto-download disabled."
        )

    _prepare_trainset(train_root, downloads)


def _prepare_trainset(train_root: Path, downloads: DatasetDownloadConfig) -> None:
    archive_path = downloads.train_archive
    download_url = downloads.train_url
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    train_root.mkdir(parents=True, exist_ok=True)

    if archive_path.exists():
        print(f"[info] Using cached ImageNet train archive at {archive_path}.")
    else:
        _download_file(download_url, archive_path)

    staging_dir = downloads.download_dir / "imagenet_train_stage"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] Extracting {archive_path} into {staging_dir} (this will take a while).")
    with tarfile.open(archive_path, "r") as tar:
        _safe_extract_all(tar, staging_dir)

    class_archives = sorted(staging_dir.glob("*.tar"))
    total_archives = len(class_archives)
    if not class_archives:
        raise FileNotFoundError(
            f"No class archives found in {staging_dir} after extracting {archive_path}."
        )

    for idx, class_archive in enumerate(class_archives, start=1):
        class_name = class_archive.stem
        class_dir = train_root / class_name
        if class_dir.exists() and _directory_has_contents(class_dir):
            class_archive.unlink(missing_ok=True)
            continue

        class_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(class_archive, "r") as class_tar:
            _safe_extract_all(class_tar, class_dir)
        class_archive.unlink(missing_ok=True)

        if idx % 25 == 0 or idx == total_archives:
            print(f"[info] Extracted {idx}/{total_archives} ImageNet class archives.")

    shutil.rmtree(staging_dir, ignore_errors=True)

    if not _directory_has_contents(train_root):
        raise FileNotFoundError(
            f"Failed to materialize ImageNet train data inside {train_root}."
        )
    print(f"[info] ImageNet train split ready at {train_root}.")


def _directory_has_contents(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def _download_file(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> None:
    print(f"[info] Downloading {url} -> {destination}")
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        total = int(response.headers.get("Content-Length", "0"))
        downloaded = 0
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            output.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = downloaded / total * 100
                print(
                    f"\r[info] Downloaded {downloaded / (1024 ** 2):.1f} MB / {total / (1024 ** 2):.1f} MB ({percent:.1f}%)",
                    end="",
                    flush=True,
                )
    print()
    print(f"[info] Download complete: {destination}")


def _safe_extract_all(tar: tarfile.TarFile, destination: Path) -> None:
    def _is_within_directory(directory: Path, target: Path) -> bool:
        try:
            target.relative_to(directory)
        except ValueError:
            return False
        return True

    members = tar.getmembers()
    for member in members:
        member_path = destination / member.name
        if not _is_within_directory(destination, member_path.parent):
            raise RuntimeError(f"Unsafe path detected in archive: {member.name}")
    tar.extractall(path=destination)

