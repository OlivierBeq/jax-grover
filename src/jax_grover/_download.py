"""Dependency-free (stdlib-only) file download helper for fetching GitHub
release assets on demand."""
from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path


def sha256sum(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_reporthook(label: str):
    """urlretrieve reporthook that only prints when the percentage changes."""
    last_pct = [-1]

    def reporthook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = min(block_num * block_size, total_size)
        pct = downloaded * 100 // total_size
        if pct == last_pct[0]:
            return
        last_pct[0] = pct
        print(
            f"\rDownloading {label}: {pct:3d}% "
            f"({downloaded // (1 << 20)} / {total_size // (1 << 20)} MB)",
            end="",
            flush=True,
        )

    return reporthook


def download_file(
    url: str,
    dest: str | Path,
    expected_sha256: str | None = None,
    force: bool = False,
    label: str | None = None,
) -> Path:
    """Download ``url`` to ``dest``, verifying its checksum if given.

    No-op if ``dest`` exists (unless ``force``). Written atomically (temp
    file, then renamed), so a failed download never leaves a corrupt file.
    On checksum mismatch the bad download is removed and ``RuntimeError``
    is raised.
    """
    dest = Path(dest)
    if dest.exists() and not force:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=dest.name + ".", suffix=".part")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        urllib.request.urlretrieve(url, tmp_path, reporthook=_make_reporthook(label or dest.name))
        print()  # newline after the progress line
        if expected_sha256 is not None:
            actual_sha256 = sha256sum(tmp_path)
            if actual_sha256 != expected_sha256:
                raise RuntimeError(f"Downloaded file checksum mismatch for {url}: expected {expected_sha256}, got {actual_sha256}")
        tmp_path.replace(dest)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return dest
