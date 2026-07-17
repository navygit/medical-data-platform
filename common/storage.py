"""Filesystem helpers and content hashing.

Deliberately a thin local-filesystem layer behind a small interface. Swapping in
S3/MinIO means implementing :func:`sha256_file` and :func:`iter_files` against a
new backend; nothing else in the platform touches ``pathlib`` for data access.

Hashing is the backbone of dataset versioning: a release manifest records the
SHA-256 of every file it contains, so any later mutation of "immutable" raw data
is detectable rather than silent.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from pathlib import Path

from common.logging import get_logger

log = get_logger(__name__)

# 1 MiB: large enough to amortise syscall overhead, small enough that hashing a
# 2 GB volume never materialises the file in RAM.
_CHUNK_SIZE = 1024 * 1024


def sha256_file(path: str | Path, chunk_size: int = _CHUNK_SIZE) -> str:
    """Compute the SHA-256 of a file, streaming it in chunks.

    Args:
        path: File to hash.
        chunk_size: Read granularity in bytes.

    Returns:
        Lowercase hex digest.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Compute the SHA-256 of an in-memory buffer."""
    return hashlib.sha256(data).hexdigest()


def iter_files(root: str | Path, patterns: Sequence[str]) -> Iterator[Path]:
    """Yield files under ``root`` matching any glob in ``patterns``.

    Results are de-duplicated and sorted so that a scan is reproducible: two runs
    over the same tree must produce manifests in the same order, otherwise every
    release diff is noise.

    Args:
        root: Directory to walk.
        patterns: Glob patterns, e.g. ``["**/*.nii.gz", "**/*.nii"]``.

    Yields:
        Matching paths in sorted order.
    """
    root = Path(root)
    if not root.exists():
        log.warning("storage.root_missing", extra={"root": str(root)})
        return

    seen: set[Path] = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file() and path not in seen:
                seen.add(path)
    yield from sorted(seen)


def file_size(path: str | Path) -> int:
    """Size of ``path`` in bytes."""
    return Path(path).stat().st_size


def human_size(n_bytes: float) -> str:
    """Format a byte count for human-facing reports, e.g. ``1.4 GB``."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n_bytes) < 1024.0:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024.0
    return f"{n_bytes:.1f} PB"


def relative_to(path: str | Path, root: str | Path) -> str:
    """Return ``path`` relative to ``root`` with POSIX separators.

    Manifests must be portable across the Windows workstations where data is
    often curated and the Linux boxes where models train, so stored paths are
    always relative and always forward-slashed. Falls back to the absolute POSIX
    path when ``path`` lies outside ``root``.
    """
    path, root = Path(path), Path(root)
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if absent and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
