"""Dataset versioning and lineage.

A *release* is an immutable, content-addressed description of a dataset at one
point in time: which files, which hashes, which QC thresholds, which splits, and
which parent release it derived from. Releases are the unit that a researcher
cites in a paper and that an auditor reconstructs years later.

The design mirrors what DVC/LakeFS give you, expressed in plain JSON so the
artifacts stay readable in a pull request and survive the tool going out of
fashion. ``dvc.yaml`` in the repo root wires the same stages to DVC for teams
that want remote storage.

Key property: the ``dataset_hash`` is derived from the sorted per-file hashes, so
two releases with identical content have identical hashes regardless of when or
where they were built.

Example:
    >>> release = create_release("brats", "v1.0.0", records, cfg, splits)
    >>> write_release(release, Path("releases"))
"""

from __future__ import annotations

import json
import platform
import subprocess
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from common.config import Config, dump_config
from common.logging import get_logger
from common.metadata import ScanRecord
from common.storage import sha256_bytes

log = get_logger(__name__)

SCHEMA_VERSION = "1.0"


@dataclass
class FileEntry:
    """One file inside a release."""

    path: str
    sha256: str
    size_bytes: int
    patient_id: str
    modality: str


@dataclass
class LineageNode:
    """One transformation step in the provenance chain."""

    stage: str
    description: str
    timestamp: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Release:
    """An immutable dataset release."""

    dataset: str
    version: str
    schema_version: str
    created_at: str
    dataset_hash: str
    n_files: int
    n_subjects: int
    total_bytes: int
    files: list[FileEntry]
    splits: dict[str, list[str]]
    class_balance: dict[str, int]
    modality_counts: dict[str, int]
    qc_summary: dict[str, Any]
    config_snapshot: dict[str, Any]
    lineage: list[LineageNode]
    parent_version: str | None
    environment: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-able dict."""
        return asdict(self)


def _git_revision() -> str:
    """Best-effort current git SHA, or ``"unknown"`` outside a repo.

    Recorded so a release can be tied to the exact pipeline code that built it.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def compute_dataset_hash(entries: Sequence[FileEntry]) -> str:
    """Derive a single content hash for a set of files.

    Hashes the sorted ``path:sha256`` pairs, so the result is invariant to file
    discovery order but sensitive to any content or naming change.
    """
    payload = "\n".join(sorted(f"{e.path}:{e.sha256}" for e in entries))
    return sha256_bytes(payload.encode("utf-8"))


def build_file_entries(records: Sequence[ScanRecord]) -> list[FileEntry]:
    """Build release file entries from scan records.

    Records lacking a hash are skipped with a warning rather than silently
    included: an unhashed file cannot be verified, so it must not contribute to
    a release that claims verifiability.
    """
    entries: list[FileEntry] = []
    for rec in records:
        if not rec.sha256:
            log.warning("versioning.skip_unhashed", extra={"path": rec.filepath})
            continue
        entries.append(FileEntry(
            path=rec.filepath,
            sha256=rec.sha256,
            size_bytes=rec.file_size_bytes or 0,
            patient_id=rec.patient_id,
            modality=rec.modality,
        ))
    return entries


def create_release(
    dataset: str,
    version: str,
    records: Sequence[ScanRecord],
    cfg: Config,
    splits: dict[str, list[str]] | None = None,
    lineage: Sequence[LineageNode] | None = None,
    parent_version: str | None = None,
    qc_summary: dict[str, Any] | None = None,
) -> Release:
    """Assemble a :class:`Release` from QC'd records.

    Args:
        dataset: Dataset name, e.g. ``"brats"``.
        version: Semantic version string, e.g. ``"v1.0.0"``.
        records: Records included in this release.
        cfg: Config used to produce them; snapshotted verbatim.
        splits: Mapping of split name to subject IDs.
        lineage: Ordered transformation steps.
        parent_version: The release this one derives from.
        qc_summary: Roll-up from :func:`common.qc.summarise`.

    Returns:
        The assembled release.
    """
    entries = build_file_entries(records)

    return Release(
        dataset=dataset,
        version=version,
        schema_version=SCHEMA_VERSION,
        created_at=datetime.now(UTC).isoformat(),
        dataset_hash=compute_dataset_hash(entries),
        n_files=len(entries),
        n_subjects=len({r.patient_id for r in records}),
        total_bytes=sum(e.size_bytes for e in entries),
        files=entries,
        splits=splits or {},
        class_balance=dict(Counter(
            str(c) for r in records for c in (r.label_classes or [])
        )),
        modality_counts=dict(Counter(r.modality for r in records)),
        qc_summary=qc_summary or {},
        config_snapshot=dump_config(cfg),
        lineage=list(lineage or []),
        parent_version=parent_version,
        environment={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git_revision": _git_revision(),
        },
    )


def write_release(release: Release, root: Path) -> Path:
    """Write a release to ``root/<version>/manifest.json`` plus a hash sidecar.

    Refuses to overwrite an existing manifest. Releases are immutable by
    contract; silently rewriting one would break every citation of it.

    Raises:
        FileExistsError: If the release directory already has a manifest.
    """
    root = Path(root)
    release_dir = root / release.version
    manifest = release_dir / "manifest.json"

    if manifest.exists():
        raise FileExistsError(
            f"release {release.version} already exists at {manifest}. "
            "Releases are immutable -- bump the version instead."
        )

    release_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(release.to_dict(), indent=2), encoding="utf-8")
    (release_dir / "SHA256SUMS").write_text(
        "\n".join(f"{e.sha256}  {e.path}" for e in release.files) + "\n",
        encoding="utf-8",
    )

    _update_index(root, release)
    log.info("release.written", extra={
        "dataset": release.dataset,
        "version": release.version,
        "hash": release.dataset_hash[:12],
        "n_files": release.n_files,
    })
    return manifest


def _update_index(root: Path, release: Release) -> None:
    """Append a release to the human-readable index at ``root/releases.json``."""
    index_path = root / "releases.json"
    index: list[dict[str, Any]] = []
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("release.index_corrupt_rebuilding", extra={"path": str(index_path)})
            index = []

    index = [row for row in index if row.get("version") != release.version]
    index.append({
        "dataset": release.dataset,
        "version": release.version,
        "created_at": release.created_at,
        "dataset_hash": release.dataset_hash,
        "n_files": release.n_files,
        "n_subjects": release.n_subjects,
        "parent_version": release.parent_version,
    })
    index.sort(key=lambda row: row["created_at"])
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")


def read_release(path: Path) -> Release:
    """Load a release manifest from disk."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data["files"] = [FileEntry(**f) for f in data["files"]]
    data["lineage"] = [LineageNode(**n) for n in data["lineage"]]
    return Release(**data)


def verify_release(release: Release, root: Path) -> list[str]:
    """Re-hash every file in a release and report drift.

    This is the check that turns "we version our data" into something an auditor
    can trust: it proves the bytes on disk still match the manifest.

    Args:
        release: The release to verify.
        root: Directory that ``release.files[*].path`` is relative to.

    Returns:
        Human-readable problem descriptions; empty means verified.
    """
    from common.storage import sha256_file  # local import keeps module import cheap

    problems: list[str] = []
    for entry in release.files:
        path = Path(root) / entry.path
        if not path.exists():
            problems.append(f"missing: {entry.path}")
            continue
        actual = sha256_file(path)
        if actual != entry.sha256:
            problems.append(
                f"hash mismatch: {entry.path} "
                f"(manifest {entry.sha256[:12]}, disk {actual[:12]})"
            )

    log.info("release.verified", extra={
        "version": release.version,
        "n_files": release.n_files,
        "n_problems": len(problems),
    })
    return problems


def node(stage: str, description: str, **params: Any) -> LineageNode:
    """Convenience constructor for a lineage node timestamped now."""
    return LineageNode(
        stage=stage,
        description=description,
        timestamp=datetime.now(UTC).isoformat(),
        params=params,
    )
