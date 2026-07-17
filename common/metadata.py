"""Typed metadata records and manifest I/O.

A *manifest* is the platform's central artifact: one row per imaging series,
carrying everything downstream stages need without re-opening the pixel data.
Reading a 200 MB volume to answer "how many T2 scans do we have?" does not scale;
reading a Parquet manifest does.

Manifests are written in three formats on purpose:

- **Parquet** -- typed, compressed, what the pipelines actually read.
- **CSV** -- what a clinical collaborator opens in Excel without asking for help.
- **JSON** -- what a reviewer diffs in a pull request.

Example:
    >>> records = [ScanRecord(patient_id="p1", modality="t1", ...)]
    >>> write_manifest(records, Path("metadata/manifest"))
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from common.logging import get_logger

log = get_logger(__name__)


class ScanRecord(BaseModel):
    """Metadata for a single imaging series.

    Field names deliberately mirror DICOM/NIfTI concepts so that a radiologist
    or PACS admin can read the manifest without a translation layer.
    """

    # --- identity -----------------------------------------------------------
    patient_id: str
    study_uid: str | None = None
    series_uid: str | None = None
    modality: str = Field(description="e.g. t1, t1ce, t2, flair, CT, DX")
    filepath: str

    # --- geometry -----------------------------------------------------------
    shape: list[int] = Field(default_factory=list)
    voxel_spacing: list[float] = Field(default_factory=list)
    orientation: str | None = None
    affine: list[list[float]] | None = None

    # --- intensity ----------------------------------------------------------
    intensity_min: float | None = None
    intensity_max: float | None = None
    intensity_mean: float | None = None
    intensity_std: float | None = None

    # --- labels -------------------------------------------------------------
    mask_available: bool = False
    mask_path: str | None = None
    tumor_volume_mm3: float | None = None
    label_classes: list[int] = Field(default_factory=list)

    # --- provenance / cohort ------------------------------------------------
    scanner: str | None = None
    institution: str | None = None
    age: float | None = None
    sex: str | None = None
    file_size_bytes: int | None = None
    sha256: str | None = None
    ingested_at: str | None = None

    # --- QC -----------------------------------------------------------------
    qc_status: str = "pending"
    qc_flags: list[str] = Field(default_factory=list)
    quality_score: float | None = None

    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("modality")
    @classmethod
    def _normalise_modality(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("sex")
    @classmethod
    def _normalise_sex(cls, v: str | None) -> str | None:
        if v is None:
            return None
        token = v.strip().upper()[:1]
        return {"M": "M", "F": "F"}.get(token, "U")

    @property
    def voxel_volume_mm3(self) -> float:
        """Volume of a single voxel; 0.0 when spacing is unknown."""
        if not self.voxel_spacing:
            return 0.0
        return float(np.prod(self.voxel_spacing))


def records_to_frame(records: Sequence[ScanRecord]) -> pd.DataFrame:
    """Flatten records into a DataFrame with Parquet-safe column types.

    List and dict columns are JSON-encoded rather than left as Python objects:
    Arrow can store nested types, but round-tripping them through CSV cannot, and
    the three output formats must agree.
    """
    if not records:
        return pd.DataFrame(columns=list(ScanRecord.model_fields))

    rows = [r.model_dump(mode="json") for r in records]
    frame = pd.DataFrame(rows)

    for column in frame.columns:
        if frame[column].apply(lambda x: isinstance(x, (list, dict))).any():
            frame[column] = frame[column].apply(
                lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x
            )
    return frame


def write_manifest(
    records: Sequence[ScanRecord],
    stem: Path,
    formats: Iterable[str] = ("parquet", "csv", "json"),
) -> dict[str, Path]:
    """Write a manifest to disk in several formats.

    Args:
        records: The scan records to persist.
        stem: Output path *without* extension, e.g. ``metadata/manifest``.
        formats: Any of ``parquet``, ``csv``, ``json``.

    Returns:
        Mapping of format name to the path written.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    frame = records_to_frame(records)
    written: dict[str, Path] = {}

    for fmt in formats:
        if fmt == "parquet":
            out = stem.with_suffix(".parquet")
            frame.to_parquet(out, index=False)
        elif fmt == "csv":
            out = stem.with_suffix(".csv")
            frame.to_csv(out, index=False)
        elif fmt == "json":
            out = stem.with_suffix(".json")
            out.write_text(
                json.dumps([r.model_dump(mode="json") for r in records], indent=2),
                encoding="utf-8",
            )
        else:
            raise ValueError(f"unsupported manifest format: {fmt!r}")
        written[fmt] = out

    log.info(
        "manifest.written",
        extra={"n_records": len(records), "paths": {k: str(v) for k, v in written.items()}},
    )
    return written


def read_manifest(path: Path) -> pd.DataFrame:
    """Read a manifest from Parquet, CSV or JSON based on file suffix."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".json":
        return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    raise ValueError(f"unsupported manifest suffix: {path.suffix!r}")


def decode_list_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """Decode a JSON-encoded list column back into Python lists.

    Handles frames that came from Parquet (already native lists) and from
    CSV (JSON strings) identically, so callers need not care about provenance.
    """

    def _decode(value: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, list) else [parsed]
            except json.JSONDecodeError:
                return []
        return []

    return frame[column].apply(_decode)
