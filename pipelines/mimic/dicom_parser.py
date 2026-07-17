"""DICOM header parsing and PHI auditing for chest radiographs.

Two responsibilities:

1. **Metadata extraction.** Pull the tags that define a cohort (view position,
   age, sex, manufacturer, institution) without loading pixel data. Reading
   headers only is ~1000x faster than decoding images, which is what makes it
   feasible to profile a 300k-study archive.

2. **PHI auditing.** MIMIC-CXR ships de-identified, but a platform that *assumes*
   that and is wrong has caused a HIPAA breach. :func:`audit_phi` re-checks every
   identifier tag on every file and reports what it finds. Verifying is cheap;
   assuming is not.

The de-identification itself (:func:`deidentify`) is included because a real
platform ingests from PACS as well as public archives, and PACS data is *not*
de-identified.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pydicom
from pydicom.dataset import Dataset

from common.logging import get_logger
from common.storage import file_size, relative_to, sha256_file

log = get_logger(__name__)

# DICOM tags carrying direct identifiers, per DICOM PS3.15 Annex E (Basic
# Application Level Confidentiality Profile). Not exhaustive -- a production
# deployment would use the full ~130-tag profile plus pixel-level burned-in text
# detection -- but these are the tags that actually carry PHI in practice.
PHI_TAGS: tuple[str, ...] = (
    "PatientName",
    "PatientBirthDate",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "OtherPatientIDs",
    "OtherPatientNames",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "OperatorsName",
    "InstitutionAddress",
    "AccessionNumber",
    "StudyDate",
    "StudyTime",
    "SeriesDate",
    "AcquisitionDate",
    "ContentDate",
    "PatientID",
)

# Tags that are safe to keep and that the cohort builder needs.
COHORT_TAGS: tuple[str, ...] = (
    "Modality",
    "ViewPosition",
    "BodyPartExamined",
    "PatientSex",
    "PatientAge",
    "Manufacturer",
    "ManufacturerModelName",
    "InstitutionName",
    "Rows",
    "Columns",
    "PhotometricInterpretation",
    "BitsStored",
    "PixelSpacing",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
)


@dataclass
class DicomRecord:
    """Header metadata for one radiograph."""

    study_id: str
    subject_id: str
    filepath: str
    modality: str | None = None
    view_position: str | None = None
    body_part: str | None = None
    sex: str | None = None
    age: float | None = None
    manufacturer: str | None = None
    institution: str | None = None
    rows: int | None = None
    columns: int | None = None
    pixel_spacing_mm: float | None = None
    study_uid: str | None = None
    series_uid: str | None = None
    sha256: str | None = None
    file_size_bytes: int | None = None
    phi_flags: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Flatten for DataFrame construction; ``phi_flags`` becomes a string."""
        data = self.__dict__.copy()
        data["phi_flags"] = ",".join(self.phi_flags or [])
        return data


def parse_age(raw: Any) -> float | None:
    """Parse a DICOM AS-format age string (``"058Y"``, ``"012M"``) into years.

    Returns ``None`` for absent or malformed values rather than guessing; an
    invented age would silently corrupt every downstream demographic audit.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        if text[-1].upper() in "YMWD":
            value, unit = float(text[:-1]), text[-1].upper()
            return value / {"Y": 1, "M": 12, "W": 52, "D": 365}[unit]
        return float(text)
    except (ValueError, KeyError):
        log.debug("dicom.bad_age", extra={"raw": text})
        return None


def audit_phi(ds: Dataset) -> list[str]:
    """Report which PHI tags carry a non-empty value.

    ``PatientID`` and ``StudyDate`` are excluded from the audit: MIMIC-CXR uses a
    surrogate patient ID and date-shifts studies into the year 2100+, both of
    which are accepted de-identification practice and are needed to link studies
    to reports and to order a patient timeline.

    Returns:
        Names of PHI tags that are populated. Empty means clean.
    """
    exempt = {"PatientID", "StudyDate", "StudyTime", "SeriesDate", "AcquisitionDate", "ContentDate"}
    flags: list[str] = []
    for tag in PHI_TAGS:
        if tag in exempt:
            continue
        value = getattr(ds, tag, None)
        if value is not None and str(value).strip():
            flags.append(tag)
    return flags


def deidentify(ds: Dataset, keep_uids: bool = True) -> Dataset:
    """Blank direct identifiers in place and return the dataset.

    Args:
        ds: Dataset to scrub.
        keep_uids: Retain study/series UIDs. Needed to preserve the link between
            images, reports and prior studies; a deployment that must break that
            link would re-map them through a secure lookup table rather than
            deleting them, since deletion is irreversible.

    Returns:
        The same dataset, mutated.
    """
    for tag in PHI_TAGS:
        if tag in ("PatientID", "StudyDate") and keep_uids:
            continue
        if tag in ds:
            setattr(ds, tag, "")
    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = "medical-data-platform basic profile (PS3.15 Annex E subset)"
    return ds


def parse_dicom(path: Path, root: Path) -> DicomRecord | None:
    """Read one DICOM header into a :class:`DicomRecord`.

    Pixel data is not decoded (``stop_before_pixels=True``), so this is fast
    enough to sweep an entire archive.

    Returns:
        The record, or ``None`` if the file is not readable as DICOM.
    """
    path = Path(path)
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
    except Exception as exc:
        log.warning("dicom.unreadable", extra={"path": str(path), "error": str(exc)})
        return None

    spacing = getattr(ds, "PixelSpacing", None)
    # MIMIC layout: files/p10/p10000000/s50000000/image.dcm
    study_id = path.parent.name
    subject_id = path.parent.parent.name

    return DicomRecord(
        study_id=study_id,
        subject_id=subject_id,
        filepath=relative_to(path, root),
        modality=getattr(ds, "Modality", None),
        view_position=getattr(ds, "ViewPosition", None),
        body_part=getattr(ds, "BodyPartExamined", None),
        sex=getattr(ds, "PatientSex", None) or None,
        age=parse_age(getattr(ds, "PatientAge", None)),
        manufacturer=getattr(ds, "Manufacturer", None),
        institution=getattr(ds, "InstitutionName", None),
        rows=getattr(ds, "Rows", None),
        columns=getattr(ds, "Columns", None),
        pixel_spacing_mm=float(spacing[0]) if spacing else None,
        study_uid=getattr(ds, "StudyInstanceUID", None),
        series_uid=getattr(ds, "SeriesInstanceUID", None),
        sha256=sha256_file(path),
        file_size_bytes=file_size(path),
        phi_flags=audit_phi(ds),
    )


def iter_dicoms(root: Path) -> Iterator[Path]:
    """Yield every ``.dcm`` file under ``root`` in sorted order."""
    yield from sorted(Path(root).rglob("*.dcm"))


def load_pixels(path: Path) -> Any:
    """Decode the pixel array for one DICOM, normalised to float32 in [0, 1].

    Separate from :func:`parse_dicom` so that metadata sweeps stay cheap; only
    the image model pays the decode cost.
    """
    import numpy as np

    ds = pydicom.dcmread(str(path))
    pixels = ds.pixel_array.astype("float32")

    # MONOCHROME1 stores white at low values (inverted relative to MONOCHROME2);
    # flip it so bone is bright under both conventions, otherwise a mixed-source
    # cohort presents the model with two visually opposite "modalities".
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixels = pixels.max() - pixels

    span = pixels.max() - pixels.min()
    if span > 0:
        pixels = (pixels - pixels.min()) / span
    return np.asarray(pixels, dtype="float32")
