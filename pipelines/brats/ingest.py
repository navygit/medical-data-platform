"""BraTS ingestion: scan the raw tree and build a typed metadata manifest.

This is the first stage of the pipeline and the only one that touches the raw
landing zone. It is intentionally *defensive*: every volume is opened, hashed and
measured, and any file that fails to decode is recorded as a row with null
statistics rather than raising. QC decides what to do about it later.

That split matters. Ingest answers "what is there?"; QC answers "is it usable?".
Merging the two produces a pipeline that dies on the first bad file at 3 a.m.

Layout expected (real BraTS 2021 and the synthetic generator both match)::

    <raw>/BraTS2021_00001/BraTS2021_00001_t1.nii.gz
                          BraTS2021_00001_t1ce.nii.gz
                          BraTS2021_00001_t2.nii.gz
                          BraTS2021_00001_flair.nii.gz
                          BraTS2021_00001_seg.nii.gz
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import nibabel as nib
import numpy as np

from common.config import Config
from common.logging import get_logger
from common.metadata import ScanRecord
from common.storage import file_size, relative_to, sha256_file

log = get_logger(__name__)

MODALITY_PATTERN = re.compile(r"_(t1ce|t1|t2|flair|seg)\.nii(\.gz)?$", re.IGNORECASE)
IMAGE_MODALITIES = ("t1", "t1ce", "t2", "flair")

# BraTS label semantics. Label 3 is unused in the official challenge data.
LABEL_NAMES: dict[int, str] = {
    1: "necrotic_and_non_enhancing_core",
    2: "peritumoral_edema",
    4: "enhancing_tumor",
}


def parse_modality(path: Path) -> str | None:
    """Extract the modality token from a BraTS filename.

    Returns ``None`` for files that do not match the convention, which the caller
    logs and skips rather than guessing at.
    """
    match = MODALITY_PATTERN.search(path.name)
    return match.group(1).lower() if match else None


def _orientation(affine: np.ndarray) -> str:
    """Anatomical axis codes for an affine, e.g. ``"RAS"``."""
    return "".join(nib.aff2axcodes(affine))


def _tumor_volume_mm3(mask: np.ndarray, spacing: tuple[float, ...]) -> float:
    """Total volume of all non-zero labels in cubic millimetres."""
    return float(np.count_nonzero(mask) * np.prod(spacing))


def _probe_volume(path: Path) -> dict[str, object]:
    """Open a NIfTI and extract geometry plus intensity statistics.

    Returns a dict with null statistics when the file cannot be decoded, so a
    corrupt volume still produces a manifest row. A missing row would silently
    shrink the cohort; a null row is visible and auditable.
    """
    try:
        img = nib.load(str(path))
        data = np.asanyarray(img.dataobj, dtype=np.float32)
        affine = np.asarray(img.affine, dtype=float)
        finite = data[np.isfinite(data)]

        return {
            "shape": [int(s) for s in data.shape],
            "voxel_spacing": [float(s) for s in img.header.get_zooms()[:3]],
            "orientation": _orientation(affine),
            "affine": affine.tolist(),
            "intensity_min": float(finite.min()) if finite.size else None,
            "intensity_max": float(finite.max()) if finite.size else None,
            "intensity_mean": float(finite.mean()) if finite.size else None,
            "intensity_std": float(finite.std()) if finite.size else None,
            "readable": True,
        }
    except Exception as exc:
        log.warning("ingest.unreadable", extra={"path": str(path), "error": str(exc)})
        return {
            "shape": [],
            "voxel_spacing": [],
            "orientation": None,
            "affine": None,
            "intensity_min": None,
            "intensity_max": None,
            "intensity_mean": None,
            "intensity_std": None,
            "readable": False,
        }


def _mask_stats(mask_path: Path, spacing: list[float]) -> dict[str, object]:
    """Read a segmentation and summarise its labels and tumour burden."""
    try:
        mask = np.asanyarray(nib.load(str(mask_path)).dataobj)
        classes = sorted(int(c) for c in np.unique(mask) if c != 0)
        return {
            "label_classes": classes,
            "tumor_volume_mm3": _tumor_volume_mm3(mask, tuple(spacing) if spacing else (1, 1, 1)),
        }
    except Exception as exc:
        log.warning("ingest.mask_unreadable", extra={"path": str(mask_path), "error": str(exc)})
        return {"label_classes": [], "tumor_volume_mm3": None}


def scan_subject(subject_dir: Path, raw_root: Path) -> list[ScanRecord]:
    """Build one :class:`ScanRecord` per image modality for a subject.

    The segmentation is not given its own record; it is attached to each image
    record via ``mask_path``. Modelling it as a row would mean every downstream
    count of "how many scans" silently included label maps.
    """
    subject = subject_dir.name
    files: dict[str, Path] = {}

    for path in sorted(subject_dir.glob("*.nii*")):
        modality = parse_modality(path)
        if modality is None:
            log.warning("ingest.unrecognised_file", extra={"path": str(path)})
            continue
        files[modality] = path

    mask_path = files.get("seg")
    records: list[ScanRecord] = []

    for modality in IMAGE_MODALITIES:
        path = files.get(modality)
        if path is None:
            continue  # absence is reported by the cohort-level missing-modality check

        probe = _probe_volume(path)
        record = ScanRecord(
            patient_id=subject,
            study_uid=subject,
            series_uid=f"{subject}_{modality}",
            modality=modality,
            filepath=relative_to(path, raw_root),
            shape=probe["shape"],                      # type: ignore[arg-type]
            voxel_spacing=probe["voxel_spacing"],      # type: ignore[arg-type]
            orientation=probe["orientation"],          # type: ignore[arg-type]
            affine=probe["affine"],                    # type: ignore[arg-type]
            intensity_min=probe["intensity_min"],      # type: ignore[arg-type]
            intensity_max=probe["intensity_max"],      # type: ignore[arg-type]
            intensity_mean=probe["intensity_mean"],    # type: ignore[arg-type]
            intensity_std=probe["intensity_std"],      # type: ignore[arg-type]
            mask_available=mask_path is not None,
            mask_path=relative_to(mask_path, raw_root) if mask_path else None,
            file_size_bytes=file_size(path),
            sha256=sha256_file(path),
            ingested_at=datetime.now(UTC).isoformat(),
        )

        if mask_path is not None and probe["readable"]:
            stats = _mask_stats(mask_path, record.voxel_spacing)
            record = record.model_copy(update=stats)

        records.append(record)

    return records


def ingest(cfg: Config) -> list[ScanRecord]:
    """Scan the raw BraTS tree and return metadata for every image series.

    Args:
        cfg: Run config; ``cfg.paths.raw`` is the dataset root.

    Returns:
        One record per (subject, modality), sorted for reproducibility.

    Raises:
        FileNotFoundError: If the raw root does not exist.
    """
    raw_root = Path(cfg.paths.raw)
    if not raw_root.exists():
        raise FileNotFoundError(
            f"raw data root not found: {raw_root}\n"
            "Run: python scripts/generate_synthetic_data.py --dataset brats"
        )

    subject_dirs = sorted(d for d in raw_root.iterdir() if d.is_dir())
    if not subject_dirs:
        raise FileNotFoundError(
            f"no subject directories under {raw_root}. Expected BraTS layout "
            f"'<raw>/BraTS2021_00001/BraTS2021_00001_t1.nii.gz'.\n"
            "Run: python scripts/generate_synthetic_data.py --dataset brats"
        )

    log.info("ingest.start", extra={"root": str(raw_root), "n_subjects": len(subject_dirs)})

    records: list[ScanRecord] = []
    for subject_dir in subject_dirs:
        records.extend(scan_subject(subject_dir, raw_root))

    n_unreadable = sum(1 for r in records if r.intensity_mean is None)
    log.info("ingest.complete", extra={
        "n_subjects": len(subject_dirs),
        "n_series": len(records),
        "n_unreadable": n_unreadable,
        "total_bytes": sum(r.file_size_bytes or 0 for r in records),
    })
    return records
