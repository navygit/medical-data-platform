"""Volume normalisation: orientation, spacing and intensity.

Three transforms, applied in this order and for these reasons:

1. **Reorient to RAS.** Scanners disagree about axis conventions. A model trained
   on mixed orientations learns the convention, not the anatomy.
2. **Resample to isotropic spacing.** A 3D convolution kernel has a fixed voxel
   footprint; if voxels are 1 mm on one scanner and 6 mm on another, the same
   kernel sees different anatomy. Labels use nearest-neighbour interpolation --
   linear interpolation of a label map invents classes that never existed.
3. **Z-score the intensities.** MRI intensity is in arbitrary units and not
   comparable across scanners, so per-volume standardisation over the foreground
   is the standard fix. (CT would use fixed HU windows instead, since HU *is*
   calibrated -- see the LiTS pipeline.)

Implemented against nibabel/scipy rather than MONAI transforms so the core
pipeline runs without the heavy ``[ml]`` extra installed.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from common.config import Config
from common.logging import get_logger
from common.metadata import ScanRecord
from common.storage import ensure_dir, file_size, relative_to, sha256_file

log = get_logger(__name__)


def reorient(img: nib.Nifti1Image, target: str = "RAS") -> nib.Nifti1Image:
    """Reorient a volume to the target anatomical convention.

    Uses nibabel's orientation transforms, which update the affine alongside the
    array, so the volume stays geometrically correct rather than merely permuted.
    """
    current = nib.io_orientation(img.affine)
    wanted = nib.orientations.axcodes2ornt(tuple(target))
    transform = nib.orientations.ornt_transform(current, wanted)
    return img.as_reoriented(transform)


def resample(
    img: nib.Nifti1Image,
    target_spacing: tuple[float, float, float],
    order: int,
) -> nib.Nifti1Image:
    """Resample a volume to the target voxel spacing.

    Args:
        img: Input volume.
        target_spacing: Desired spacing in mm.
        order: Interpolation order. Use ``0`` (nearest) for label maps and ``1``
            (linear) for images. Anything higher on a label map is a bug.

    Returns:
        The resampled volume with an updated affine.
    """
    from scipy.ndimage import zoom  # local import: scipy only needed here

    current = np.asarray(img.header.get_zooms()[:3], dtype=float)
    target = np.asarray(target_spacing, dtype=float)

    if np.allclose(current, target, atol=1e-3):
        return img

    factors = current / target
    data = np.asanyarray(img.dataobj, dtype=np.float32 if order > 0 else None)
    resampled = zoom(data, factors, order=order, mode="nearest")

    # Scale the direction columns of the affine so world coordinates survive.
    affine = img.affine.copy()
    affine[:3, :3] = affine[:3, :3] @ np.diag(target / current)

    if order == 0:
        resampled = np.rint(resampled).astype(np.uint8)
    return nib.Nifti1Image(resampled, affine)


def normalise_intensity(
    data: np.ndarray,
    method: str = "zscore",
    clip_percentiles: tuple[float, float] = (0.5, 99.5),
) -> np.ndarray:
    """Standardise intensities, computing statistics over the foreground only.

    Background is ~70% of a brain MRI volume. Including it drags the mean toward
    zero and shrinks the variance, so foreground contrast gets compressed into a
    narrow band. Masking to non-zero voxels avoids that.

    Args:
        data: Input volume.
        method: ``zscore``, ``minmax`` or ``none``.
        clip_percentiles: Percentiles to clip to before scaling, which limits the
            influence of hyper-intense artifacts.

    Returns:
        The normalised volume as float32.
    """
    if method == "none":
        return data.astype(np.float32)

    out = data.astype(np.float32)
    foreground = out[out > 0]
    if foreground.size == 0:
        return out

    lo, hi = np.percentile(foreground, clip_percentiles)
    out = np.clip(out, lo, hi)
    foreground = out[out > 0]

    if method == "zscore":
        std = float(foreground.std())
        if std == 0:
            return out - float(foreground.mean())
        return (out - float(foreground.mean())) / std

    if method == "minmax":
        span = float(foreground.max() - foreground.min())
        if span == 0:
            return np.zeros_like(out)
        return (out - float(foreground.min())) / span

    raise ValueError(f"unknown intensity normalisation: {method!r}")


def preprocess_record(rec: ScanRecord, cfg: Config) -> ScanRecord | None:
    """Normalise one series (and its mask) and write it to ``paths.processed``.

    Returns:
        An updated record pointing at the processed files, or ``None`` if the
        source could not be read.
    """
    raw_root = Path(cfg.paths.raw)
    out_root = ensure_dir(Path(cfg.paths.processed) / rec.patient_id)
    source = raw_root / rec.filepath

    try:
        img = nib.load(str(source))
    except Exception as exc:
        log.warning("preprocess.skip_unreadable", extra={"path": str(source), "error": str(exc)})
        return None

    spacing = tuple(cfg.preprocess.target_spacing)  # type: ignore[assignment]
    img = reorient(img, cfg.preprocess.target_orientation)
    img = resample(img, spacing, order=1)

    data = normalise_intensity(
        np.asanyarray(img.dataobj, dtype=np.float32),
        method=cfg.preprocess.intensity_norm,
        clip_percentiles=tuple(cfg.preprocess.clip_percentiles),  # type: ignore[arg-type]
    )
    out_img = nib.Nifti1Image(data, img.affine)
    out_path = out_root / f"{rec.patient_id}_{rec.modality}.nii.gz"
    nib.save(out_img, out_path)

    mask_out: str | None = None
    tumor_volume = rec.tumor_volume_mm3

    if rec.mask_path:
        try:
            mask_img = nib.load(str(raw_root / rec.mask_path))
            mask_img = reorient(mask_img, cfg.preprocess.target_orientation)
            mask_img = resample(mask_img, spacing, order=0)  # nearest: never blend labels
            mask_path = out_root / f"{rec.patient_id}_seg.nii.gz"
            nib.save(mask_img, mask_path)
            mask_out = relative_to(mask_path, Path(cfg.paths.processed))
            mask_data = np.asanyarray(mask_img.dataobj)
            tumor_volume = float(np.count_nonzero(mask_data) * np.prod(spacing))
        except Exception as exc:
            log.warning("preprocess.mask_failed",
                        extra={"path": rec.mask_path, "error": str(exc)})

    finite = data[np.isfinite(data)]
    return rec.model_copy(update={
        "filepath": relative_to(out_path, Path(cfg.paths.processed)),
        "mask_path": mask_out,
        "shape": [int(s) for s in data.shape],
        "voxel_spacing": list(spacing),
        "orientation": cfg.preprocess.target_orientation,
        "affine": np.asarray(out_img.affine, dtype=float).tolist(),
        "intensity_min": float(finite.min()) if finite.size else None,
        "intensity_max": float(finite.max()) if finite.size else None,
        "intensity_mean": float(finite.mean()) if finite.size else None,
        "intensity_std": float(finite.std()) if finite.size else None,
        "tumor_volume_mm3": tumor_volume,
        "file_size_bytes": file_size(out_path),
        "sha256": sha256_file(out_path),
    })


def preprocess(records: list[ScanRecord], cfg: Config) -> list[ScanRecord]:
    """Normalise every QC-passing record.

    Records with ``qc_status == "fail"`` are skipped: preprocessing a volume that
    QC already rejected wastes compute and risks it reaching a release.
    """
    eligible = [r for r in records if r.qc_status != "fail"]
    log.info("preprocess.start", extra={
        "n_total": len(records),
        "n_eligible": len(eligible),
        "target_spacing": cfg.preprocess.target_spacing,
        "norm": cfg.preprocess.intensity_norm,
    })

    out: list[ScanRecord] = []
    for rec in eligible:
        processed = preprocess_record(rec, cfg)
        if processed is not None:
            out.append(processed)

    log.info("preprocess.complete", extra={"n_written": len(out)})
    return out
