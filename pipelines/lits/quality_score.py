"""Per-study quality scoring, 0-100.

Rationale
---------
Binary pass/fail QC forces a false choice. A study with slightly coarse spacing
is not equivalent to a corrupt file, but a boolean gate treats them the same. A
graded score lets the release policy pick a threshold per use case: a
segmentation training set might demand >= 80, an exploratory cohort >= 50.

The score is a **weighted sum of independent components**, each in [0, 1]:

=====================  ======  =================================================
Component              Weight  Penalises
=====================  ======  =================================================
integrity               0.30   unreadable / non-finite / constant volumes
metadata_completeness   0.15   missing demographics and acquisition parameters
spacing_consistency     0.15   anisotropic or coarse voxels
noise                   0.15   low signal-to-noise ratio
contrast                0.15   poor dynamic range in the tissue of interest
slice_continuity        0.10   dropped or duplicated slices
=====================  ======  =================================================

Weights sum to 1.0 and live in one dict so the trade-off is reviewable rather
than scattered through the code. Integrity dominates because a corrupt volume is
worthless regardless of how good its metadata looks.

Every component returns *both* a score and a human-readable reason, so a
rejected study can be explained to the clinician who submitted it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from common.logging import get_logger

log = get_logger(__name__)

WEIGHTS: dict[str, float] = {
    "integrity": 0.30,
    "metadata_completeness": 0.15,
    "spacing_consistency": 0.15,
    "noise": 0.15,
    "contrast": 0.15,
    "slice_continuity": 0.10,
}

# Metadata fields a study needs before it can be used for cohort selection.
REQUIRED_METADATA: tuple[str, ...] = (
    "patient_id", "age", "sex", "institution", "scanner", "voxel_spacing", "shape",
)


@dataclass
class QualityScore:
    """Graded quality assessment for one study."""

    patient_id: str
    overall: float
    components: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    @property
    def grade(self) -> str:
        """Letter grade, for at-a-glance reading in the dashboard."""
        if self.overall >= 90:
            return "A"
        if self.overall >= 80:
            return "B"
        if self.overall >= 65:
            return "C"
        if self.overall >= 50:
            return "D"
        return "F"

    def to_dict(self) -> dict[str, Any]:
        """Flatten for DataFrame construction."""
        return {
            "patient_id": self.patient_id,
            "quality_score": round(self.overall, 2),
            "grade": self.grade,
            **{f"q_{k}": round(v, 3) for k, v in self.components.items()},
            "reasons": "; ".join(self.reasons),
        }


def score_integrity(volume: np.ndarray | None) -> tuple[float, str | None]:
    """1.0 for a readable, finite, non-constant volume; 0.0 otherwise."""
    if volume is None:
        return 0.0, "volume unreadable"
    finite = np.isfinite(volume)
    if not finite.all():
        bad = int((~finite).sum())
        return 0.0, f"{bad} non-finite voxels"
    if float(volume.std()) == 0.0:
        return 0.0, "volume is constant (no signal)"
    return 1.0, None


def score_metadata(row: dict[str, Any]) -> tuple[float, str | None]:
    """Fraction of required metadata fields that are populated."""
    missing = [
        field_name for field_name in REQUIRED_METADATA
        if row.get(field_name) in (None, "", [], float("nan"))
        or (isinstance(row.get(field_name), float) and np.isnan(row[field_name]))
    ]
    score = 1.0 - len(missing) / len(REQUIRED_METADATA)
    return score, f"missing metadata: {missing}" if missing else None


def score_spacing(spacing: list[float] | None, max_mm: float = 5.0) -> tuple[float, str | None]:
    """Penalise anisotropic and coarse voxels.

    Anisotropy matters more than absolute size for 3D CNNs: a 5x0.8x0.8 mm voxel
    means the kernel's receptive field is six times larger through-plane than
    in-plane, so the network sees a distorted anatomy.
    """
    if not spacing or len(spacing) < 3:
        return 0.0, "voxel spacing unavailable"

    array = np.asarray(spacing, dtype=float)
    if (array <= 0).any():
        return 0.0, f"invalid spacing {spacing}"

    anisotropy = float(array.max() / array.min())
    aniso_score = float(np.clip(1.0 - (anisotropy - 1.0) / 5.0, 0.0, 1.0))
    coarse_score = float(np.clip(1.0 - (array.max() - 1.0) / max_mm, 0.0, 1.0))
    score = 0.5 * aniso_score + 0.5 * coarse_score

    reason = None
    if anisotropy > 3:
        reason = f"anisotropic voxels (ratio {anisotropy:.1f}:1)"
    elif array.max() > max_mm:
        reason = f"coarse spacing ({array.max():.1f} mm)"
    return score, reason


def score_noise(volume: np.ndarray | None) -> tuple[float, str | None]:
    """Estimate SNR from the foreground and map it onto [0, 1].

    Foreground is taken as voxels above the 60th percentile -- for abdominal CT
    that is roughly organ tissue rather than air or table. Crude, but it is a
    *relative* quality signal for ranking studies within a cohort, not a
    physics-grade measurement.
    """
    if volume is None or volume.size == 0:
        return 0.0, "volume unreadable"

    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return 0.0, "no finite voxels"

    threshold = np.percentile(finite, 60)
    foreground = finite[finite > threshold]
    if foreground.size < 10 or float(foreground.std()) == 0:
        return 0.5, "insufficient foreground to estimate SNR"

    snr = float(abs(foreground.mean()) / (foreground.std() + 1e-8))
    score = float(np.clip(snr / 10.0, 0.0, 1.0))  # SNR >= 10 is "good enough"
    return score, f"low SNR ({snr:.1f})" if snr < 3 else None


def score_contrast(volume: np.ndarray | None) -> tuple[float, str | None]:
    """Score dynamic range via the interquartile spread of the foreground.

    A study where liver and lesion sit within a few HU of each other is
    unsegmentable no matter how clean the file is.
    """
    if volume is None or volume.size == 0:
        return 0.0, "volume unreadable"

    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return 0.0, "no finite voxels"

    p25, p75 = np.percentile(finite, [25, 75])
    spread = float(p75 - p25)
    full = float(finite.max() - finite.min())
    if full == 0:
        return 0.0, "zero dynamic range"

    score = float(np.clip(spread / (0.25 * full), 0.0, 1.0))
    return score, "low tissue contrast" if score < 0.3 else None


def score_slice_continuity(volume: np.ndarray | None, axis: int = 0) -> tuple[float, str | None]:
    """Detect dropped or duplicated slices via inter-slice differences.

    A duplicated slice gives a near-zero difference from its neighbour; a dropped
    slice gives an outlier spike. Both are common in archives assembled from
    partial PACS transfers, and both are invisible to a file-level checksum.
    """
    if volume is None or volume.ndim != 3 or volume.shape[axis] < 3:
        return 1.0, None  # nothing to check; do not penalise

    slices = np.moveaxis(volume, axis, 0)
    diffs = np.array([
        float(np.abs(slices[i + 1] - slices[i]).mean()) for i in range(len(slices) - 1)
    ])
    if not np.isfinite(diffs).all() or diffs.mean() == 0:
        return 0.0, "identical slices throughout"

    n_duplicate = int((diffs < 1e-6).sum())
    median = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - median))) + 1e-8
    n_jumps = int((np.abs(diffs - median) > 8 * mad).sum())

    penalty = 0.15 * n_duplicate + 0.10 * n_jumps
    score = float(np.clip(1.0 - penalty, 0.0, 1.0))

    reasons = []
    if n_duplicate:
        reasons.append(f"{n_duplicate} duplicate slice(s)")
    if n_jumps:
        reasons.append(f"{n_jumps} discontinuity(ies)")
    return score, ", ".join(reasons) if reasons else None


def score_study(row: dict[str, Any], volume: np.ndarray | None) -> QualityScore:
    """Compute the weighted quality score for one study.

    Args:
        row: Metadata dict; needs the keys in :data:`REQUIRED_METADATA`.
        volume: The decoded image array, or ``None`` if unreadable.

    Returns:
        The assembled :class:`QualityScore`.
    """
    results: dict[str, tuple[float, str | None]] = {
        "integrity": score_integrity(volume),
        "metadata_completeness": score_metadata(row),
        "spacing_consistency": score_spacing(row.get("voxel_spacing")),
        "noise": score_noise(volume),
        "contrast": score_contrast(volume),
        "slice_continuity": score_slice_continuity(volume),
    }

    components = {name: value for name, (value, _) in results.items()}
    reasons = [reason for _, (_, reason) in results.items() if reason]
    overall = 100.0 * sum(components[name] * weight for name, weight in WEIGHTS.items())

    return QualityScore(
        patient_id=str(row.get("patient_id", "unknown")),
        overall=overall,
        components=components,
        reasons=reasons,
    )


def score_cohort(rows: list[dict[str, Any]], volume_root: Path) -> list[QualityScore]:
    """Score every study in a cohort, loading each volume once.

    An unreadable volume yields a score rather than an exception -- a governance
    report that omits the broken studies is worse than useless.
    """
    import nibabel as nib

    scores: list[QualityScore] = []
    for row in rows:
        volume: np.ndarray | None = None
        path = row.get("volume_path")
        if path:
            try:
                volume = np.asanyarray(
                    nib.load(str(Path(volume_root) / str(path))).dataobj, dtype=np.float32
                )
            except Exception as exc:
                log.warning("quality.unreadable",
                            extra={"path": str(path), "error": str(exc)})
        scores.append(score_study(row, volume))

    grades = [s.grade for s in scores]
    log.info("quality.cohort_scored", extra={
        "n_studies": len(scores),
        "mean_score": round(float(np.mean([s.overall for s in scores])), 1) if scores else 0.0,
        "grades": {g: grades.count(g) for g in sorted(set(grades))},
    })
    return scores
