"""Generate synthetic stand-ins for BraTS, MIMIC-CXR and LiTS.

Why this exists
---------------
The real datasets are credentialed downloads measured in hundreds of gigabytes.
A reviewer cloning this repository cannot run the pipelines against them, and CI
certainly cannot. So the pipelines are written against a *schema*, not against a
specific download, and this script emits data in that schema: real NIfTI volumes
with real affines, real DICOM files with real tags, and free-text reports with
the phrasing conventions of actual radiology reports.

The same pipeline code runs on synthetic and real data. Point the config's
``paths.raw`` at a real download and nothing else changes.

Injected defects
----------------
This is not a happy-path generator. It deliberately plants the failure modes a
real ingest hits, so QC has something to find and the tests can assert it finds
exactly these:

===============================  ==========================================
Defect                           Planted as
===============================  ==========================================
Corrupt NIfTI                    truncated gzip stream
Empty mask                       all-background segmentation
Wrong orientation                LAS-oriented affine instead of RAS
Missing modality                 subject with no FLAIR
Duplicate subject                byte-identical copy under a new ID
Constant volume                  zero-variance image
Coarse spacing                   6 mm slice thickness
===============================  ==========================================

Usage:
    python scripts/generate_synthetic_data.py --dataset all --n 12
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from common.logging import configure_logging, get_logger

log = get_logger(__name__)

# Subject indices that receive each planted defect. Fixed rather than random so
# the tests can assert on specific subjects and stay deterministic.
DEFECT_PLAN: dict[str, int] = {
    "corrupt": 3,
    "empty_mask": 5,
    "wrong_orientation": 6,
    "missing_modality": 7,
    "constant_volume": 8,
    "coarse_spacing": 9,
}

BRATS_MODALITIES = ("t1", "t1ce", "t2", "flair")
SCANNERS = ("Siemens Skyra 3T", "GE Discovery 750", "Philips Ingenia 1.5T")
INSTITUTIONS = ("Site-A", "Site-B", "Site-C")


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _brain_phantom(shape: tuple[int, int, int], rng: np.random.Generator) -> np.ndarray:
    """A crude head-shaped intensity volume: ellipsoid brain plus noise."""
    zz, yy, xx = np.meshgrid(
        *[np.linspace(-1, 1, s) for s in shape], indexing="ij"
    )
    radius = (xx / 0.75) ** 2 + (yy / 0.9) ** 2 + (zz / 0.85) ** 2
    brain = np.clip(1.0 - radius, 0, 1)
    tissue = brain * (0.6 + 0.4 * np.sin(6 * xx) * np.cos(5 * yy))
    volume = np.where(brain > 0, 300 + 400 * tissue, 0.0)
    volume += rng.normal(0, 12, shape) * (brain > 0)
    return np.clip(volume, 0, None).astype(np.float32)


def _tumor_mask(
    shape: tuple[int, int, int], rng: np.random.Generator, empty: bool = False
) -> np.ndarray:
    """A concentric multi-label lesion, mirroring BraTS label semantics."""
    mask = np.zeros(shape, dtype=np.uint8)
    if empty:
        return mask

    centre = np.array([rng.uniform(0.3, 0.7) * s for s in shape])
    zz, yy, xx = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    dist = np.sqrt(
        (zz - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (xx - centre[2]) ** 2
    )
    outer = rng.uniform(0.14, 0.26) * min(shape)
    mask[dist < outer] = 2          # peritumoral edema
    mask[dist < outer * 0.62] = 4   # enhancing tumor
    mask[dist < outer * 0.30] = 1   # necrotic core
    return mask


def _affine(spacing: tuple[float, float, float], orientation: str = "RAS") -> np.ndarray:
    """Build a diagonal affine for the requested spacing and orientation."""
    affine = np.eye(4, dtype=np.float64)
    signs = {"RAS": (1, 1, 1), "LAS": (-1, 1, 1), "LPS": (-1, -1, 1)}[orientation]
    for i, (s, sign) in enumerate(zip(spacing, signs, strict=True)):
        affine[i, i] = s * sign
    affine[:3, 3] = [-90.0, -110.0, -70.0]
    return affine


def _corrupt(path: Path) -> None:
    """Truncate a gzip stream so decompression fails mid-read.

    Models the real failure: a transfer that dropped, leaving a file that exists
    and has a plausible size but cannot be decoded.
    """
    data = path.read_bytes()
    path.write_bytes(data[: max(64, len(data) // 3)])


# --------------------------------------------------------------------------- #
# BraTS                                                                        #
# --------------------------------------------------------------------------- #


def generate_brats(root: Path, n_subjects: int, shape: tuple[int, int, int], seed: int) -> int:
    """Emit a BraTS-style tree: ``<root>/BraTS2021_00001/*_t1.nii.gz`` etc."""
    import nibabel as nib

    root.mkdir(parents=True, exist_ok=True)
    rng = _rng(seed)
    written = 0

    for i in range(1, n_subjects + 1):
        subject = f"BraTS2021_{i:05d}"
        subject_dir = root / subject
        subject_dir.mkdir(exist_ok=True)

        spacing = (6.0, 1.0, 1.0) if i == DEFECT_PLAN["coarse_spacing"] else (1.0, 1.0, 1.0)
        orientation = "LAS" if i == DEFECT_PLAN["wrong_orientation"] else "RAS"
        affine = _affine(spacing, orientation)

        base = _brain_phantom(shape, rng)
        mask = _tumor_mask(shape, rng, empty=(i == DEFECT_PLAN["empty_mask"]))

        for modality in BRATS_MODALITIES:
            if modality == "flair" and i == DEFECT_PLAN["missing_modality"]:
                continue

            if i == DEFECT_PLAN["constant_volume"] and modality == "t2":
                volume = np.full(shape, 500.0, dtype=np.float32)
            else:
                # Give each modality its own contrast so they are not clones.
                gain = {"t1": 1.0, "t1ce": 1.15, "t2": 0.8, "flair": 0.9}[modality]
                volume = base * gain + rng.normal(0, 6, shape).astype(np.float32)
                if modality == "t1ce":
                    volume[mask == 4] *= 1.6  # enhancing rim takes up contrast
                if modality in ("t2", "flair"):
                    volume[mask == 2] *= 1.4  # edema is bright on T2/FLAIR

            path = subject_dir / f"{subject}_{modality}.nii.gz"
            nib.save(nib.Nifti1Image(volume.astype(np.float32), affine), path)
            if i == DEFECT_PLAN["corrupt"] and modality == "t1":
                _corrupt(path)
            written += 1

        nib.save(nib.Nifti1Image(mask, affine), subject_dir / f"{subject}_seg.nii.gz")
        written += 1

    # Duplicate subject: byte-identical copy under a new ID. Only detectable by
    # content hash, which is exactly the point of hashing in the first place.
    source = root / "BraTS2021_00001"
    dup = root / f"BraTS2021_{n_subjects + 1:05d}"
    if source.exists() and not dup.exists():
        dup.mkdir()
        for src in source.glob("*.nii.gz"):
            shutil.copy2(src, dup / src.name.replace("BraTS2021_00001", dup.name))
            written += 1

    log.info("synth.brats.done", extra={"root": str(root), "n_files": written})
    return written


# --------------------------------------------------------------------------- #
# LiTS                                                                         #
# --------------------------------------------------------------------------- #


def generate_lits(root: Path, n_subjects: int, shape: tuple[int, int, int], seed: int) -> int:
    """Emit a LiTS-style tree of abdominal CT volumes plus liver/lesion masks.

    Intensities are in Hounsfield units and demographics are skewed on purpose:
    Site-A is over-represented and the sex balance is uneven, so the bias audit
    in case study 3 has genuine imbalance to detect rather than a flat
    distribution that makes the audit look like decoration.
    """
    import nibabel as nib
    import pandas as pd

    volumes_dir = root / "volumes"
    labels_dir = root / "labels"
    volumes_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    rng = _rng(seed + 1)
    rows = []
    written = 0

    for i in range(n_subjects):
        # Skewed cohort: Site-A dominates, males over-represented, age bimodal.
        institution = INSTITUTIONS[0] if rng.random() < 0.6 else INSTITUTIONS[rng.integers(1, 3)]
        sex = "M" if rng.random() < 0.68 else "F"
        age = float(np.clip(rng.normal(64 if sex == "M" else 55, 11), 19, 92))
        spacing = (float(rng.choice([1.0, 2.5, 5.0])), 0.8, 0.8)

        body = _brain_phantom(shape, rng) / 400.0
        ct = (body * 120 - 60).astype(np.float32)          # soft tissue ~ -60..60 HU
        liver = (body > 0.55).astype(np.uint8)
        ct[liver > 0] = rng.normal(105, 8, int(liver.sum()))  # contrast-enhanced liver

        mask = liver.copy()
        lesion = _tumor_mask(shape, rng)
        mask[(lesion > 0) & (liver > 0)] = 2
        ct[mask == 2] = rng.normal(45, 10, int((mask == 2).sum()))

        affine = _affine(spacing)
        vol_path = volumes_dir / f"volume-{i}.nii.gz"
        nib.save(nib.Nifti1Image(ct, affine), vol_path)
        nib.save(nib.Nifti1Image(mask, affine), labels_dir / f"segmentation-{i}.nii.gz")
        written += 2

        if i == DEFECT_PLAN["corrupt"]:
            _corrupt(vol_path)

        rows.append({
            "patient_id": f"lits-{i:04d}",
            "age": round(age, 1),
            "sex": sex,
            "institution": institution,
            "scanner": SCANNERS[rng.integers(0, len(SCANNERS))],
            "contrast": bool(rng.random() < 0.75),
            "volume_path": f"volumes/volume-{i}.nii.gz",
            "label_path": f"labels/segmentation-{i}.nii.gz",
        })

    pd.DataFrame(rows).to_csv(root / "clinical_metadata.csv", index=False)
    log.info("synth.lits.done", extra={"root": str(root), "n_files": written})
    return written


# --------------------------------------------------------------------------- #
# MIMIC-CXR                                                                    #
# --------------------------------------------------------------------------- #

_REPORT_TEMPLATES = [
    ("FINDINGS: The lungs are clear bilaterally. The cardiomediastinal silhouette "
     "is within normal limits. No pleural effusion or pneumothorax.\n"
     "IMPRESSION: No acute cardiopulmonary abnormality.", []),
    ("FINDINGS: Patchy airspace opacity in the right lower lobe measuring "
     "approximately 3.2 cm. No pleural effusion.\n"
     "IMPRESSION: Findings compatible with pneumonia. Recommend follow-up "
     "radiograph in 6 weeks.", ["pneumonia"]),
    ("FINDINGS: Enlargement of the cardiac silhouette. Mild pulmonary vascular "
     "congestion with interstitial edema.\n"
     "IMPRESSION: Cardiomegaly with mild pulmonary edema.", ["cardiomegaly", "edema"]),
    ("FINDINGS: Moderate left pleural effusion with associated basilar "
     "atelectasis. Heart size is normal.\n"
     "IMPRESSION: Moderate left pleural effusion. Consider thoracentesis.",
     ["pleural_effusion"]),
    ("FINDINGS: There is no evidence of consolidation. No focal opacity. "
     "Cardiac silhouette is enlarged, measuring 16.1 cm.\n"
     "IMPRESSION: Cardiomegaly without acute process.", ["cardiomegaly"]),
    ("FINDINGS: Bilateral perihilar opacities with Kerley B lines. Small "
     "bilateral pleural effusions.\n"
     "IMPRESSION: Pulmonary edema, likely cardiogenic. Effusions noted.",
     ["edema", "pleural_effusion"]),
]


def _chest_phantom(size: int, rng: np.random.Generator, opacity: bool) -> np.ndarray:
    """A crude frontal chest radiograph: dark lungs, bright mediastinum/ribs."""
    yy, xx = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size), indexing="ij")
    image = np.full((size, size), 1600.0)

    lungs = (((xx - 0.42) / 0.30) ** 2 + ((yy + 0.05) / 0.55) ** 2 < 1) | (
        ((xx + 0.42) / 0.30) ** 2 + ((yy + 0.05) / 0.55) ** 2 < 1
    )
    image[lungs] = 400.0
    image[np.abs(xx) < 0.12] = 2200.0                       # mediastinum
    image += 180 * np.sin(yy * 40)                          # ribs
    if opacity:
        blob = ((xx - 0.45) ** 2 + (yy + 0.35) ** 2) < 0.02
        image[blob] += 900.0
    image += rng.normal(0, 45, image.shape)
    return np.clip(image, 0, 4095).astype(np.uint16)


def generate_mimic(root: Path, n_studies: int, size: int, seed: int) -> int:
    """Emit a MIMIC-CXR-style tree of DICOM images plus paired free-text reports."""
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    files_dir = root / "files"
    reports_dir = root / "reports"
    files_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    rng = _rng(seed + 2)
    written = 0

    for i in range(n_studies):
        subject = f"p{10000000 + i}"
        study = f"s{50000000 + i}"
        template, labels = _REPORT_TEMPLATES[i % len(_REPORT_TEMPLATES)]

        subject_dir = files_dir / subject[:3] / subject / study
        subject_dir.mkdir(parents=True, exist_ok=True)

        pixels = _chest_phantom(size, rng, opacity=bool(labels))

        meta = Dataset()
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1.1"
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.ImplementationClassUID = generate_uid()

        ds = FileDataset(str(subject_dir / "image.dcm"), {},
                         file_meta=meta, preamble=b"\0" * 128)

        # De-identified in the MIMIC style: real dataset ships these blanked, and
        # the pipeline's PHI audit asserts they stay that way.
        ds.PatientName = ""
        ds.PatientID = subject
        ds.PatientBirthDate = ""
        ds.PatientSex = "M" if rng.random() < 0.55 else "F"
        ds.PatientAge = f"{int(np.clip(rng.normal(58, 16), 18, 95)):03d}Y"

        ds.StudyInstanceUID = generate_uid()
        ds.SeriesInstanceUID = generate_uid()
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.StudyID = study
        ds.StudyDate = (datetime(2150, 1, 1) + timedelta(days=int(rng.integers(0, 3000)))).strftime("%Y%m%d")
        ds.Modality = "DX"
        ds.ViewPosition = "PA" if rng.random() < 0.7 else "AP"
        ds.BodyPartExamined = "CHEST"
        ds.Manufacturer = str(rng.choice(["SIEMENS", "GE", "PHILIPS"]))
        ds.InstitutionName = str(rng.choice(INSTITUTIONS))

        ds.Rows, ds.Columns = pixels.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 12
        ds.HighBit = 11
        ds.PixelRepresentation = 0
        ds.PixelSpacing = [0.139, 0.139]
        ds.PixelData = pixels.tobytes()

        ds.save_as(subject_dir / "image.dcm", enforce_file_format=True)
        written += 1

        report_path = reports_dir / f"{study}.txt"
        report_path.write_text(
            f"                                 FINAL REPORT\n"
            f" EXAMINATION:  CHEST (PA AND LAT)\n\n"
            f" INDICATION:  ___ year old patient with shortness of breath.\n\n"
            f" COMPARISON:  None.\n\n {template}\n",
            encoding="utf-8",
        )
        written += 1

    log.info("synth.mimic.done", extra={"root": str(root), "n_files": written})
    return written


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", choices=["brats", "mimic", "lits", "all"], default="all")
    parser.add_argument("--n", type=int, default=12, help="subjects/studies per dataset")
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--size", type=int, default=48, help="volume edge length in voxels")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(level=args.log_level)
    shape = (args.size, args.size, args.size)
    total = 0

    if args.dataset in ("brats", "all"):
        total += generate_brats(args.out / "brats", args.n, shape, args.seed)
    if args.dataset in ("mimic", "all"):
        total += generate_mimic(args.out / "mimic", args.n, args.size * 4, args.seed)
    if args.dataset in ("lits", "all"):
        total += generate_lits(args.out / "lits", args.n, shape, args.seed)

    log.info("synth.complete", extra={"n_files": total, "out": str(args.out)})
    print(f"\nGenerated {total} files under {args.out.resolve()}")
    print("Planted defects (QC should find these):")
    for defect, idx in DEFECT_PLAN.items():
        print(f"  - {defect:<20} subject index {idx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
