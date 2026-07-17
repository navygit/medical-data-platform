"""Tests for the three case-study pipelines.

Covers the invariants that make the data trustworthy:
- no patient leaks across splits (BraTS, MIMIC)
- preprocessing preserves label semantics and fixes orientation
- report parsing handles negation and uncertainty
- cohort attrition is accounted for exactly
- bias auditing detects a known planted skew
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from common.config import Config
from common.metadata import ScanRecord
from pipelines.brats.ingest import ingest, parse_modality
from pipelines.brats.preprocess import normalise_intensity, preprocess_record, reorient, resample
from pipelines.brats.split import LeakageError, assign_splits, make_splits, verify_no_leakage
from pipelines.lits.bias_audit import age_band, audit_cohort, audit_representation
from pipelines.lits.cohort_builder import PRESETS, CohortSpec, Criterion, build_cohort
from pipelines.lits.quality_score import (
    WEIGHTS,
    score_integrity,
    score_slice_continuity,
    score_spacing,
    score_study,
)
from pipelines.mimic.dicom_parser import audit_phi, parse_age, parse_dicom
from pipelines.mimic.fusion_dataset import (
    binarise_labels,
    build_fusion_table,
    label_prevalence,
    split_by_subject,
)
from pipelines.mimic.report_parser import extract_labels, parse_report, split_sections

# --------------------------------------------------------------------------- #
# BraTS: ingest                                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("BraTS2021_00001_t1.nii.gz", "t1"),
        ("BraTS2021_00001_t1ce.nii.gz", "t1ce"),  # must not be shadowed by 't1'
        ("BraTS2021_00001_flair.nii.gz", "flair"),
        ("BraTS2021_00001_seg.nii.gz", "seg"),
        ("BraTS2021_00001_t2.nii", "t2"),
        ("readme.txt", None),
    ],
)
def test_parse_modality(filename: str, expected: str | None) -> None:
    assert parse_modality(Path(filename)) == expected


def test_ingest_records_unreadable_files_rather_than_raising(brats_cfg: Config) -> None:
    """A corrupt file must appear in the manifest with null stats, not vanish."""
    records = ingest(brats_cfg)
    assert records

    unreadable = [r for r in records if r.intensity_mean is None]
    assert unreadable, "the planted corrupt volume should produce a null-stat record"
    assert all(r.sha256 for r in records), "every record must be hashed"


def test_ingest_is_deterministic(brats_cfg: Config) -> None:
    a = [(r.patient_id, r.modality, r.sha256) for r in ingest(brats_cfg)]
    b = [(r.patient_id, r.modality, r.sha256) for r in ingest(brats_cfg)]
    assert a == b


# --------------------------------------------------------------------------- #
# BraTS: preprocess                                                            #
# --------------------------------------------------------------------------- #


def test_reorient_to_ras() -> None:
    import nibabel as nib

    affine = np.diag([-1.0, 1.0, 1.0, 1.0])  # LAS
    img = nib.Nifti1Image(np.random.rand(8, 8, 8).astype(np.float32), affine)
    assert "".join(nib.aff2axcodes(img.affine)) == "LAS"

    out = reorient(img, "RAS")
    assert "".join(nib.aff2axcodes(out.affine)) == "RAS"


def test_resample_nearest_preserves_label_values() -> None:
    """Linear interpolation of a label map invents classes. Nearest must not."""
    import nibabel as nib

    mask = np.zeros((8, 8, 8), dtype=np.uint8)
    mask[2:6, 2:6, 2:6] = 4
    img = nib.Nifti1Image(mask, np.diag([2.0, 2.0, 2.0, 1.0]))

    out = resample(img, (1.0, 1.0, 1.0), order=0)
    values = set(np.unique(np.asanyarray(out.dataobj)).tolist())

    assert values <= {0, 4}, f"nearest-neighbour invented labels: {values}"
    assert out.shape == (16, 16, 16)


def test_resample_updates_affine_scale() -> None:
    import nibabel as nib

    img = nib.Nifti1Image(np.random.rand(8, 8, 8).astype(np.float32), np.diag([2.0, 2.0, 2.0, 1.0]))
    out = resample(img, (1.0, 1.0, 1.0), order=1)
    assert np.allclose(np.abs(np.diag(out.affine)[:3]), [1.0, 1.0, 1.0])


def test_resample_is_a_noop_at_target_spacing() -> None:
    import nibabel as nib

    img = nib.Nifti1Image(np.random.rand(8, 8, 8).astype(np.float32), np.eye(4))
    assert resample(img, (1.0, 1.0, 1.0), order=1).shape == (8, 8, 8)


def test_zscore_uses_foreground_only() -> None:
    """Background dominates a brain volume; including it would crush contrast."""
    volume = np.zeros((10, 10, 10), dtype=np.float32)
    volume[5:, :, :] = 100.0  # half foreground

    out = normalise_intensity(volume, method="zscore")
    foreground = out[volume > 0]

    assert np.isfinite(out).all()
    assert abs(float(foreground.mean())) < 1e-3  # foreground centred on zero


def test_normalise_handles_constant_and_empty_volumes() -> None:
    assert np.isfinite(normalise_intensity(np.zeros((4, 4, 4), np.float32))).all()
    assert np.isfinite(normalise_intensity(np.full((4, 4, 4), 7.0, np.float32))).all()


def test_normalise_none_is_passthrough() -> None:
    volume = np.array([[[1.0, 2.0]]], dtype=np.float32)
    assert np.array_equal(normalise_intensity(volume, method="none"), volume)


def test_unknown_normalisation_raises() -> None:
    with pytest.raises(ValueError, match="unknown intensity normalisation"):
        normalise_intensity(np.ones((2, 2, 2), np.float32), method="bogus")


def test_preprocess_writes_processed_volume(brats_cfg: Config) -> None:
    records = [r for r in ingest(brats_cfg) if r.intensity_mean is not None]
    out = preprocess_record(records[0], brats_cfg)

    assert out is not None
    assert (Path(brats_cfg.paths.processed) / out.filepath).exists()
    assert out.orientation == "RAS"
    assert out.sha256 != records[0].sha256  # content changed, so the hash must too


# --------------------------------------------------------------------------- #
# BraTS: splitting                                                             #
# --------------------------------------------------------------------------- #


def _records(n_subjects: int) -> list[ScanRecord]:
    return [
        ScanRecord(
            patient_id=f"p{i:03d}",
            modality=m,
            filepath=f"p{i}_{m}.nii.gz",
            voxel_spacing=[1.0, 1.0, 1.0],
            tumor_volume_mm3=100.0 * (i + 1),
        )
        for i in range(n_subjects)
        for m in ("t1", "t1ce", "t2", "flair")
    ]


def test_split_never_leaks_a_patient(brats_cfg: Config) -> None:
    """The invariant that makes reported metrics meaningful."""
    splits = make_splits(_records(20), brats_cfg)

    everyone = [s for members in splits.values() for s in members]
    assert len(everyone) == len(set(everyone)) == 20


def test_split_is_deterministic_under_seed(brats_cfg: Config) -> None:
    records = _records(20)
    assert make_splits(records, brats_cfg) == make_splits(records, brats_cfg)

    brats_cfg.split.seed = 999
    assert make_splits(records, brats_cfg) != make_splits(
        records,
        brats_cfg.model_copy(update={"split": brats_cfg.split.model_copy(update={"seed": 42})}),
    )


def test_split_allocates_all_three_partitions(brats_cfg: Config) -> None:
    splits = make_splits(_records(20), brats_cfg)
    assert all(len(v) > 0 for v in splits.values())


def test_verify_no_leakage_raises_on_overlap() -> None:
    with pytest.raises(LeakageError, match="p1"):
        verify_no_leakage({"train": ["p1", "p2"], "val": ["p1"], "test": ["p3"]})


def test_verify_no_leakage_passes_when_disjoint() -> None:
    verify_no_leakage({"train": ["p1"], "val": ["p2"], "test": ["p3"]})


def test_assign_splits_stamps_records(brats_cfg: Config) -> None:
    records = _records(10)
    splits = make_splits(records, brats_cfg)
    stamped = assign_splits(records, splits)

    assert all(r.extra["split"] in ("train", "val", "test") for r in stamped)
    # Every record of one patient shares that patient's split.
    by_patient: dict[str, set[str]] = {}
    for r in stamped:
        by_patient.setdefault(r.patient_id, set()).add(r.extra["split"])
    assert all(len(v) == 1 for v in by_patient.values())


def test_split_handles_empty_input(brats_cfg: Config) -> None:
    assert make_splits([], brats_cfg) == {"train": [], "val": [], "test": []}


# --------------------------------------------------------------------------- #
# MIMIC: report parsing                                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text, finding, expected",
    [
        ("No evidence of pneumonia.", "pneumonia", 0),
        ("There is pneumonia in the RLL.", "pneumonia", 1),
        ("Pneumonia cannot be excluded.", "pneumonia", -1),
        ("Findings compatible with pneumonia.", "pneumonia", -1),
        ("Possible pneumonia.", "pneumonia", -1),
        # One negation scoping over two findings.
        ("No pleural effusion or pneumothorax.", "pleural_effusion", 0),
        ("No pleural effusion or pneumothorax.", "pneumothorax", 0),
        # Termination: the negation must not survive "but".
        ("No effusion, but consolidation is present.", "pneumonia", 1),
        ("No effusion, but consolidation is present.", "pleural_effusion", 0),
        # Sentence boundary must stop scope bleed.
        ("Heart size is normal. Small effusion.", "pleural_effusion", 1),
        # Not mentioned stays None, distinct from negated.
        ("The lungs are clear bilaterally.", "pneumothorax", None),
    ],
)
def test_negation_and_uncertainty(text: str, finding: str, expected: int | None) -> None:
    assert extract_labels(text)[0][finding] == expected


def test_positive_beats_negative_across_sections() -> None:
    """A patient described as having an effusion has one, whatever FINDINGS said."""
    report = parse_report(
        "FINDINGS: No pleural effusion.\nIMPRESSION: Small left pleural effusion."
    )
    assert report.labels["pleural_effusion"] == 1


def test_indication_section_excluded_from_labels() -> None:
    """'History of pneumonia' is the referral question, not a finding. Leak guard."""
    report = parse_report(
        "INDICATION: History of pneumonia.\n"
        "FINDINGS: The lungs are clear.\n"
        "IMPRESSION: No acute process."
    )
    assert report.labels["pneumonia"] != 1


def test_split_sections() -> None:
    sections = split_sections("EXAMINATION: CHEST\nFINDINGS: Clear.\nIMPRESSION: Normal.")
    assert set(sections) == {"examination", "findings", "impression"}
    assert sections["impression"] == "Normal."


def test_unstructured_report_still_parses() -> None:
    report = parse_report("Large pleural effusion noted.")
    assert "preamble" in report.sections
    assert report.labels["pleural_effusion"] == 1


def test_severity_and_measurements() -> None:
    report = parse_report("FINDINGS: Moderate effusion measuring 3.2 cm and a 5 mm nodule.")
    assert report.severity["pleural_effusion"] == 3  # 'moderate'
    assert 3.2 in report.measurements_cm
    assert 0.5 in report.measurements_cm  # 5 mm normalised to cm


def test_recommendations_extracted() -> None:
    report = parse_report("IMPRESSION: Effusion. Recommend thoracentesis for drainage.")
    assert any("thoracentesis" in r.lower() for r in report.recommendations)


# --------------------------------------------------------------------------- #
# MIMIC: DICOM + fusion                                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("058Y", 58.0),
        ("012M", 1.0),
        ("100Y", 100.0),
        ("", None),
        (None, None),
        ("garbage", None),
    ],
)
def test_parse_age(raw, expected) -> None:
    result = parse_age(raw)
    assert result is None if expected is None else result == pytest.approx(expected)


def test_synthetic_dicoms_are_deidentified(mimic_cfg: Config) -> None:
    from pipelines.mimic.dicom_parser import iter_dicoms

    raw = Path(mimic_cfg.paths.raw)
    records = [parse_dicom(p, raw) for p in iter_dicoms(raw / "files")]

    assert records and all(r is not None for r in records)
    assert all(not r.phi_flags for r in records), "PHI audit flagged a supposedly clean archive"


def test_phi_audit_detects_a_name() -> None:
    from pydicom.dataset import Dataset

    ds = Dataset()
    ds.PatientName = "DOE^JOHN"
    ds.ReferringPhysicianName = ""

    flags = audit_phi(ds)
    assert "PatientName" in flags
    assert "ReferringPhysicianName" not in flags  # empty is clean


def test_deidentify_blanks_identifiers() -> None:
    from pydicom.dataset import Dataset

    from pipelines.mimic.dicom_parser import deidentify

    ds = Dataset()
    ds.PatientName = "DOE^JOHN"
    ds.PatientBirthDate = "19800101"

    out = deidentify(ds)
    assert not str(out.PatientName)
    assert out.PatientIdentityRemoved == "YES"
    assert audit_phi(out) == []


def _fusion_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject_id": "p1",
                "study_id": "s1",
                "label_pneumonia": 1,
                "label_edema": 0,
                "label_cardiomegaly": None,
                "label_pleural_effusion": -1,
            },
            {
                "subject_id": "p1",
                "study_id": "s2",
                "label_pneumonia": 0,
                "label_edema": 1,
                "label_cardiomegaly": 1,
                "label_pleural_effusion": 0,
            },
            {
                "subject_id": "p2",
                "study_id": "s3",
                "label_pneumonia": -1,
                "label_edema": 0,
                "label_cardiomegaly": 0,
                "label_pleural_effusion": 1,
            },
            {
                "subject_id": "p3",
                "study_id": "s4",
                "label_pneumonia": 1,
                "label_edema": 1,
                "label_cardiomegaly": 0,
                "label_pleural_effusion": 0,
            },
        ]
    )


def test_binarise_uncertain_policies() -> None:
    frame = _fusion_frame()

    zeros, _ = binarise_labels(frame, uncertain="zeros")
    assert zeros[2, 0] == 0.0  # -1 -> 0

    ones, _ = binarise_labels(frame, uncertain="ones")
    assert ones[2, 0] == 1.0  # -1 -> 1

    ignore, _ = binarise_labels(frame, uncertain="ignore")
    assert np.isnan(ignore[2, 0])  # -1 -> NaN for a masked loss

    # 'Not mentioned' is always negative, under every policy.
    assert zeros[0, 2] == 0.0 and ones[0, 2] == 0.0 and ignore[0, 2] == 0.0


def test_label_prevalence_separates_uncertain_from_absent() -> None:
    row = label_prevalence(_fusion_frame()).set_index("finding").loc["cardiomegaly"]
    assert row["positive"] == 1
    assert row["not_mentioned"] == 1
    assert row["negative"] == 2


def test_fusion_join_reports_orphans() -> None:
    from pipelines.mimic.report_parser import parse_report

    images = [
        {"study_id": "s1", "subject_id": "p1", "filepath": "a.dcm"},
        {"study_id": "s2", "subject_id": "p1", "filepath": "b.dcm"},  # no report
    ]
    reports = {
        "s1": parse_report("FINDINGS: Clear.", "s1"),
        "s3": parse_report("FINDINGS: Clear.", "s3"),  # no image
    }

    table, stats = build_fusion_table(images, reports)

    assert stats.n_joined == 1
    assert stats.n_images_without_report == 1
    assert stats.n_reports_without_image == 1
    assert stats.join_rate == pytest.approx(0.5)
    assert len(table) == 1


def test_fusion_split_groups_by_subject() -> None:
    """p1 has two studies; both must land in the same split."""
    out = split_by_subject(_fusion_frame(), seed=1)
    assert out.groupby("subject_id")["split"].nunique().max() == 1


# --------------------------------------------------------------------------- #
# LiTS: quality scoring                                                        #
# --------------------------------------------------------------------------- #


def test_weights_sum_to_one() -> None:
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


def test_integrity_scoring() -> None:
    assert score_integrity(np.random.rand(4, 4, 4))[0] == 1.0
    assert score_integrity(None)[0] == 0.0
    assert score_integrity(np.full((4, 4, 4), 5.0))[0] == 0.0  # constant
    assert score_integrity(np.array([[[np.nan, 1.0]]]))[0] == 0.0  # non-finite


def test_spacing_scoring_penalises_anisotropy() -> None:
    isotropic, _ = score_spacing([1.0, 1.0, 1.0])
    anisotropic, reason = score_spacing([5.0, 0.8, 0.8])

    assert isotropic > anisotropic
    assert isotropic == pytest.approx(1.0)
    assert reason and "anisotropic" in reason
    assert score_spacing(None)[0] == 0.0
    assert score_spacing([0.0, 1.0, 1.0])[0] == 0.0  # invalid


def test_slice_continuity_detects_duplicates() -> None:
    volume = np.random.rand(10, 8, 8).astype(np.float32)
    clean, _ = score_slice_continuity(volume)

    volume[5] = volume[4]  # duplicate a slice
    duplicated, reason = score_slice_continuity(volume)

    assert duplicated < clean
    assert reason and "duplicate" in reason


def test_score_study_is_bounded_and_graded() -> None:
    row = {
        "patient_id": "p1",
        "age": 60,
        "sex": "M",
        "institution": "A",
        "scanner": "S",
        "voxel_spacing": [1.0, 1.0, 1.0],
        "shape": [16, 16, 16],
    }
    good = score_study(row, np.random.rand(16, 16, 16).astype(np.float32) * 100)

    assert 0 <= good.overall <= 100
    assert good.grade in "ABCDF"
    assert set(good.components) == set(WEIGHTS)

    broken = score_study({"patient_id": "p2"}, None)
    assert broken.overall < good.overall
    assert broken.grade == "F"
    assert broken.reasons


# --------------------------------------------------------------------------- #
# LiTS: cohort + bias                                                          #
# --------------------------------------------------------------------------- #


def _cohort_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "patient_id": f"p{i}",
                "age": 20 + i * 6,
                "sex": "M" if i % 4 else "F",
                "contrast": i % 3 != 0,
                "has_label": True,
                "quality_score": 50 + i * 4,
                "institution": "Site-A" if i < 8 else "Site-B",
                "scanner": "GE",
            }
            for i in range(12)
        ]
    )


def test_attrition_accounts_for_every_study() -> None:
    frame = _cohort_frame()
    cohort, attrition = build_cohort(frame, PRESETS["adult_contrast_liver_ct"])

    assert len(cohort) <= len(frame)
    assert attrition.iloc[0]["n_after"] == len(frame)
    assert attrition.iloc[-1]["n_after"] == len(cohort)
    # Each step's arithmetic must close.
    for _, row in attrition.iterrows():
        assert row["n_before"] - row["n_removed"] == row["n_after"]


def test_cohort_criteria_are_applied() -> None:
    spec = CohortSpec(name="adults_only", criteria=[Criterion("adult", "age >= 40")])
    cohort, _ = build_cohort(_cohort_frame(), spec)
    assert (cohort["age"] >= 40).all()


def test_bad_criterion_raises_rather_than_being_skipped() -> None:
    """A silently ignored inclusion rule yields a cohort that lies about itself."""
    spec = CohortSpec(name="broken", criteria=[Criterion("typo", "nonexistent_column > 1")])
    with pytest.raises(ValueError, match="typo"):
        build_cohort(_cohort_frame(), spec)


@pytest.mark.parametrize(
    "age, expected",
    [
        (25, "<40"),
        (45, "40-59"),
        (70, "60-79"),
        (85, "80+"),
        (None, "unknown"),
    ],
)
def test_age_band(age, expected) -> None:
    assert age_band(age) == expected


def test_bias_audit_flags_a_dominant_subgroup() -> None:
    skewed = pd.DataFrame({"sex": ["M"] * 19 + ["F"], "age": [60] * 20})
    finding = audit_representation(skewed, "sex")

    assert finding.severity == "WARN"
    assert "dominates" in finding.message


def test_bias_audit_accepts_a_balanced_cohort() -> None:
    balanced = pd.DataFrame({"sex": ["M"] * 10 + ["F"] * 10})
    assert audit_representation(balanced, "sex").severity == "INFO"


def test_audit_cohort_returns_findings() -> None:
    findings = audit_cohort(_cohort_frame(), outcome_columns=("quality_score",))
    assert findings
    assert all(f.severity in ("INFO", "WARN") for f in findings)


def test_audit_handles_empty_frame() -> None:
    finding = audit_representation(pd.DataFrame({"sex": []}), "sex")
    assert finding.severity == "INFO"
