"""Tests for the QC engine.

The core assertion of this suite: each defect planted by the synthetic generator
is caught by the check that is supposed to catch it. That is a *behavioural*
contract, not an implementation detail -- it is the reason to trust the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from common.config import Config
from common.metadata import ScanRecord
from common.qc import (
    QCFinding,
    apply_findings,
    check_duplicates,
    check_missing_modalities,
    findings_to_frame,
    registered_checks,
    resolve_path,
    run_checks,
    summarise,
    write_html_report,
)
from pipelines.brats.ingest import ingest
from scripts.generate_synthetic_data import DEFECT_PLAN


def _errors_for(findings: list[QCFinding], check: str) -> set[str]:
    """Subjects with an ERROR from a given check."""
    return {f.patient_id for f in findings if f.check == check and f.severity == "ERROR"}


def _subject(index: int) -> str:
    return f"BraTS2021_{index:05d}"


# --------------------------------------------------------------------------- #
# Planted-defect detection (the contract)                                      #
# --------------------------------------------------------------------------- #


@pytest.fixture
def brats_findings(brats_cfg: Config) -> tuple[list[ScanRecord], list[QCFinding]]:
    records = ingest(brats_cfg)
    findings = run_checks(records, brats_cfg)
    findings += check_duplicates(records)
    findings += check_missing_modalities(records, brats_cfg)
    return records, findings


def test_detects_corrupt_volume(brats_findings) -> None:
    _, findings = brats_findings
    assert _subject(DEFECT_PLAN["corrupt"]) in _errors_for(findings, "corrupt_volume")


def test_detects_empty_mask(brats_findings) -> None:
    _, findings = brats_findings
    assert _subject(DEFECT_PLAN["empty_mask"]) in _errors_for(findings, "empty_mask")


def test_detects_constant_volume(brats_findings) -> None:
    _, findings = brats_findings
    assert _subject(DEFECT_PLAN["constant_volume"]) in _errors_for(findings, "empty_volume")


def test_detects_missing_modality(brats_findings) -> None:
    _, findings = brats_findings
    assert _subject(DEFECT_PLAN["missing_modality"]) in _errors_for(findings, "missing_modalities")


def test_detects_wrong_orientation_as_warning(brats_findings) -> None:
    """Orientation is a WARN: preprocess fixes it, so it must not drop the subject."""
    _, findings = brats_findings
    subject = _subject(DEFECT_PLAN["wrong_orientation"])
    hits = [f for f in findings if f.check == "orientation" and f.patient_id == subject]
    assert hits and all(f.severity == "WARN" for f in hits)
    assert subject not in _errors_for(findings, "orientation")


def test_detects_coarse_spacing_as_warning(brats_findings) -> None:
    _, findings = brats_findings
    subject = _subject(DEFECT_PLAN["coarse_spacing"])
    assert any(
        f.check == "spacing" and f.patient_id == subject and f.severity == "WARN" for f in findings
    )


def test_duplicate_policy_keeps_canonical_rejects_copy(brats_findings) -> None:
    """A duplicated subject must cost the copy, not the original."""
    _, findings = brats_findings
    rejected = _errors_for(findings, "duplicate_content")

    assert rejected, "expected the planted duplicate subject to be rejected"
    assert _subject(1) not in rejected, "the canonical subject must be retained"


def test_clean_subjects_survive_qc(brats_findings) -> None:
    """QC must not be a blunt instrument -- clean subjects have to pass."""
    records, findings = brats_findings
    updated = apply_findings(records, findings)
    passing = {r.patient_id for r in updated if r.qc_status != "fail"}

    defective = {_subject(i) for i in DEFECT_PLAN.values()}
    assert passing - defective, "no clean subject survived QC"
    assert _subject(2) in passing


# --------------------------------------------------------------------------- #
# Check semantics                                                              #
# --------------------------------------------------------------------------- #


def test_good_record_produces_no_errors(
    brats_cfg: Config, sample_record: ScanRecord, tmp_path: Path
) -> None:
    """A well-formed record on disk must be clean."""
    # Point raw at a private dir: the synth corpus is session-scoped and shared,
    # so writing into it would leak fixture state into other tests.
    brats_cfg.paths.raw = tmp_path / "raw"
    target = brats_cfg.paths.raw / sample_record.filepath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"not empty")

    findings = run_checks(
        [sample_record],
        brats_cfg,
        only=[
            "file_readable",
            "corrupt_volume",
            "empty_volume",
            "dimensions",
            "orientation",
            "spacing",
        ],
    )
    assert [f for f in findings if f.severity == "ERROR"] == []


def test_missing_file_is_an_error(brats_cfg: Config, sample_record: ScanRecord) -> None:
    ghost = sample_record.model_copy(update={"filepath": "does/not/exist.nii.gz"})
    findings = run_checks([ghost], brats_cfg, only=["file_readable"])
    assert len(findings) == 1 and findings[0].severity == "ERROR"


def test_resolve_path_prefers_raw_then_falls_back_to_processed(
    brats_cfg: Config, sample_record: ScanRecord, tmp_path: Path
) -> None:
    """Regression: relative manifest paths must resolve against the data roots.

    Raw wins when the file is in both, since QC normally gates the landing zone;
    processed is the fallback for QC runs over already-normalised records.
    """
    brats_cfg.paths.raw = tmp_path / "raw"
    brats_cfg.paths.processed = tmp_path / "processed"

    in_processed = brats_cfg.paths.processed / sample_record.filepath
    in_processed.parent.mkdir(parents=True, exist_ok=True)
    in_processed.write_bytes(b"x")
    assert resolve_path(sample_record, brats_cfg) == in_processed

    in_raw = brats_cfg.paths.raw / sample_record.filepath
    in_raw.parent.mkdir(parents=True, exist_ok=True)
    in_raw.write_bytes(b"x")
    assert resolve_path(sample_record, brats_cfg) == in_raw


def test_a_raising_check_becomes_a_finding_not_a_crash(
    brats_cfg: Config, sample_record: ScanRecord
) -> None:
    """One malformed study must not abort QC of the rest."""
    from common.qc import _REGISTRY

    def _boom(rec, cfg):
        raise RuntimeError("simulated check failure")

    _REGISTRY["_boom"] = _boom
    try:
        findings = run_checks([sample_record], brats_cfg, only=["_boom"])
        assert len(findings) == 1
        assert findings[0].severity == "ERROR"
        assert "RuntimeError" in findings[0].message
    finally:
        del _REGISTRY["_boom"]


def test_empty_mask_threshold_is_configurable(brats_cfg: Config, sample_record: ScanRecord) -> None:
    tiny = sample_record.model_copy(update={"tumor_volume_mm3": 5.0})  # 5 voxels at 1mm

    brats_cfg.qc.min_tumor_voxels = 3
    assert run_checks([tiny], brats_cfg, only=["empty_mask"]) == []

    brats_cfg.qc.min_tumor_voxels = 50
    assert len(run_checks([tiny], brats_cfg, only=["empty_mask"])) == 1


def test_missing_modalities_respects_config(brats_cfg: Config) -> None:
    records = [
        ScanRecord(patient_id="p1", modality=m, filepath=f"p1_{m}.nii.gz") for m in ("t1", "t2")
    ]
    brats_cfg.qc.expected_modalities = ["t1", "t1ce", "t2", "flair"]
    findings = check_missing_modalities(records, brats_cfg)

    assert len(findings) == 1
    assert "t1ce" in findings[0].message and "flair" in findings[0].message

    brats_cfg.qc.expected_modalities = []
    assert check_missing_modalities(records, brats_cfg) == []


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #


def test_summarise_counts_and_pass_rate(sample_record: ScanRecord) -> None:
    records = [sample_record, sample_record.model_copy(update={"patient_id": "subj-002"})]
    findings = [QCFinding("subj-001", "t1", "spacing", "ERROR", "bad")]

    report = summarise(findings, records)

    assert report.n_subjects == 2
    assert report.n_errors == 1
    assert report.failed_subjects == ["subj-001"]
    assert report.pass_rate == pytest.approx(0.5)


def test_apply_findings_sets_status() -> None:
    records = [
        ScanRecord(patient_id="p1", modality="t1", filepath="a"),
        ScanRecord(patient_id="p2", modality="t1", filepath="b"),
        ScanRecord(patient_id="p3", modality="t1", filepath="c"),
    ]
    findings = [
        QCFinding("p1", "t1", "spacing", "ERROR", "bad"),
        QCFinding("p2", "t1", "orientation", "WARN", "meh"),
    ]
    status = {r.patient_id: r.qc_status for r in apply_findings(records, findings)}

    assert status == {"p1": "fail", "p2": "warn", "p3": "pass"}


def test_html_report_is_self_contained(tmp_path: Path, sample_record: ScanRecord) -> None:
    """No external requests: the report must open on an air-gapped workstation."""
    findings = [QCFinding("subj-001", "t1", "spacing", "ERROR", "coarse <script>alert(1)</script>")]
    report = summarise(findings, [sample_record])
    path = write_html_report(report, findings, tmp_path / "r.html", name="test")

    html = path.read_text(encoding="utf-8")
    assert "<table>" in html
    assert "http://" not in html and "cdn" not in html.lower()
    # Findings text is escaped, not injected.
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_empty_report_renders_pass_state(tmp_path: Path, sample_record: ScanRecord) -> None:
    report = summarise([], [sample_record])
    html = write_html_report(report, [], tmp_path / "r.html").read_text(encoding="utf-8")
    assert "No findings" in html


def test_findings_frame_has_stable_columns() -> None:
    frame = findings_to_frame([])
    assert list(frame.columns) == [
        "patient_id",
        "modality",
        "check",
        "severity",
        "message",
        "filepath",
    ]


def test_registry_is_populated() -> None:
    checks = registered_checks()
    assert "file_readable" in checks and "empty_mask" in checks
    assert checks == sorted(checks)
