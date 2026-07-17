"""End-to-end smoke tests for all three pipelines.

These run the real orchestrators over the synthetic corpus and assert on the
artifacts that actually land on disk. They are the tests that would catch a
regression a unit test cannot: a stage wired to the wrong config key, a manifest
written to the wrong path, a release that silently contains zero files.

Marked ``slow`` -- deselect with ``pytest -m "not slow"`` during tight loops.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.config import Config
from common.versioning import read_release, verify_release

pytestmark = pytest.mark.slow


# --------------------------------------------------------------------------- #
# BraTS                                                                        #
# --------------------------------------------------------------------------- #


def test_brats_pipeline_end_to_end(brats_cfg: Config) -> None:
    from pipelines.brats.run import run

    result = run(brats_cfg)
    outputs = Path(brats_cfg.paths.outputs)

    # Every artifact the README promises must exist.
    assert (outputs / "QC_REPORT.html").exists()
    assert (outputs / "QC_REPORT.csv").exists()
    assert (outputs / "metadata" / "manifest_raw.parquet").exists()
    assert (outputs / "metadata" / "manifest_processed.parquet").exists()
    assert (outputs / "metadata" / "splits.csv").exists()
    assert list((outputs / "figures").glob("*.png")), "no figures exported"

    assert result["n_ingested"] > 0
    assert result["n_processed"] > 0
    assert result["qc"]["n_errors"] > 0, "QC found nothing in a corpus with planted defects"
    assert 0 < result["qc"]["pass_rate"] < 1, "QC should reject some but not all subjects"


def test_brats_release_is_verifiable(brats_cfg: Config) -> None:
    """A release must re-verify against the bytes on disk."""
    from pipelines.brats.run import run

    run(brats_cfg)
    manifest = Path(brats_cfg.paths.releases) / "v1.0.0" / "manifest.json"
    assert manifest.exists()

    release = read_release(manifest)
    assert release.n_files > 0
    assert release.dataset_hash
    assert release.config_snapshot["name"] == "brats"
    assert [n.stage for n in release.lineage] == ["ingest", "qc", "preprocess", "split"]

    assert verify_release(release, Path(brats_cfg.paths.processed)) == []


def test_brats_release_detects_corruption_after_the_fact(brats_cfg: Config) -> None:
    """Mutating a released file must be detectable. This is the point of hashing."""
    from pipelines.brats.run import run

    run(brats_cfg)
    release = read_release(Path(brats_cfg.paths.releases) / "v1.0.0" / "manifest.json")

    victim = Path(brats_cfg.paths.processed) / release.files[0].path
    victim.write_bytes(b"corrupted after release")

    problems = verify_release(release, Path(brats_cfg.paths.processed))
    assert len(problems) == 1 and "hash mismatch" in problems[0]


def test_brats_defective_subjects_are_excluded_from_release(brats_cfg: Config) -> None:
    """The planted defects must not reach the released dataset."""
    from pipelines.brats.run import run
    from scripts.generate_synthetic_data import DEFECT_PLAN

    run(brats_cfg)
    release = read_release(Path(brats_cfg.paths.releases) / "v1.0.0" / "manifest.json")
    released = {f.patient_id for f in release.files}

    for defect in ("corrupt", "empty_mask", "missing_modality", "constant_volume"):
        subject = f"BraTS2021_{DEFECT_PLAN[defect]:05d}"
        assert subject not in released, f"{defect} subject {subject} reached the release"

    assert released, "release is empty"


def test_brats_stage_limiting(brats_cfg: Config) -> None:
    from pipelines.brats.run import run

    result = run(brats_cfg, stage="ingest")
    assert "n_ingested" in result
    assert "n_processed" not in result
    assert not (Path(brats_cfg.paths.outputs) / "QC_REPORT.html").exists()


def test_brats_rerun_does_not_mutate_release(brats_cfg: Config) -> None:
    """Re-running is safe and must not overwrite a published release."""
    from pipelines.brats.run import run

    run(brats_cfg)
    first = (Path(brats_cfg.paths.releases) / "v1.0.0" / "manifest.json").read_text()

    result = run(brats_cfg)
    second = (Path(brats_cfg.paths.releases) / "v1.0.0" / "manifest.json").read_text()

    assert first == second
    assert "skipped" in str(result["release"])


def test_brats_missing_raw_data_raises_clearly(brats_cfg: Config, tmp_path: Path) -> None:
    from pipelines.brats.run import run

    brats_cfg.paths.raw = tmp_path / "nonexistent"
    with pytest.raises(FileNotFoundError, match="generate_synthetic_data"):
        run(brats_cfg)


# --------------------------------------------------------------------------- #
# MIMIC                                                                        #
# --------------------------------------------------------------------------- #


def test_mimic_pipeline_end_to_end(mimic_cfg: Config) -> None:
    from pipelines.mimic.run import run

    result = run(mimic_cfg)
    outputs = Path(mimic_cfg.paths.outputs)
    processed = Path(mimic_cfg.paths.processed)

    assert (outputs / "PHI_AUDIT.csv").exists()
    assert (outputs / "label_prevalence.csv").exists()
    assert (processed / "fusion_dataset.parquet").exists()
    assert (processed / "structured_reports.csv").exists()
    assert (processed / "patient_timeline.csv").exists()

    assert result["phi_clean"] is True
    assert result["join_rate"] == 1.0, "synthetic images and reports must pair 1:1"
    assert result["n_joined"] > 0


def test_mimic_fusion_dataset_has_labels(mimic_cfg: Config) -> None:
    import pandas as pd

    from pipelines.mimic.run import run

    run(mimic_cfg)
    table = pd.read_parquet(Path(mimic_cfg.paths.processed) / "fusion_dataset.parquet")

    assert "label_pneumonia" in table.columns
    assert "report_text" in table.columns
    assert "split" in table.columns
    # The generator plants positive findings; the parser must recover some.
    assert (table["label_pneumonia"] == 1).any() or (table["label_cardiomegaly"] == 1).any()


def test_mimic_split_groups_by_subject(mimic_cfg: Config) -> None:
    import pandas as pd

    from pipelines.mimic.run import run

    run(mimic_cfg)
    table = pd.read_parquet(Path(mimic_cfg.paths.processed) / "fusion_dataset.parquet")
    assert table.groupby("subject_id")["split"].nunique().max() == 1


# --------------------------------------------------------------------------- #
# LiTS                                                                         #
# --------------------------------------------------------------------------- #


def test_lits_pipeline_end_to_end(lits_cfg: Config) -> None:
    from pipelines.lits.run import run

    result = run(lits_cfg)
    outputs = Path(lits_cfg.paths.outputs)

    assert (outputs / "dataset_card.md").exists()
    assert (outputs / "quality_scores.csv").exists()
    assert (outputs / "cohort_attrition.csv").exists()
    assert (outputs / "bias_audit.csv").exists()
    assert (outputs / "cohort_shift.csv").exists()
    assert list((outputs / "figures").glob("*.png"))

    assert result["n_studies_cohort"] > 0
    assert result["n_studies_cohort"] <= result["n_studies_total"]
    assert 0 <= result["mean_quality"] <= 100


def test_lits_dataset_card_contains_governance_sections(lits_cfg: Config) -> None:
    """The card must carry the sections a reviewer needs -- especially the limits."""
    from pipelines.lits.run import run

    run(lits_cfg)
    card = (Path(lits_cfg.paths.outputs) / "dataset_card.md").read_text(encoding="utf-8")

    for section in (
        "## Purpose",
        "## Provenance",
        "## Cohort definition",
        "### Attrition",
        "## Quality control",
        "## Bias and limitations",
        "## Recommended uses",
        "## Uses that are NOT recommended",
        "## Ethical considerations",
        "## Maintenance",
    ):
        assert section in card, f"dataset card is missing {section!r}"

    assert "Clinical deployment" in card, "card must warn against clinical use"


def test_lits_bias_audit_detects_planted_skew(lits_cfg: Config) -> None:
    """The generator skews sex and site on purpose; the audit must notice."""
    import pandas as pd

    from pipelines.lits.run import run

    run(lits_cfg)
    bias = pd.read_csv(Path(lits_cfg.paths.outputs) / "bias_audit.csv")

    assert not bias.empty
    assert (bias["severity"] == "WARN").any(), "audit found no skew in a deliberately skewed cohort"


def test_lits_cohort_preset_is_configurable(lits_cfg: Config) -> None:
    """The permissive preset must retain at least as much as the strict one."""
    from pipelines.lits.run import run

    strict = run(lits_cfg)

    lits_cfg.extra["cohort"] = "exploratory_all_liver_ct"
    lits_cfg.extra["dataset_version"] = "v2.0.0"
    permissive = run(lits_cfg)

    assert permissive["n_studies_cohort"] >= strict["n_studies_cohort"]


def test_lits_unknown_cohort_raises(lits_cfg: Config) -> None:
    from pipelines.lits.run import run

    lits_cfg.extra["cohort"] = "does_not_exist"
    with pytest.raises(ValueError, match="unknown cohort"):
        run(lits_cfg)


# --------------------------------------------------------------------------- #
# Cross-pipeline                                                               #
# --------------------------------------------------------------------------- #


def test_all_releases_share_one_manifest_schema(
    brats_cfg: Config, mimic_cfg: Config, lits_cfg: Config
) -> None:
    """One release format across all three datasets: learn the schema once."""
    from pipelines.brats.run import run as run_brats
    from pipelines.lits.run import run as run_lits
    from pipelines.mimic.run import run as run_mimic

    run_brats(brats_cfg)
    run_mimic(mimic_cfg)
    run_lits(lits_cfg)

    required = {
        "dataset",
        "version",
        "schema_version",
        "created_at",
        "dataset_hash",
        "n_files",
        "n_subjects",
        "files",
        "splits",
        "config_snapshot",
        "lineage",
        "environment",
    }
    for cfg in (brats_cfg, mimic_cfg, lits_cfg):
        manifest = Path(cfg.paths.releases) / "v1.0.0" / "manifest.json"
        assert manifest.exists(), f"{cfg.name} produced no release"

        data = json.loads(manifest.read_text(encoding="utf-8"))
        assert required <= set(data), f"{cfg.name} manifest is missing {required - set(data)}"
        assert data["schema_version"] == "1.0"
        assert data["n_files"] > 0
