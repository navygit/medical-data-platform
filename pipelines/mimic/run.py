"""MIMIC-CXR multimodal pipeline entrypoint.

Runs::

    scan DICOM headers -> PHI audit -> parse reports -> join -> label -> timeline
    -> split -> figures -> release

Deliberately stops short of training a model by default. The data product *is*
the deliverable: a joined, labelled, split, audited table plus a timeline. Model
training lives in ``train_multimodal.py`` behind the optional ``[ml]`` extra, so
this pipeline runs anywhere.

Usage:
    python -m pipelines.mimic.run --config configs/mimic.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from common.config import Config, load_config
from common.logging import configure_logging, get_logger
from common.versioning import Release, create_release, node, write_release
from common.visualization import plot_categorical, plot_distribution
from pipelines.mimic.dicom_parser import DicomRecord, iter_dicoms, parse_dicom
from pipelines.mimic.fusion_dataset import (
    TARGET_FINDINGS,
    build_fusion_table,
    build_patient_timeline,
    label_prevalence,
    split_by_subject,
    write_fusion_dataset,
)
from pipelines.mimic.report_parser import parse_report_file

log = get_logger(__name__)


def stage_scan_images(cfg: Config) -> list[DicomRecord]:
    """Read every DICOM header under ``paths.raw/files``."""
    raw = Path(cfg.paths.raw)
    files_root = raw / "files"
    if not files_root.exists():
        raise FileNotFoundError(
            f"MIMIC image root not found: {files_root}\n"
            "Run: python scripts/generate_synthetic_data.py --dataset mimic"
        )

    records = [r for path in iter_dicoms(files_root) if (r := parse_dicom(path, raw))]
    log.info("mimic.images_scanned", extra={"n_images": len(records)})
    return records


def stage_phi_audit(cfg: Config, records: list[DicomRecord]) -> dict:
    """Verify the archive is de-identified; write the audit report.

    A non-empty result is a compliance incident, not a data-quality nit, so it is
    logged at ERROR and surfaced in the release manifest.
    """
    flagged = [r for r in records if r.phi_flags]
    rows = [
        {"filepath": r.filepath, "study_id": r.study_id, "phi_tags": ",".join(r.phi_flags or [])}
        for r in flagged
    ]

    out = Path(cfg.paths.outputs)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["filepath", "study_id", "phi_tags"]).to_csv(
        out / "PHI_AUDIT.csv", index=False
    )

    summary = {
        "n_scanned": len(records),
        "n_flagged": len(flagged),
        "clean": not flagged,
        "tags_seen": sorted({t for r in flagged for t in (r.phi_flags or [])}),
    }
    if flagged:
        log.error("phi.violations_found", extra=summary)
    else:
        log.info("phi.audit_clean", extra=summary)
    return summary


def stage_parse_reports(cfg: Config) -> dict:
    """Parse every report under ``paths.raw/reports``."""
    reports_root = Path(cfg.paths.raw) / "reports"
    if not reports_root.exists():
        raise FileNotFoundError(f"MIMIC report root not found: {reports_root}")

    reports = {path.stem: parse_report_file(path) for path in sorted(reports_root.glob("*.txt"))}
    n_positive = sum(1 for r in reports.values() if r.positive_findings)
    log.info(
        "mimic.reports_parsed",
        extra={
            "n_reports": len(reports),
            "n_with_positive_finding": n_positive,
        },
    )
    return reports


def stage_filter_views(cfg: Config, records: list[DicomRecord]) -> list[DicomRecord]:
    """Restrict the cohort to the configured view positions."""
    allowed = {v.upper() for v in cfg.extra.get("allowed_views", ["PA", "AP"])}
    if not allowed:
        return records

    kept = [r for r in records if (r.view_position or "").upper() in allowed]
    dropped = len(records) - len(kept)
    if dropped:
        log.info(
            "mimic.views_filtered",
            extra={
                "allowed": sorted(allowed),
                "n_kept": len(kept),
                "n_dropped": dropped,
            },
        )
    return kept


def stage_figures(cfg: Config, table: pd.DataFrame, prevalence: pd.DataFrame) -> int:
    """Export cohort composition and label-prevalence figures."""
    figures = Path(cfg.paths.outputs) / "figures"
    n = 0

    plot_categorical(
        {r["finding"]: int(r["positive"]) for _, r in prevalence.iterrows()},
        figures / "label_prevalence.png",
        "Positive label counts",
        "finding",
    )
    n += 1

    for column, title in (
        ("view_position", "Studies per view"),
        ("sex", "Sex distribution"),
        ("manufacturer", "Studies per manufacturer"),
    ):
        if column in table and table[column].notna().any():
            plot_categorical(
                table[column].value_counts().to_dict(),
                figures / f"{column}_counts.png",
                title,
                column,
            )
            n += 1

    if "age" in table and table["age"].notna().any():
        plot_distribution(
            table["age"].dropna().tolist(),
            figures / "age_distribution.png",
            "Age distribution",
            "age (years)",
        )
        n += 1

    log.info("mimic.figures", extra={"n_figures": n})
    return n


def _release_from_table(cfg: Config, table: pd.DataFrame, qc_summary: dict) -> Release:
    """Adapt the fusion table to the shared release schema.

    The platform's release model is series-oriented; a MIMIC row is an
    image+report pair. Mapping it onto :class:`~common.metadata.ScanRecord` keeps
    one release format across all three case studies, so a consumer learns the
    manifest schema once.
    """
    from common.metadata import ScanRecord

    records = [
        ScanRecord(
            patient_id=str(row["subject_id"]),
            study_uid=str(row["study_id"]),
            series_uid=str(row.get("series_uid") or row["study_id"]),
            modality=str(row.get("modality") or "dx"),
            filepath=str(row["filepath"]),
            shape=(
                [int(row["rows"]), int(row["columns"])]
                if pd.notna(row.get("rows")) and pd.notna(row.get("columns"))
                else []
            ),
            sha256=str(row["sha256"]) if pd.notna(row.get("sha256")) else None,
            file_size_bytes=(
                int(row["file_size_bytes"]) if pd.notna(row.get("file_size_bytes")) else None
            ),
            institution=str(row["institution"]) if pd.notna(row.get("institution")) else None,
            scanner=str(row["manufacturer"]) if pd.notna(row.get("manufacturer")) else None,
            age=float(row["age"]) if pd.notna(row.get("age")) else None,
            sex=str(row["sex"]) if pd.notna(row.get("sex")) else None,
            mask_available=False,
            label_classes=[i for i, f in enumerate(TARGET_FINDINGS) if row.get(f"label_{f}") == 1],
            qc_status="pass",
            extra={"split": str(row.get("split", "unassigned"))},
        )
        for _, row in table.iterrows()
    ]

    splits: dict[str, list[str]] = {}
    if "split" in table:
        for name, group in table.groupby("split"):
            splits[str(name)] = sorted(group["subject_id"].astype(str).unique())

    return create_release(
        dataset="mimic",
        version=str(cfg.extra.get("dataset_version", "v1.0.0")),
        records=records,
        cfg=cfg,
        splits=splits,
        qc_summary=qc_summary,
        lineage=[
            node(
                "scan_images",
                "Read DICOM headers without decoding pixels.",
                root=str(cfg.paths.raw),
            ),
            node(
                "phi_audit",
                "Re-verified de-identification across PHI tag set.",
                result=qc_summary.get("phi", {}),
            ),
            node(
                "parse_reports",
                "Rule-based section/negation/severity extraction.",
                findings=list(TARGET_FINDINGS),
            ),
            node(
                "fusion",
                "Joined images to reports on study_id.",
                join_rate=qc_summary.get("join_rate"),
            ),
            node("split", "Subject-grouped partition.", **cfg.split.model_dump(mode="json")),
        ],
    )


def run(cfg: Config) -> dict:
    """Execute the MIMIC pipeline and return summary counts."""
    cfg.paths.mkdirs()

    images = stage_filter_views(cfg, stage_scan_images(cfg))
    phi = stage_phi_audit(cfg, images)
    reports = stage_parse_reports(cfg)

    table, stats = build_fusion_table([r.to_dict() for r in images], reports)
    if table.empty:
        log.error("mimic.empty_join")
        return {"error": "no image/report pairs found"}

    table = split_by_subject(table, train=cfg.split.train, val=cfg.split.val, seed=cfg.split.seed)
    prevalence = label_prevalence(table)
    timeline = build_patient_timeline(table)

    paths = write_fusion_dataset(table, timeline, Path(cfg.paths.processed))
    out = Path(cfg.paths.outputs)
    prevalence.to_csv(out / "label_prevalence.csv", index=False)
    log.info("mimic.prevalence\n%s", prevalence.to_string(index=False))

    n_figures = stage_figures(cfg, table, prevalence)

    qc_summary = {
        "phi": phi,
        "join_rate": round(stats.join_rate, 4),
        "n_images": stats.n_images,
        "n_reports": stats.n_reports,
        "orphan_images": stats.n_images_without_report,
        "orphan_reports": stats.n_reports_without_image,
        "prevalence": prevalence.to_dict(orient="records"),
    }

    release_path: str | None
    try:
        release_path = str(
            write_release(_release_from_table(cfg, table, qc_summary), Path(cfg.paths.releases))
        )
    except FileExistsError:
        log.warning("release.exists_skipping")
        release_path = "skipped (version exists)"

    return {
        "n_images": stats.n_images,
        "n_reports": stats.n_reports,
        "n_joined": stats.n_joined,
        "join_rate": round(stats.join_rate, 3),
        "phi_clean": phi["clean"],
        "splits": table["split"].value_counts().to_dict(),
        "n_figures": n_figures,
        "fusion_dataset": str(paths["fusion_parquet"]),
        "release": release_path,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the MIMIC-CXR multimodal pipeline.")
    parser.add_argument("--config", type=Path, default=Path("configs/mimic.yaml"))
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=args.overrides)
    configure_logging(level=cfg.log_level, json_logs=cfg.json_logs)

    log.info("pipeline.start", extra={"dataset": cfg.name})
    try:
        result = run(cfg)
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2

    print("\n" + "=" * 62)
    print(f"  MIMIC-CXR pipeline complete -- artifacts under {cfg.paths.outputs}")
    print("=" * 62)
    for key, value in result.items():
        print(f"  {key:<16} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
