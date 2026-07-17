"""BraTS pipeline entrypoint.

Runs the full data path and emits every artifact a downstream consumer needs::

    ingest -> QC -> preprocess -> split -> visualise -> release

Each stage is a pure function over records, which is what makes the pipeline
testable: no stage reaches into global state, and any stage can be exercised in
isolation from a list of :class:`~common.metadata.ScanRecord`.

The ``@flow``/``@task`` decorators from Prefect are deliberately *not* imported
here. Orchestration is a deployment concern; ``prefect_flow.py`` wraps these same
functions when you want scheduling, retries and a UI. Keeping them out of the
core means ``python -m pipelines.brats.run`` works on a laptop with no server.

Usage:
    python -m pipelines.brats.run --config configs/brats.yaml
    python -m pipelines.brats.run --config configs/brats.yaml --set qc.min_tumor_voxels=50
    python -m pipelines.brats.run --config configs/brats.yaml --stage qc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from common.config import Config, load_config
from common.logging import configure_logging, get_logger
from common.metadata import ScanRecord, write_manifest
from common.qc import (
    apply_findings,
    check_duplicates,
    check_missing_modalities,
    registered_checks,
    run_checks,
    summarise,
    write_csv_report,
    write_html_report,
)
from common.versioning import create_release, node, write_release
from common.visualization import plot_categorical, plot_distribution, plot_slice_grid
from pipelines.brats.ingest import ingest
from pipelines.brats.preprocess import preprocess
from pipelines.brats.split import assign_splits, make_splits, split_summary

log = get_logger(__name__)


def stage_ingest(cfg: Config) -> list[ScanRecord]:
    """Scan raw data and write the raw manifest."""
    records = ingest(cfg)
    write_manifest(records, Path(cfg.paths.outputs) / "metadata" / "manifest_raw")
    return records


def stage_qc(cfg: Config, records: list[ScanRecord]) -> tuple[list[ScanRecord], dict]:
    """Run per-series and cohort-level checks; write CSV + HTML reports."""
    findings = run_checks(records, cfg)
    findings += check_duplicates(records)
    findings += check_missing_modalities(records, cfg)

    records = apply_findings(records, findings)
    report = summarise(findings, records)

    outputs = Path(cfg.paths.outputs)
    write_csv_report(findings, outputs / "QC_REPORT.csv")
    write_html_report(report, findings, outputs / "QC_REPORT.html", name=cfg.name)
    write_manifest(records, outputs / "metadata" / "manifest_qc")

    log.info("qc.summary", extra={
        "n_errors": report.n_errors,
        "n_warnings": report.n_warnings,
        "pass_rate": round(report.pass_rate, 3),
        "failed_subjects": report.failed_subjects,
    })

    if cfg.qc.fail_on == "error" and report.n_errors and not _passing(records):
        raise RuntimeError("QC rejected every subject; refusing to continue")

    return records, {
        "n_records": report.n_records,
        "n_subjects": report.n_subjects,
        "n_errors": report.n_errors,
        "n_warnings": report.n_warnings,
        "pass_rate": round(report.pass_rate, 4),
        "failed_subjects": report.failed_subjects,
        "by_check": report.by_check,
        "checks_run": registered_checks(),
    }


def _passing(records: list[ScanRecord]) -> list[ScanRecord]:
    """Records that QC did not reject."""
    return [r for r in records if r.qc_status != "fail"]


def require_complete_subjects(records: list[ScanRecord], cfg: Config) -> list[ScanRecord]:
    """Drop subjects that lack a passing series for every required modality.

    QC marks records at *series* level, but the training sample is a *subject*:
    the 3D U-Net stacks T1/T1Gd/T2/FLAIR as four input channels. A subject whose
    FLAIR failed QC still has three passing series, and keeping them would either
    crash the loader or, worse, silently train on a zero-filled channel.

    Excluding at subject level here keeps that decision explicit and logged
    rather than buried in the dataset class.
    """
    required = {m.lower() for m in cfg.qc.expected_modalities}
    if not required:
        return records

    have: dict[str, set[str]] = {}
    for rec in records:
        have.setdefault(rec.patient_id, set()).add(rec.modality)

    complete = {s for s, mods in have.items() if required <= mods}
    dropped = sorted(set(have) - complete)
    if dropped:
        log.warning("pipeline.incomplete_subjects_dropped", extra={
            "n_dropped": len(dropped),
            "subjects": dropped,
            "required": sorted(required),
        })
    return [r for r in records if r.patient_id in complete]


def stage_preprocess(cfg: Config, records: list[ScanRecord]) -> list[ScanRecord]:
    """Normalise QC-passing volumes and write the processed manifest."""
    processed = preprocess(records, cfg)
    write_manifest(processed, Path(cfg.paths.outputs) / "metadata" / "manifest_processed")
    return processed


def stage_split(cfg: Config, records: list[ScanRecord]) -> tuple[list[ScanRecord], dict]:
    """Partition subjects, stamp the records, and refresh the processed manifest."""
    splits = make_splits(records, cfg)
    records = assign_splits(records, splits)

    summary = split_summary(records, splits)
    out = Path(cfg.paths.outputs) / "metadata"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "splits.csv", index=False)

    # Rewrite the processed manifest now that `extra.split` is populated. The
    # manifest is the single source of truth for downstream consumers -- the
    # trainer reads splits from here rather than globbing the filesystem, so a
    # manifest without splits silently yields an empty training set.
    write_manifest(records, out / "manifest_processed")

    log.info("split.summary\n%s", summary.to_string(index=False))
    return records, splits


def stage_visualise(cfg: Config, records: list[ScanRecord], n: int = 3) -> list[Path]:
    """Export review figures: slice overlays plus cohort distributions.

    Renders the first ``n`` subjects that have a mask. These land in
    ``outputs/brats/figures/`` and are what the README embeds -- a reviewer sees
    the data without running anything.
    """
    figures_dir = Path(cfg.paths.outputs) / "figures"
    processed_root = Path(cfg.paths.processed)
    written: list[Path] = []

    with_mask = [r for r in records if r.mask_path and r.modality == "t1ce"]
    for rec in with_mask[:n]:
        try:
            volume = np.asanyarray(nib.load(str(processed_root / rec.filepath)).dataobj)
            mask = np.asanyarray(nib.load(str(processed_root / rec.mask_path)).dataobj)
            written.append(plot_slice_grid(
                volume, mask,
                figures_dir / f"{rec.patient_id}_overlay.png",
                title=f"{rec.patient_id} ({rec.modality.upper()})",
            ))
        except Exception as exc:
            log.warning("viz.failed", extra={"patient": rec.patient_id, "error": str(exc)})

    volumes = [r.tumor_volume_mm3 for r in records if r.tumor_volume_mm3]
    if volumes:
        written.append(plot_distribution(
            volumes, figures_dir / "tumor_volume_distribution.png",
            "Tumour burden across cohort", "tumour volume (mm3)",
        ))

    counts: dict[str, int] = {}
    for rec in records:
        counts[rec.modality] = counts.get(rec.modality, 0) + 1
    written.append(plot_categorical(
        counts, figures_dir / "modality_counts.png", "Series per modality", "modality",
    ))

    log.info("viz.complete", extra={"n_figures": len(written)})
    return written


def stage_release(
    cfg: Config, records: list[ScanRecord], splits: dict, qc_summary: dict
) -> Path | None:
    """Cut an immutable, content-addressed dataset release.

    Returns ``None`` when the version already exists -- re-running the pipeline
    must not be an error, but it also must not mutate a published release.
    """
    version = str(cfg.extra.get("dataset_version", "v1.0.0"))
    lineage = [
        node("ingest", "Scanned raw BraTS tree; hashed and probed every series.",
             raw_root=str(cfg.paths.raw)),
        node("qc", "Ran per-series and cohort QC checks.",
             checks=registered_checks(), thresholds=cfg.qc.model_dump(mode="json")),
        node("preprocess", "Reoriented to RAS, resampled to isotropic spacing, z-scored.",
             **cfg.preprocess.model_dump(mode="json")),
        node("split", "Patient-grouped, tumour-burden-stratified partition.",
             **cfg.split.model_dump(mode="json")),
    ]

    release = create_release(
        dataset="brats", version=version, records=records, cfg=cfg,
        splits=splits, lineage=lineage, qc_summary=qc_summary,
    )
    try:
        return write_release(release, Path(cfg.paths.releases))
    except FileExistsError:
        log.warning("release.exists_skipping", extra={"version": version})
        return None


def run(cfg: Config, stage: str = "all") -> dict:
    """Execute the pipeline up to and including ``stage``.

    Args:
        cfg: Resolved configuration.
        stage: One of ``ingest``, ``qc``, ``preprocess``, ``split``, ``all``.

    Returns:
        Summary counts for the stages that ran.
    """
    cfg.paths.mkdirs()
    order = ["ingest", "qc", "preprocess", "split", "all"]
    if stage not in order:
        raise ValueError(f"unknown stage {stage!r}; expected one of {order}")
    limit = order.index(stage)

    records = stage_ingest(cfg)
    result: dict = {"n_ingested": len(records)}
    if limit == 0:
        return result

    records, qc_summary = stage_qc(cfg, records)
    result["qc"] = qc_summary
    if limit == 1:
        return result

    passing = require_complete_subjects(_passing(records), cfg)
    result["n_subjects_eligible"] = len({r.patient_id for r in passing})
    processed = stage_preprocess(cfg, passing)
    result["n_processed"] = len(processed)
    if limit == 2:
        return result

    processed, splits = stage_split(cfg, processed)
    result["splits"] = {k: len(v) for k, v in splits.items()}
    if limit == 3:
        return result

    figures = stage_visualise(cfg, processed)
    result["n_figures"] = len(figures)

    manifest = stage_release(cfg, processed, splits, qc_summary)
    result["release"] = str(manifest) if manifest else "skipped (version exists)"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the BraTS data pipeline.")
    parser.add_argument("--config", type=Path, default=Path("configs/brats.yaml"))
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="dotted config override; repeatable")
    parser.add_argument("--stage", default="all",
                        choices=["ingest", "qc", "preprocess", "split", "all"])
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=args.overrides)
    configure_logging(level=cfg.log_level, json_logs=cfg.json_logs)

    log.info("pipeline.start", extra={"dataset": cfg.name, "stage": args.stage})
    try:
        result = run(cfg, stage=args.stage)
    except FileNotFoundError as exc:
        log.error("pipeline.missing_data", extra={"error": str(exc)})
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2

    log.info("pipeline.complete", extra=result)
    print("\n" + "=" * 62)
    print(f"  BraTS pipeline complete -- artifacts under {cfg.paths.outputs}")
    print("=" * 62)
    for key, value in result.items():
        print(f"  {key:<14} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
