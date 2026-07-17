"""LiTS governance pipeline entrypoint.

Runs::

    ingest metadata -> quality scoring -> cohort build -> bias audit
    -> figures -> dataset card -> release

No model is trained. The deliverable is the governance artifact set: an audited,
defensible, documented cohort. That is the Data Manager's product, and it is the
part of the lifecycle that determines whether every downstream model is
trustworthy.

Usage:
    python -m pipelines.lits.run --config configs/lits.yaml
    python -m pipelines.lits.run --set extra.cohort=exploratory_all_liver_ct
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd

from common.config import Config, load_config
from common.logging import configure_logging, get_logger
from common.metadata import ScanRecord, write_manifest
from common.storage import file_size, sha256_file
from common.versioning import create_release, node, write_release
from pipelines.lits.bias_audit import audit_cohort, findings_to_frame, plot_bias_figures
from pipelines.lits.cohort_builder import PRESETS, build_cohort, compare_cohorts
from pipelines.lits.dataset_card import build_card, write_card
from pipelines.lits.quality_score import score_cohort

log = get_logger(__name__)


def stage_ingest(cfg: Config) -> pd.DataFrame:
    """Join the clinical metadata CSV to the volume headers on disk.

    The clinical CSV is the source of demographics; the NIfTI headers are the
    source of geometry. Neither alone supports the audit, and a study present in
    one but not the other is itself a finding worth recording.
    """
    raw = Path(cfg.paths.raw)
    csv_path = raw / "clinical_metadata.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"clinical metadata not found: {csv_path}\n"
            "Run: python scripts/generate_synthetic_data.py --dataset lits"
        )

    frame = pd.read_csv(csv_path)
    rows: list[dict[str, Any]] = []

    for row in frame.to_dict(orient="records"):
        volume_path = raw / str(row["volume_path"])
        label_path = raw / str(row["label_path"])
        record = dict(row)

        record["has_label"] = label_path.exists()
        record["file_exists"] = volume_path.exists()
        record["voxel_spacing"] = None
        record["shape"] = None
        record["slice_thickness_mm"] = None
        record["tumor_volume_mm3"] = None
        record["sha256"] = None
        record["file_size_bytes"] = None

        if volume_path.exists():
            record["sha256"] = sha256_file(volume_path)
            record["file_size_bytes"] = file_size(volume_path)
            try:
                img = nib.load(str(volume_path))
                spacing = [float(z) for z in img.header.get_zooms()[:3]]
                record["voxel_spacing"] = spacing
                record["shape"] = [int(s) for s in img.shape]
                record["slice_thickness_mm"] = max(spacing)
            except Exception as exc:
                log.warning(
                    "lits.header_unreadable", extra={"path": str(volume_path), "error": str(exc)}
                )

        if label_path.exists() and record["voxel_spacing"]:
            try:
                mask = np.asanyarray(nib.load(str(label_path)).dataobj)
                record["tumor_volume_mm3"] = float(
                    np.count_nonzero(mask == 2) * np.prod(record["voxel_spacing"])
                )
            except Exception as exc:
                log.warning(
                    "lits.label_unreadable", extra={"path": str(label_path), "error": str(exc)}
                )

        rows.append(record)

    out = pd.DataFrame(rows)
    log.info(
        "lits.ingested",
        extra={
            "n_studies": len(out),
            "n_with_label": int(out["has_label"].sum()),
            "n_unreadable": int(out["voxel_spacing"].isna().sum()),
        },
    )
    return out


def stage_quality(cfg: Config, frame: pd.DataFrame) -> pd.DataFrame:
    """Score every study and merge the scores into the table."""
    scores = score_cohort(frame.to_dict(orient="records"), Path(cfg.paths.raw))
    quality = pd.DataFrame([s.to_dict() for s in scores])

    merged = frame.merge(quality, on="patient_id", how="left")
    out = Path(cfg.paths.outputs)
    out.mkdir(parents=True, exist_ok=True)
    quality.to_csv(out / "quality_scores.csv", index=False)

    log.info(
        "lits.quality_summary",
        extra={
            "mean": round(float(quality["quality_score"].mean()), 1),
            "min": round(float(quality["quality_score"].min()), 1),
            "grades": quality["grade"].value_counts().to_dict(),
        },
    )
    return merged


def stage_cohort(cfg: Config, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    """Apply the configured cohort spec and write the attrition trail."""
    name = str(cfg.extra.get("cohort", "adult_contrast_liver_ct"))
    if name not in PRESETS:
        raise ValueError(f"unknown cohort {name!r}; available: {sorted(PRESETS)}")

    spec = PRESETS[name]
    cohort, attrition = build_cohort(frame, spec)

    out = Path(cfg.paths.outputs)
    attrition.to_csv(out / "cohort_attrition.csv", index=False)
    log.info("lits.attrition\n%s", attrition.to_string(index=False))
    return cohort, attrition, spec


def stage_bias(
    cfg: Config, cohort: pd.DataFrame, full: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit the cohort and compare it against the full archive."""
    findings = audit_cohort(cohort, outcome_columns=("quality_score", "tumor_volume_mm3"))
    frame = findings_to_frame(findings)

    shift = compare_cohorts(cohort, full, columns=("sex", "institution", "scanner"))

    out = Path(cfg.paths.outputs)
    frame.to_csv(out / "bias_audit.csv", index=False)
    shift.to_csv(out / "cohort_shift.csv", index=False)
    plot_bias_figures(cohort, out / "figures")

    warnings = frame[frame["severity"] == "WARN"] if not frame.empty else pd.DataFrame()
    for _, row in warnings.iterrows():
        log.warning(
            "bias.finding", extra={"attribute": row["attribute"], "message": row["message"]}
        )
    return frame, shift


def stage_splits(cfg: Config, cohort: pd.DataFrame) -> dict[str, list[str]]:
    """Partition the cohort by patient."""
    rng = np.random.default_rng(cfg.split.seed)
    subjects = sorted(cohort["patient_id"].astype(str).unique())
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = max(1, round(n * cfg.split.train)) if n >= 3 else n
    n_val = max(1, round(n * cfg.split.val)) if n >= 3 else 0
    if n >= 3:
        n_train = min(n_train, n - 2)

    splits = {
        "train": sorted(subjects[:n_train]),
        "val": sorted(subjects[n_train : n_train + n_val]),
        "test": sorted(subjects[n_train + n_val :]),
    }
    log.info("lits.splits", extra={k: len(v) for k, v in splits.items()})
    return splits


def _records(cohort: pd.DataFrame, splits: dict[str, list[str]]) -> list[ScanRecord]:
    """Adapt cohort rows to the shared release schema."""
    lookup = {s: name for name, members in splits.items() for s in members}
    records: list[ScanRecord] = []

    for row in cohort.to_dict(orient="records"):
        spacing = row.get("voxel_spacing")
        if isinstance(spacing, str):
            spacing = None
        records.append(
            ScanRecord(
                patient_id=str(row["patient_id"]),
                modality="ct",
                filepath=str(row["volume_path"]),
                shape=list(row["shape"]) if isinstance(row.get("shape"), list) else [],
                voxel_spacing=list(spacing) if isinstance(spacing, list) else [],
                mask_available=bool(row.get("has_label")),
                mask_path=str(row.get("label_path")) if row.get("has_label") else None,
                tumor_volume_mm3=(
                    row.get("tumor_volume_mm3") if pd.notna(row.get("tumor_volume_mm3")) else None
                ),
                scanner=str(row.get("scanner")) if pd.notna(row.get("scanner")) else None,
                institution=(
                    str(row.get("institution")) if pd.notna(row.get("institution")) else None
                ),
                age=float(row["age"]) if pd.notna(row.get("age")) else None,
                sex=str(row.get("sex")) if pd.notna(row.get("sex")) else None,
                sha256=str(row.get("sha256")) if pd.notna(row.get("sha256")) else None,
                file_size_bytes=(
                    int(row["file_size_bytes"]) if pd.notna(row.get("file_size_bytes")) else None
                ),
                quality_score=(
                    float(row["quality_score"]) if pd.notna(row.get("quality_score")) else None
                ),
                qc_status="pass",
                label_classes=[1, 2] if row.get("has_label") else [],
                extra={
                    "split": lookup.get(str(row["patient_id"]), "unassigned"),
                    "grade": str(row.get("grade", "")),
                },
            )
        )
    return records


def run(cfg: Config) -> dict:
    """Execute the governance pipeline and return summary counts."""
    cfg.paths.mkdirs()

    full = stage_quality(cfg, stage_ingest(cfg))
    cohort, attrition, spec = stage_cohort(cfg, full)

    if cohort.empty:
        log.error("lits.empty_cohort", extra={"cohort": spec.name})
        return {"error": "cohort is empty; loosen the criteria"}

    bias, shift = stage_bias(cfg, cohort, full)
    splits = stage_splits(cfg, cohort)

    records = _records(cohort, splits)
    write_manifest(records, Path(cfg.paths.outputs) / "metadata" / "cohort_manifest")

    version = str(cfg.extra.get("dataset_version", "v1.0.0"))
    release = create_release(
        dataset="lits",
        version=version,
        records=records,
        cfg=cfg,
        splits=splits,
        qc_summary={
            "mean_quality_score": round(float(cohort["quality_score"].mean()), 2),
            "n_below_threshold": int((cohort["quality_score"] < 65).sum()),
            "bias_warnings": (
                bias[bias["severity"] == "WARN"]["message"].tolist() if not bias.empty else []
            ),
        },
        lineage=[
            node("ingest", "Joined clinical metadata to NIfTI headers.", root=str(cfg.paths.raw)),
            node("quality_score", "Scored each study 0-100 on six weighted components."),
            # Nested under `spec` rather than splatted: the spec dict has its own
            # `description` key, which would collide with node()'s parameter.
            node("cohort", f"Applied cohort spec {spec.name}.", spec=spec.to_dict()),
            node("bias_audit", "Chi-square/Kruskal-Wallis audit with effect sizes."),
        ],
    )

    release_path: str
    try:
        release_path = str(write_release(release, Path(cfg.paths.releases)))
    except FileExistsError:
        log.warning("release.exists_skipping", extra={"version": version})
        release_path = "skipped (version exists)"

    card = build_card(
        dataset="lits",
        version=version,
        description=spec.description,
        n_studies=len(cohort),
        n_subjects=int(cohort["patient_id"].nunique()),
        dataset_hash=release.dataset_hash,
        cohort_spec=spec.to_dict(),
        attrition=attrition,
        quality=cohort[["patient_id", "quality_score", "grade", "reasons"]],
        bias_findings=bias,
        splits=splits,
        cohort_shift=shift,
        source=(
            "Derived from the LiTS (Liver Tumour Segmentation) challenge structure. "
            "This run used the synthetic generator; point `paths.raw` at a real "
            "download to reproduce against actual data."
        ),
        licence="LiTS challenge terms (CC BY-NC-ND 4.0 for the original data).",
    )
    card_path = write_card(card, Path(cfg.paths.outputs) / "dataset_card.md")

    return {
        "n_studies_total": len(full),
        "n_studies_cohort": len(cohort),
        "retention": f"{len(cohort) / len(full):.0%}" if len(full) else "0%",
        "mean_quality": round(float(cohort["quality_score"].mean()), 1),
        "bias_warnings": int((bias["severity"] == "WARN").sum()) if not bias.empty else 0,
        "splits": {k: len(v) for k, v in splits.items()},
        "dataset_card": str(card_path),
        "release": release_path,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the LiTS governance pipeline.")
    parser.add_argument("--config", type=Path, default=Path("configs/lits.yaml"))
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=args.overrides)
    configure_logging(level=cfg.log_level, json_logs=cfg.json_logs)

    log.info("pipeline.start", extra={"dataset": cfg.name})
    try:
        result = run(cfg)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2

    print("\n" + "=" * 62)
    print(f"  LiTS governance pipeline complete -- artifacts under {cfg.paths.outputs}")
    print("=" * 62)
    for key, value in result.items():
        print(f"  {key:<18} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
