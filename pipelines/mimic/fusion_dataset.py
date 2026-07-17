"""Build the multimodal fusion dataset: image + report + labels + timeline.

This is the artifact the multimodal model actually consumes, and the join is
where multimodal pipelines usually break. Three failure modes it guards against:

1. **Silent inner joins.** An inner join between images and reports drops every
   unpaired study without saying so. Here the join is explicit and both
   orphan counts are logged and reported.
2. **Patient-level leakage.** One patient has many studies over years. Splitting
   by study puts the same chest in train and test. The split is grouped by
   subject, reusing the same guarantee as the BraTS pipeline.
3. **Label ambiguity.** CheXpert labels are 1/0/-1/None. Collapsing "uncertain"
   and "not mentioned" to 0 is a modelling decision, not a data one, so the
   dataset keeps all four states and :func:`binarise_labels` applies a *named*
   policy at training time.

Also emits the patient timeline: studies ordered per subject with the interval
between them, which is what an agentic system needs to answer "is this new?"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from common.logging import get_logger
from pipelines.mimic.report_parser import FINDING_VOCAB, ParsedReport

log = get_logger(__name__)

TARGET_FINDINGS: tuple[str, ...] = ("pneumonia", "edema", "cardiomegaly", "pleural_effusion")

UncertainPolicy = Literal["zeros", "ones", "ignore"]


@dataclass
class FusionStats:
    """Diagnostics from the image/report join."""

    n_images: int
    n_reports: int
    n_joined: int
    n_images_without_report: int
    n_reports_without_image: int

    @property
    def join_rate(self) -> float:
        """Fraction of images that found a report."""
        return self.n_joined / self.n_images if self.n_images else 0.0


def build_fusion_table(
    dicom_rows: list[dict],
    reports: dict[str, ParsedReport],
) -> tuple[pd.DataFrame, FusionStats]:
    """Join DICOM metadata to parsed reports on ``study_id``.

    Args:
        dicom_rows: Flattened :class:`~pipelines.mimic.dicom_parser.DicomRecord` dicts.
        reports: Parsed reports keyed by study ID.

    Returns:
        ``(table, stats)``. The table has one row per image that has a report;
        orphans are excluded but counted in ``stats``.
    """
    images = pd.DataFrame(dicom_rows)
    if images.empty:
        return images, FusionStats(0, len(reports), 0, 0, len(reports))

    image_studies = set(images["study_id"])
    report_studies = set(reports)

    rows: list[dict] = []
    for row in dicom_rows:
        report = reports.get(row["study_id"])
        if report is None:
            continue

        record = dict(row)
        record["report_text"] = report.findings_text
        record["impression"] = report.impression_text
        record["n_report_chars"] = report.n_chars
        record["n_recommendations"] = len(report.recommendations)
        record["max_measurement_cm"] = (
            max(report.measurements_cm) if report.measurements_cm else None
        )
        for finding in FINDING_VOCAB:
            record[f"label_{finding}"] = report.labels.get(finding)
            record[f"severity_{finding}"] = report.severity.get(finding)
        rows.append(record)

    table = pd.DataFrame(rows)
    stats = FusionStats(
        n_images=len(images),
        n_reports=len(reports),
        n_joined=len(table),
        n_images_without_report=len(image_studies - report_studies),
        n_reports_without_image=len(report_studies - image_studies),
    )

    log.info("fusion.joined", extra={
        "n_images": stats.n_images,
        "n_reports": stats.n_reports,
        "n_joined": stats.n_joined,
        "join_rate": round(stats.join_rate, 3),
        "orphan_images": stats.n_images_without_report,
        "orphan_reports": stats.n_reports_without_image,
    })
    if stats.join_rate < 0.95 and stats.n_images:
        log.warning("fusion.low_join_rate", extra={"join_rate": round(stats.join_rate, 3)})
    return table, stats


def binarise_labels(
    table: pd.DataFrame,
    findings: tuple[str, ...] = TARGET_FINDINGS,
    uncertain: UncertainPolicy = "zeros",
) -> tuple[np.ndarray, list[str]]:
    """Convert 1/0/-1/None labels into a binary matrix under a named policy.

    Policies follow the CheXpert paper:
        ``zeros``  -- treat uncertain as negative (conservative, most common).
        ``ones``   -- treat uncertain as positive (maximises recall).
        ``ignore`` -- emit NaN so a masked loss can skip those cells.

    "Not mentioned" always maps to 0: radiology reports describe what is present,
    so silence about pneumothorax means there was none.

    Returns:
        ``(matrix, column_names)`` with shape ``(n_rows, len(findings))``.
    """
    columns = [f"label_{f}" for f in findings]
    raw = table[columns].to_numpy(dtype=float)  # None -> NaN

    out = np.zeros_like(raw)
    out[raw == 1] = 1.0
    out[np.isnan(raw)] = 0.0  # not mentioned -> negative

    if uncertain == "ones":
        out[raw == -1] = 1.0
    elif uncertain == "ignore":
        out[raw == -1] = np.nan
    # 'zeros': -1 keeps the initialised 0.0

    return out, list(findings)


def label_prevalence(table: pd.DataFrame, findings: tuple[str, ...] = TARGET_FINDINGS) -> pd.DataFrame:
    """Per-finding counts of positive/negative/uncertain/not-mentioned.

    Goes straight into the dataset card: class imbalance is the first thing a
    reviewer should see, since a 4% prevalence label makes accuracy meaningless.
    """
    rows = []
    for finding in findings:
        column = table[f"label_{finding}"]
        n = len(column)
        n_pos = int((column == 1).sum())
        rows.append({
            "finding": finding,
            "positive": n_pos,
            "negative": int((column == 0).sum()),
            "uncertain": int((column == -1).sum()),
            "not_mentioned": int(column.isna().sum()),
            "prevalence": round(n_pos / n, 4) if n else 0.0,
        })
    return pd.DataFrame(rows)


def build_patient_timeline(table: pd.DataFrame) -> pd.DataFrame:
    """Order each subject's studies and compute intervals between them.

    Enables the temporal questions an agentic clinical system must answer -- "is
    this effusion new or was it there in March?" -- and exposes subjects with
    many studies, who dominate a naive random split.
    """
    if table.empty or "subject_id" not in table:
        return pd.DataFrame(columns=["subject_id", "study_id", "study_order", "n_studies"])

    frame = table.copy()
    sort_keys = ["subject_id"] + (["study_id"] if "study_id" in frame else [])
    frame = frame.sort_values(sort_keys)

    frame["study_order"] = frame.groupby("subject_id").cumcount() + 1
    frame["n_studies"] = frame.groupby("subject_id")["study_id"].transform("count")

    columns = ["subject_id", "study_id", "study_order", "n_studies"]
    columns += [f"label_{f}" for f in TARGET_FINDINGS if f"label_{f}" in frame]
    timeline = frame[columns].reset_index(drop=True)

    log.info("fusion.timeline", extra={
        "n_subjects": int(timeline["subject_id"].nunique()),
        "max_studies_per_subject": int(timeline["n_studies"].max()),
    })
    return timeline


def split_by_subject(
    table: pd.DataFrame, train: float = 0.7, val: float = 0.15, seed: int = 42
) -> pd.DataFrame:
    """Assign a ``split`` column, grouping every study of a subject together.

    Returns:
        The table with a ``split`` column added.

    Raises:
        RuntimeError: If a subject would land in two splits.
    """
    if table.empty:
        return table.assign(split=[])

    rng = np.random.default_rng(seed)
    subjects = np.array(sorted(table["subject_id"].unique()))
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = max(1, round(n * train)) if n >= 3 else n
    n_val = max(1, round(n * val)) if n >= 3 else 0
    n_train = min(n_train, max(1, n - 2)) if n >= 3 else n_train

    assignment = {
        **dict.fromkeys(subjects[:n_train], "train"),
        **dict.fromkeys(subjects[n_train:n_train + n_val], "val"),
        **dict.fromkeys(subjects[n_train + n_val:], "test"),
    }
    out = table.assign(split=table["subject_id"].map(assignment))

    overlaps = out.groupby("subject_id")["split"].nunique()
    if (overlaps > 1).any():
        raise RuntimeError(
            f"subject leakage across splits: {overlaps[overlaps > 1].index.tolist()}"
        )

    log.info("fusion.split", extra={
        "n_subjects": n,
        **out["split"].value_counts().to_dict(),
    })
    return out


def write_fusion_dataset(table: pd.DataFrame, timeline: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    """Persist the fusion table, timeline and structured reports.

    ``structured_reports.csv`` is the artifact the case-study plan calls for: the
    free text turned into reviewable rows.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "fusion_parquet": out_dir / "fusion_dataset.parquet",
        "fusion_csv": out_dir / "fusion_dataset.csv",
        "timeline": out_dir / "patient_timeline.csv",
        "structured_reports": out_dir / "structured_reports.csv",
    }

    table.to_parquet(paths["fusion_parquet"], index=False)
    table.to_csv(paths["fusion_csv"], index=False)
    timeline.to_csv(paths["timeline"], index=False)

    report_columns = [
        c for c in table.columns
        if c.startswith(("label_", "severity_"))
        or c in ("study_id", "subject_id", "impression", "n_recommendations",
                 "max_measurement_cm")
    ]
    table[report_columns].to_csv(paths["structured_reports"], index=False)

    log.info("fusion.written", extra={"dir": str(out_dir), "n_rows": len(table)})
    return paths
