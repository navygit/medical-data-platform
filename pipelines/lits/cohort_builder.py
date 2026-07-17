"""Declarative cohort construction with inclusion/exclusion audit trails.

A cohort is a *named, versioned, reproducible* subset of the archive defined by
explicit criteria -- not a pandas filter someone typed once in a notebook.

The critical output is not the cohort but the **attrition table**: how many
studies each criterion removed, in order. That table is what makes a cohort
defensible ("why are there only 340 studies?") and it is the first thing a
reviewer or regulator asks for. It is also what reveals when a criterion is
accidentally destroying the cohort.

Example:
    >>> spec = CohortSpec(
    ...     name="adult_contrast_liver_ct",
    ...     criteria=[
    ...         Criterion("adult", "age >= 18"),
    ...         Criterion("contrast", "contrast == True"),
    ...         Criterion("quality", "quality_score >= 65"),
    ...     ],
    ... )
    >>> cohort, attrition = build_cohort(frame, spec)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from common.logging import get_logger

log = get_logger(__name__)


@dataclass
class Criterion:
    """One inclusion rule, expressed as a pandas query string."""

    name: str
    expression: str
    rationale: str = ""


@dataclass
class CohortSpec:
    """A named, versioned cohort definition."""

    name: str
    criteria: list[Criterion] = field(default_factory=list)
    description: str = ""
    version: str = "v1.0.0"

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the dataset card and release manifest."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "criteria": [
                {"name": c.name, "expression": c.expression, "rationale": c.rationale}
                for c in self.criteria
            ],
        }


# Cohorts referenced by the case study. Each mirrors a real clinical question.
PRESETS: dict[str, CohortSpec] = {
    "adult_contrast_liver_ct": CohortSpec(
        name="adult_contrast_liver_ct",
        description=(
            "Adult contrast-enhanced abdominal CT with a liver segmentation, "
            "suitable for supervised liver/lesion segmentation training."
        ),
        criteria=[
            Criterion(
                "adult",
                "age >= 18",
                "Paediatric liver anatomy differs; a mixed cohort needs a separate model.",
            ),
            Criterion(
                "contrast_enhanced",
                "contrast == True",
                "Lesion conspicuity depends on contrast; mixing phases confounds the label.",
            ),
            Criterion("has_label", "has_label == True", "Supervised training requires a mask."),
            Criterion(
                "quality_threshold",
                "quality_score >= 65",
                "Grade C or better; excludes corrupt and severely degraded studies.",
            ),
        ],
    ),
    "exploratory_all_liver_ct": CohortSpec(
        name="exploratory_all_liver_ct",
        description=(
            "All readable liver CT regardless of contrast phase or age. For "
            "exploratory analysis and pretraining only -- not for evaluation."
        ),
        criteria=[
            Criterion(
                "readable",
                "quality_score >= 30",
                "Excludes only unusable studies; deliberately permissive.",
            ),
        ],
    ),
}


def build_cohort(frame: pd.DataFrame, spec: CohortSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a cohort spec and return the cohort plus its attrition table.

    Criteria are applied in order and each one's removal count is recorded. A
    criterion that fails to evaluate (typo, missing column) raises rather than
    being skipped -- a silently ignored inclusion rule produces a cohort that
    does not match its own definition, which is worse than a crash.

    Args:
        frame: The full study table.
        spec: The cohort definition.

    Returns:
        ``(cohort, attrition)``.

    Raises:
        ValueError: If a criterion expression cannot be evaluated.
    """
    current = frame.copy()
    rows: list[dict[str, Any]] = [
        {
            "step": 0,
            "criterion": "all_studies",
            "expression": "-",
            "n_before": len(frame),
            "n_after": len(frame),
            "n_removed": 0,
        }
    ]

    for i, criterion in enumerate(spec.criteria, start=1):
        before = len(current)
        try:
            current = current.query(criterion.expression)
        except Exception as exc:
            raise ValueError(
                f"criterion {criterion.name!r} failed: {criterion.expression!r} -- {exc}"
            ) from exc

        rows.append(
            {
                "step": i,
                "criterion": criterion.name,
                "expression": criterion.expression,
                "n_before": before,
                "n_after": len(current),
                "n_removed": before - len(current),
            }
        )
        log.info(
            "cohort.criterion_applied",
            extra={
                "cohort": spec.name,
                "criterion": criterion.name,
                "n_removed": before - len(current),
                "n_remaining": len(current),
            },
        )

    attrition = pd.DataFrame(rows)
    retained = len(current) / len(frame) if len(frame) else 0.0

    log.info(
        "cohort.built",
        extra={
            "cohort": spec.name,
            "n_input": len(frame),
            "n_output": len(current),
            "retention_rate": round(retained, 3),
        },
    )
    if retained < 0.2 and len(frame):
        log.warning(
            "cohort.severe_attrition",
            extra={
                "cohort": spec.name,
                "retention_rate": round(retained, 3),
            },
        )
    return current.reset_index(drop=True), attrition


def compare_cohorts(
    cohort: pd.DataFrame, full: pd.DataFrame, columns: tuple[str, ...]
) -> pd.DataFrame:
    """Compare attribute distributions between a cohort and the full archive.

    Selection criteria are themselves a source of bias: filtering to
    contrast-enhanced studies may also filter out a site that rarely uses
    contrast. This surfaces that shift instead of leaving it to be discovered
    after deployment.

    Returns:
        One row per attribute value, with cohort/full shares and the delta.
    """
    rows: list[dict[str, Any]] = []
    for column in columns:
        if column not in cohort or column not in full:
            continue
        cohort_share = cohort[column].fillna("unknown").value_counts(normalize=True)
        full_share = full[column].fillna("unknown").value_counts(normalize=True)
        for value in sorted(set(full_share.index) | set(cohort_share.index), key=str):
            in_cohort = float(cohort_share.get(value, 0.0))
            in_full = float(full_share.get(value, 0.0))
            rows.append(
                {
                    "attribute": column,
                    "value": value,
                    "share_in_cohort": round(in_cohort, 4),
                    "share_in_full": round(in_full, 4),
                    "delta": round(in_cohort - in_full, 4),
                }
            )
    return pd.DataFrame(rows)
