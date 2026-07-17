"""Automated dataset card generation.

A dataset card is only trustworthy if it is *generated from the run*, not written
by hand. A hand-written card describes what someone believed six months ago; a
generated card describes the release that exists. Every number below traces back
to a pipeline artifact.

Structure follows the Gebru et al. "Datasheets for Datasets" framing and the
HuggingFace dataset-card conventions, with the sections a clinical reviewer
actually needs: intended use, **out-of-scope use**, bias, QC, provenance, ethics.

The "Not recommended uses" section is the one that matters most and the one most
portfolios omit. A dataset card that lists only strengths is marketing.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from common.logging import get_logger

log = get_logger(__name__)


def _table(frame: pd.DataFrame, max_rows: int = 25) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table."""
    if frame is None or frame.empty:
        return "_No data._"
    shown = frame.head(max_rows)
    header = "| " + " | ".join(str(c) for c in shown.columns) + " |"
    divider = "| " + " | ".join("---" for _ in shown.columns) + " |"
    rows = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in shown.itertuples(index=False)
    ]
    out = "\n".join([header, divider, *rows])
    if len(frame) > max_rows:
        out += f"\n\n_({len(frame) - max_rows} further rows omitted.)_"
    return out


def _bullets(items: list[str]) -> str:
    """Render a markdown bullet list, or a placeholder when empty."""
    return "\n".join(f"- {item}" for item in items) if items else "- _None recorded._"


def build_card(
    *,
    dataset: str,
    version: str,
    description: str,
    n_studies: int,
    n_subjects: int,
    dataset_hash: str,
    cohort_spec: dict[str, Any],
    attrition: pd.DataFrame,
    quality: pd.DataFrame,
    bias_findings: pd.DataFrame,
    splits: dict[str, list[str]],
    cohort_shift: pd.DataFrame | None = None,
    source: str = "",
    licence: str = "See original dataset licence.",
) -> str:
    """Render a complete dataset card as markdown.

    Args:
        dataset: Dataset name.
        version: Release version.
        description: One-paragraph purpose statement.
        n_studies: Study count in the released cohort.
        n_subjects: Distinct subject count.
        dataset_hash: Content hash from the release manifest.
        cohort_spec: Serialised :class:`~pipelines.lits.cohort_builder.CohortSpec`.
        attrition: Attrition table.
        quality: Per-study quality scores.
        bias_findings: Output of the bias audit.
        splits: Split name to subject IDs.
        cohort_shift: Optional cohort-vs-archive distribution comparison.
        source: Provenance statement.
        licence: Licence statement.

    Returns:
        The card as a markdown string.
    """
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    warnings = (
        bias_findings[bias_findings["severity"] == "WARN"]
        if not bias_findings.empty
        else pd.DataFrame()
    )
    bias_bullets = [str(row["message"]) for _, row in warnings.iterrows()]

    grade_counts = (
        quality["grade"].value_counts().sort_index().to_dict() if "grade" in quality else {}
    )
    mean_score = (
        round(float(quality["quality_score"].mean()), 1)
        if "quality_score" in quality and len(quality)
        else 0.0
    )
    low_quality = int((quality["quality_score"] < 65).sum()) if "quality_score" in quality else 0

    criteria_rows = pd.DataFrame(cohort_spec.get("criteria", []))
    split_rows = pd.DataFrame(
        [{"split": name, "n_subjects": len(members)} for name, members in splits.items()]
    )

    return f"""# Dataset Card: {dataset} ({version})

> Generated automatically by `pipelines.lits.dataset_card` on {generated}.
> Every figure below is derived from the release artifacts, not written by hand.

| | |
|---|---|
| **Dataset** | `{dataset}` |
| **Version** | `{version}` |
| **Content hash** | `{dataset_hash[:16]}` |
| **Studies** | {n_studies} |
| **Subjects** | {n_subjects} |
| **Mean quality score** | {mean_score} / 100 |
| **Licence** | {licence} |

## Purpose

{description}

## Provenance

{source or "_Not recorded._"}

Lineage from raw source to this release is recorded in the release manifest
(`releases/{dataset}/{version}/manifest.json`), including the config snapshot,
the QC thresholds in force, and the git revision of the pipeline that built it.
Per-file SHA-256 hashes are in `SHA256SUMS`; `common.versioning.verify_release`
re-checks them against disk.

## Cohort definition

**{cohort_spec.get('name', 'unnamed')}** -- {cohort_spec.get('description', '')}

{_table(criteria_rows)}

### Attrition

Studies removed by each criterion, in application order:

{_table(attrition)}

## Composition

### Splits

{_table(split_rows)}

Splits are grouped by subject: no subject appears in more than one split, and the
invariant is asserted at runtime rather than assumed.

### Quality distribution

Grades: {grade_counts or "_not computed_"}

Each study is scored 0-100 as a weighted sum of integrity, metadata completeness,
spacing consistency, noise, contrast and slice continuity. See
`pipelines/lits/quality_score.py` for the weights and the rationale for each.

{_table(quality.head(10))}

## Quality control

- Automated per-study checks run on every ingest; results in `outputs/{dataset}/QC_REPORT.html`.
- {low_quality} study(ies) scored below the grade-C threshold of 65.
- QC thresholds are configuration, not code, and are snapshotted into the release.

## Bias and limitations

The following were detected by the automated audit
(`pipelines/lits/bias_audit.py`), which pairs each distribution with a
statistical test and an effect size:

{_bullets(bias_bullets)}

{_table(bias_findings)}

{"### Cohort selection shift" + chr(10) + chr(10) + "Selection criteria shift the distribution relative to the full archive:" + chr(10) + chr(10) + _table(cohort_shift) if cohort_shift is not None and not cohort_shift.empty else ""}

## Recommended uses

- Training and evaluating liver/lesion segmentation models **within the
  represented population**.
- Benchmarking preprocessing and QC tooling.
- Methods research where the cohort's limitations are stated explicitly.

## Uses that are NOT recommended

- **Clinical deployment.** Nothing here is validated for patient care. No
  regulatory clearance, no prospective validation.
- **Claims about under-represented subgroups.** Where the audit above reports a
  dominant subgroup, performance for the minority subgroup is not measurable
  from this data at a useful confidence level.
- **Cross-institution generalisation claims.** Scanner and site are confounded
  with the cohort composition; a model may learn site signatures rather than
  anatomy.
- **Fairness auditing as the sole evidence base.** The demographic fields here
  are coarse (binary sex, age bands) and cannot support intersectional analysis.
- **Any use requiring PHI guarantees beyond the source's own de-identification.**
  This pipeline verifies de-identification; it does not re-derive it.

## Ethical considerations

- Derived from de-identified data released under its own governance terms; the
  pipeline re-verifies identifier tags on ingest and reports violations rather
  than assuming compliance.
- Demographic attributes are recorded **to enable bias auditing**, not to
  condition models on them. Training on sex or age directly would embed
  demographic shortcuts.
- Automated quality scores rank studies; they do not replace radiologist review.
  The intended workflow is automated triage followed by human spot-check.

## Maintenance

- Releases are immutable. Corrections are issued as a new version with the
  parent recorded in the manifest, never by editing a published release.
- Re-verify integrity at any time with `common.versioning.verify_release`.

---
_Card generated by the medical-data-platform governance pipeline._
"""


def write_card(card: str, path: Path) -> Path:
    """Write a dataset card to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(card, encoding="utf-8")
    log.info("dataset_card.written", extra={"path": str(path)})
    return path
