"""Cohort bias auditing.

Plots alone do not constitute an audit. A bar chart showing 68% male tells you
the number but not whether it *matters*, and "looks skewed" is not something a
governance sign-off can rest on. So every audit here pairs a distribution with a
**statistical test and an effect size**:

- **Representation** vs. a reference population -> chi-square goodness-of-fit,
  plus Cramer's V for effect size.
- **Subgroup outcome disparity** (disease prevalence, quality score by site)
  -> chi-square / Kruskal-Wallis, plus the max-min disparity ratio.

Effect size is reported alongside p-values on purpose: with 300k studies every
trivial difference is "significant", so a p-value alone would flag everything.
With 12 studies nothing is significant, so a p-value alone would flag nothing.
The disparity ratio is what a reviewer should act on.

Reference distributions default to broad US population/clinical values and are
overridable -- the "correct" balance is a function of the deployment population,
not a universal constant.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common.logging import get_logger
from common.visualization import plot_categorical, plot_distribution

log = get_logger(__name__)

# Reference distributions. Deliberately explicit and overridable: auditing a
# paediatric cohort against adult norms would produce nonsense findings.
REFERENCE: dict[str, dict[str, float]] = {
    "sex": {"M": 0.5, "F": 0.5},
    "age_band": {"<40": 0.20, "40-59": 0.30, "60-79": 0.35, "80+": 0.15},
}

AGE_BANDS: tuple[tuple[str, float, float], ...] = (
    ("<40", 0, 40),
    ("40-59", 40, 60),
    ("60-79", 60, 80),
    ("80+", 80, 200),
)

# A subgroup holding >50% of a cohort dominates what the model learns.
DOMINANCE_THRESHOLD = 0.50
# Prevalence differing >2x across subgroups signals a confound worth explaining.
DISPARITY_THRESHOLD = 2.0


@dataclass
class BiasFinding:
    """One audited attribute."""

    attribute: str
    kind: str
    severity: str
    message: str
    statistic: float | None = None
    p_value: float | None = None
    effect_size: float | None = None
    distribution: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Flatten for DataFrame construction."""
        return {
            "attribute": self.attribute,
            "kind": self.kind,
            "severity": self.severity,
            "message": self.message,
            "statistic": round(self.statistic, 4) if self.statistic is not None else None,
            "p_value": round(self.p_value, 5) if self.p_value is not None else None,
            "effect_size": round(self.effect_size, 4) if self.effect_size is not None else None,
        }


def age_band(age: float | None) -> str:
    """Bucket an age into a reporting band."""
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return "unknown"
    for name, lo, hi in AGE_BANDS:
        if lo <= age < hi:
            return name
    return "unknown"


def cramers_v(observed: np.ndarray, chi2: float) -> float:
    """Cramer's V effect size for a chi-square statistic.

    Ranges 0 (no association) to 1 (complete). Interpreted as ~0.1 small,
    ~0.3 medium, ~0.5 large.
    """
    n = observed.sum()
    if n == 0:
        return 0.0
    if observed.ndim == 1:
        k = len(observed)
        return float(np.sqrt(chi2 / (n * max(k - 1, 1))))
    r, k = observed.shape
    return float(np.sqrt(chi2 / (n * max(min(r - 1, k - 1), 1))))


def audit_representation(
    frame: pd.DataFrame, column: str, reference: dict[str, float] | None = None
) -> BiasFinding:
    """Compare an attribute's distribution against a reference population.

    Args:
        frame: Cohort table.
        column: Attribute to audit.
        reference: Expected proportions. Falls back to :data:`REFERENCE`, and to
            a uniform distribution when the attribute has no known reference
            (e.g. scanner make, where no natural "correct" mix exists).

    Returns:
        The finding, with a chi-square test when sample size permits.
    """
    from scipy import stats

    counts = frame[column].fillna("unknown").value_counts()
    total = int(counts.sum())
    if total == 0:
        return BiasFinding(column, "representation", "INFO", "no data")

    proportions = (counts / total).to_dict()
    top_category, top_share = max(proportions.items(), key=lambda kv: kv[1])

    expected_map = reference or REFERENCE.get(column)
    statistic = p_value = effect = None
    message = f"{top_category} accounts for {top_share:.0%} of the cohort"
    severity = "INFO"

    if expected_map:
        categories = [c for c in expected_map if c in counts.index]
        if categories and len(categories) > 1:
            observed = np.array([counts[c] for c in categories], dtype=float)
            weights = np.array([expected_map[c] for c in categories], dtype=float)
            expected = weights / weights.sum() * observed.sum()

            # Chi-square is unreliable when any expected cell < 5.
            if (expected >= 5).all():
                statistic, p_value = stats.chisquare(observed, expected)
                effect = cramers_v(observed, float(statistic))
                if p_value < 0.05 and effect > 0.1:
                    severity = "WARN"
                    message = (
                        f"{column} distribution differs from reference "
                        f"(chi2={statistic:.1f}, p={p_value:.3g}, V={effect:.2f}); "
                        f"{top_category} at {top_share:.0%}"
                    )
            else:
                message += " (sample too small for chi-square)"

    if top_share > DOMINANCE_THRESHOLD and len(proportions) > 1:
        severity = "WARN"
        message = f"{top_category} dominates {column} at {top_share:.0%} of the cohort"

    return BiasFinding(
        attribute=column,
        kind="representation",
        severity=severity,
        message=message,
        statistic=float(statistic) if statistic is not None else None,
        p_value=float(p_value) if p_value is not None else None,
        effect_size=effect,
        distribution=counts.to_dict(),
    )


def audit_outcome_disparity(
    frame: pd.DataFrame, group_column: str, outcome_column: str
) -> BiasFinding:
    """Test whether a continuous outcome differs across subgroups.

    Uses Kruskal-Wallis (non-parametric): quality scores and tumour burdens are
    skewed and bounded, so a t-test's normality assumption does not hold.

    The reported effect is the **disparity ratio** (max subgroup median / min),
    which is what a governance reviewer can act on directly.
    """
    from scipy import stats

    if group_column not in frame or outcome_column not in frame:
        return BiasFinding(group_column, "disparity", "INFO", "column unavailable")

    groups = [
        group[outcome_column].dropna().to_numpy(dtype=float)
        for _, group in frame.groupby(frame[group_column].fillna("unknown"))
    ]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return BiasFinding(
            group_column, "disparity", "INFO", f"too few groups with data for {outcome_column}"
        )

    medians = [float(np.median(g)) for g in groups]
    lo, hi = min(medians), max(medians)
    ratio = hi / lo if lo > 0 else float("inf")

    statistic = p_value = None
    # kruskal raises ValueError when every value is identical -- that is not an
    # error, it just means there is no variance to test.
    with contextlib.suppress(ValueError):
        statistic, p_value = stats.kruskal(*groups)

    severity = "INFO"
    message = (
        f"{outcome_column} medians across {group_column} span {lo:.1f}-{hi:.1f} "
        f"(ratio {ratio:.2f}x)"
    )
    if ratio > DISPARITY_THRESHOLD or (p_value is not None and p_value < 0.05):
        severity = "WARN"
        message = (
            f"{outcome_column} differs across {group_column}: medians {lo:.1f}-{hi:.1f} "
            f"(ratio {ratio:.2f}x"
            + (f", Kruskal-Wallis p={p_value:.3g}" if p_value is not None else "")
            + ")"
        )

    return BiasFinding(
        attribute=group_column,
        kind="disparity",
        severity=severity,
        message=message,
        statistic=float(statistic) if statistic is not None else None,
        p_value=float(p_value) if p_value is not None else None,
        effect_size=float(ratio) if np.isfinite(ratio) else None,
    )


def audit_cohort(frame: pd.DataFrame, outcome_columns: tuple[str, ...] = ()) -> list[BiasFinding]:
    """Run the full audit: representation for each attribute, disparity for each outcome.

    Args:
        frame: Cohort table. An ``age`` column is auto-banded.
        outcome_columns: Continuous outcomes to test across subgroups, e.g.
            ``("quality_score", "tumor_volume_mm3")``.

    Returns:
        All findings, representation first.
    """
    frame = frame.copy()
    if "age" in frame:
        frame["age_band"] = frame["age"].apply(age_band)

    findings: list[BiasFinding] = []
    for column in ("sex", "age_band", "institution", "scanner", "contrast"):
        if column in frame and frame[column].notna().any():
            findings.append(audit_representation(frame, column))

    for outcome in outcome_columns:
        for group in ("institution", "sex", "scanner", "age_band"):
            if group in frame and outcome in frame:
                findings.append(audit_outcome_disparity(frame, group, outcome))

    n_warn = sum(f.severity == "WARN" for f in findings)
    log.info(
        "bias.audit_complete",
        extra={
            "n_findings": len(findings),
            "n_warnings": n_warn,
        },
    )
    return findings


def plot_bias_figures(frame: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Export the distribution figures the dataset card references."""
    out_dir = Path(out_dir)
    written: list[Path] = []
    frame = frame.copy()
    if "age" in frame:
        frame["age_band"] = frame["age"].apply(age_band)

    for column, title in (
        ("sex", "Sex distribution"),
        ("age_band", "Age band distribution"),
        ("institution", "Studies per institution"),
        ("scanner", "Studies per scanner"),
    ):
        if column in frame and frame[column].notna().any():
            written.append(
                plot_categorical(
                    frame[column].fillna("unknown").value_counts().to_dict(),
                    out_dir / f"bias_{column}.png",
                    title,
                    column,
                )
            )

    for column, title, xlabel in (
        ("age", "Age distribution", "age (years)"),
        ("quality_score", "Quality score distribution", "score (0-100)"),
        ("tumor_volume_mm3", "Tumour burden distribution", "volume (mm3)"),
        ("slice_thickness_mm", "Slice thickness distribution", "mm"),
    ):
        if column in frame and frame[column].notna().any():
            written.append(
                plot_distribution(
                    frame[column].dropna().tolist(),
                    out_dir / f"dist_{column}.png",
                    title,
                    xlabel,
                )
            )

    log.info("bias.figures", extra={"n_figures": len(written)})
    return written


def findings_to_frame(findings: list[BiasFinding]) -> pd.DataFrame:
    """Convert findings to a DataFrame with stable columns."""
    columns = ["attribute", "kind", "severity", "message", "statistic", "p_value", "effect_size"]
    if not findings:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame([f.to_dict() for f in findings])[columns]
