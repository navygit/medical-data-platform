"""Leakage-safe train/val/test partitioning.

The single most common defect in medical ML papers is a split done at the *scan*
level when the unit of independence is the *patient*. A patient with four
modalities and two timepoints contributes eight rows; a random row-wise split
puts some in train and some in test, the model memorises that patient's anatomy,
and the reported Dice score is fiction.

So splitting here is **grouped by patient by default** and the invariant is
asserted, not assumed -- :func:`verify_no_leakage` runs on every pipeline
execution and raises rather than warns.

Stratification is applied on top of grouping where possible, to keep tumour-burden
strata balanced across splits when the cohort is small enough for chance to
matter.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import pandas as pd

from common.config import Config
from common.logging import get_logger
from common.metadata import ScanRecord

log = get_logger(__name__)


class LeakageError(RuntimeError):
    """Raised when a group appears in more than one split."""


def _subject_strata(records: Sequence[ScanRecord], column: str | None) -> dict[str, str]:
    """Map each subject to a stratum label.

    ``tumor_volume_mm3`` is special-cased into tertiles, because stratifying on a
    continuous value is meaningless; every other column is used verbatim.
    """
    if not column:
        return {r.patient_id: "all" for r in records}

    by_subject: dict[str, list[float | str]] = defaultdict(list)
    for rec in records:
        value = getattr(rec, column, None) or rec.extra.get(column)
        if value is not None:
            by_subject[rec.patient_id].append(value)

    if column == "tumor_volume_mm3":
        volumes = {s: float(np.mean([float(v) for v in vs])) for s, vs in by_subject.items() if vs}
        if len(set(volumes.values())) < 3:
            return {r.patient_id: "all" for r in records}
        edges = np.quantile(list(volumes.values()), [1 / 3, 2 / 3])
        return {
            s: ("low" if v <= edges[0] else "mid" if v <= edges[1] else "high")
            for s, v in volumes.items()
        }

    return {s: str(vs[0]) if vs else "unknown" for s, vs in by_subject.items()}


def _allocate(subjects: list[str], cfg: Config, rng: np.random.Generator) -> dict[str, list[str]]:
    """Shuffle and cut one stratum into train/val/test by configured ratios."""
    shuffled = list(subjects)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * cfg.split.train)
    n_val = round(n * cfg.split.val)

    # With tiny strata, rounding can allocate everything to train and leave val
    # and test empty. Guarantee at least one subject each once n allows it.
    if n >= 3:
        n_train = min(max(n_train, 1), n - 2)
        n_val = min(max(n_val, 1), n - n_train - 1)

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def make_splits(records: Sequence[ScanRecord], cfg: Config) -> dict[str, list[str]]:
    """Partition subjects into train/val/test.

    Args:
        records: Records to partition (typically QC-passing only).
        cfg: Supplies ratios, seed, ``group_by`` and ``stratify_by``.

    Returns:
        Mapping of split name to sorted subject IDs.

    Raises:
        LeakageError: If the result would place a subject in two splits.
    """
    rng = np.random.default_rng(cfg.split.seed)
    subjects = sorted({r.patient_id for r in records})

    if not subjects:
        log.warning("split.no_subjects")
        return {"train": [], "val": [], "test": []}

    strata = _subject_strata(records, cfg.split.stratify_by)
    grouped: dict[str, list[str]] = defaultdict(list)
    for subject in subjects:
        grouped[strata.get(subject, "all")].append(subject)

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for stratum, members in sorted(grouped.items()):
        allocated = _allocate(members, cfg, rng)
        for name, chunk in allocated.items():
            splits[name].extend(chunk)
        log.debug(
            "split.stratum",
            extra={
                "stratum": stratum,
                "n": len(members),
                **{k: len(v) for k, v in allocated.items()},
            },
        )

    splits = {k: sorted(v) for k, v in splits.items()}
    verify_no_leakage(splits)

    log.info(
        "split.complete",
        extra={
            "n_subjects": len(subjects),
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
            "stratify_by": cfg.split.stratify_by,
            "seed": cfg.split.seed,
        },
    )
    return splits


def verify_no_leakage(splits: dict[str, list[str]]) -> None:
    """Assert that no subject appears in more than one split.

    Raises:
        LeakageError: On any overlap, naming the offending subjects.
    """
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = set(splits[a]) & set(splits[b])
            if overlap:
                raise LeakageError(
                    f"{len(overlap)} subject(s) in both '{a}' and '{b}': " f"{sorted(overlap)[:5]}"
                )


def assign_splits(records: Sequence[ScanRecord], splits: dict[str, list[str]]) -> list[ScanRecord]:
    """Stamp each record with its split name in ``extra['split']``."""
    lookup = {subject: name for name, members in splits.items() for subject in members}
    return [
        r.model_copy(update={"extra": {**r.extra, "split": lookup.get(r.patient_id, "unassigned")}})
        for r in records
    ]


def split_summary(records: Sequence[ScanRecord], splits: dict[str, list[str]]) -> pd.DataFrame:
    """Per-split subject/series counts and mean tumour burden.

    Written into the dataset card so a reader can see the splits are balanced
    rather than taking it on faith.
    """
    rows = []
    for name, members in splits.items():
        subset = [r for r in records if r.patient_id in set(members)]
        volumes = [r.tumor_volume_mm3 for r in subset if r.tumor_volume_mm3 is not None]
        rows.append(
            {
                "split": name,
                "n_subjects": len(members),
                "n_series": len(subset),
                "mean_tumor_volume_mm3": round(float(np.mean(volumes)), 1) if volumes else 0.0,
                "modalities": ",".join(sorted({r.modality for r in subset})),
            }
        )
    return pd.DataFrame(rows)
