"""Quality-control engine.

QC is expressed as a registry of small, independent checks. Each check receives a
:class:`~common.metadata.ScanRecord` plus the run config and returns zero or more
:class:`QCFinding` objects. Adding a new rule means writing one function and
decorating it -- no edits to the engine.

The separation matters operationally: clinical reviewers propose rules, and a rule
that lives in its own function with its own docstring can be reviewed by someone
who does not read the whole pipeline.

Severity ladder:
    ``ERROR`` -- exclude the series from training releases.
    ``WARN``  -- keep, but surface in the report and the dataset card.
    ``INFO``  -- informational only.

Example:
    >>> findings = run_checks(records, cfg)
    >>> report = summarise(findings, records)
    >>> write_html_report(report, findings, Path("outputs/QC_REPORT.html"))
"""

from __future__ import annotations

import html
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from common.config import Config
from common.logging import get_logger
from common.metadata import ScanRecord

log = get_logger(__name__)

Severity = Literal["ERROR", "WARN", "INFO"]


@dataclass(frozen=True)
class QCFinding:
    """A single QC observation about one series."""

    patient_id: str
    modality: str
    check: str
    severity: Severity
    message: str
    filepath: str = ""


CheckFn = Callable[[ScanRecord, Config], Iterable[QCFinding]]
_REGISTRY: dict[str, CheckFn] = {}


def resolve_path(rec: ScanRecord, cfg: Config) -> Path:
    """Resolve a record's stored path against the configured data roots.

    ``ScanRecord.filepath`` is stored *relative* to its data root so manifests
    stay portable between the workstation where data is curated and the cluster
    where models train. Checks therefore cannot use the path verbatim -- they
    must resolve it, and QC may legitimately run over either raw records
    (pre-preprocess) or processed ones.

    Tries, in order: the path as given, then the raw root, then the processed
    root. Falls back to the raw root so failure messages name a canonical
    location rather than whatever the CWD happened to be.
    """
    candidate = Path(rec.filepath)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    for root in (cfg.paths.raw, cfg.paths.processed):
        resolved = Path(root) / rec.filepath
        if resolved.exists():
            return resolved
    return Path(cfg.paths.raw) / rec.filepath


def check(name: str) -> Callable[[CheckFn], CheckFn]:
    """Register a QC check under ``name``."""

    def decorator(fn: CheckFn) -> CheckFn:
        if name in _REGISTRY:
            raise ValueError(f"duplicate QC check name: {name!r}")
        _REGISTRY[name] = fn
        return fn

    return decorator


def registered_checks() -> list[str]:
    """Names of every registered check, for docs and the dataset card."""
    return sorted(_REGISTRY)


# --------------------------------------------------------------------------- #
# Checks                                                                       #
# --------------------------------------------------------------------------- #


@check("file_readable")
def _file_readable(rec: ScanRecord, cfg: Config) -> Iterable[QCFinding]:
    """The file exists on disk and is not zero-length.

    Catches truncated transfers, which are the single most common failure when
    pulling multi-GB studies over a hospital VPN.
    """
    path = resolve_path(rec, cfg)
    if not path.exists():
        yield QCFinding(rec.patient_id, rec.modality, "file_readable", "ERROR",
                        f"file missing: {rec.filepath}", rec.filepath)
    elif path.stat().st_size == 0:
        yield QCFinding(rec.patient_id, rec.modality, "file_readable", "ERROR",
                        "file is zero bytes", rec.filepath)


@check("corrupt_volume")
def _corrupt_volume(rec: ScanRecord, cfg: Config) -> Iterable[QCFinding]:
    """Intensity statistics were parseable and finite.

    ``ingest`` records NaN/None stats when a volume fails to decode, so a null
    here is a proxy for "the header parsed but the pixel data did not".
    """
    stats = (rec.intensity_min, rec.intensity_max, rec.intensity_mean)
    if any(s is None for s in stats):
        yield QCFinding(rec.patient_id, rec.modality, "corrupt_volume", "ERROR",
                        "intensity statistics unavailable (unreadable pixel data)",
                        rec.filepath)
        return
    if any(not np.isfinite(s) for s in stats):  # type: ignore[arg-type]
        yield QCFinding(rec.patient_id, rec.modality, "corrupt_volume", "ERROR",
                        "non-finite intensity values (NaN/Inf in volume)", rec.filepath)


@check("empty_volume")
def _empty_volume(rec: ScanRecord, cfg: Config) -> Iterable[QCFinding]:
    """The volume carries signal rather than a constant value."""
    if rec.intensity_std is not None and float(rec.intensity_std) == 0.0:
        yield QCFinding(rec.patient_id, rec.modality, "empty_volume", "ERROR",
                        "volume is constant (zero variance)", rec.filepath)


@check("dimensions")
def _dimensions(rec: ScanRecord, cfg: Config) -> Iterable[QCFinding]:
    """The volume is 3D and at least ``qc.min_shape`` in every axis."""
    if not rec.shape:
        yield QCFinding(rec.patient_id, rec.modality, "dimensions", "ERROR",
                        "shape unavailable", rec.filepath)
        return
    if len(rec.shape) != 3:
        yield QCFinding(rec.patient_id, rec.modality, "dimensions", "ERROR",
                        f"expected 3D volume, got {len(rec.shape)}D {rec.shape}",
                        rec.filepath)
        return
    minimum = cfg.qc.min_shape
    if any(actual < want for actual, want in zip(rec.shape, minimum, strict=False)):
        yield QCFinding(rec.patient_id, rec.modality, "dimensions", "ERROR",
                        f"shape {rec.shape} below minimum {minimum}", rec.filepath)


@check("orientation")
def _orientation(rec: ScanRecord, cfg: Config) -> Iterable[QCFinding]:
    """Anatomical orientation matches the cohort standard.

    A WARN rather than an ERROR: ``preprocess`` reorients deterministically, so a
    non-standard orientation is a fact to record, not a reason to drop a study.
    """
    if rec.orientation and rec.orientation.upper() != cfg.qc.expected_orientation:
        yield QCFinding(rec.patient_id, rec.modality, "orientation", "WARN",
                        f"orientation {rec.orientation} != expected "
                        f"{cfg.qc.expected_orientation} (will be reoriented)",
                        rec.filepath)


@check("spacing")
def _spacing(rec: ScanRecord, cfg: Config) -> Iterable[QCFinding]:
    """Voxel spacing is present, positive and within the configured ceiling."""
    if not rec.voxel_spacing:
        yield QCFinding(rec.patient_id, rec.modality, "spacing", "WARN",
                        "voxel spacing unavailable", rec.filepath)
        return
    if any(s <= 0 for s in rec.voxel_spacing):
        yield QCFinding(rec.patient_id, rec.modality, "spacing", "ERROR",
                        f"non-positive voxel spacing {rec.voxel_spacing}", rec.filepath)
        return
    if any(s > cfg.qc.max_spacing_mm for s in rec.voxel_spacing):
        yield QCFinding(rec.patient_id, rec.modality, "spacing", "WARN",
                        f"coarse spacing {[round(s, 2) for s in rec.voxel_spacing]} mm "
                        f"exceeds {cfg.qc.max_spacing_mm} mm", rec.filepath)


@check("missing_mask")
def _missing_mask(rec: ScanRecord, cfg: Config) -> Iterable[QCFinding]:
    """A segmentation mask exists for the series."""
    if not rec.mask_available:
        yield QCFinding(rec.patient_id, rec.modality, "missing_mask", "WARN",
                        "no segmentation mask for this subject", rec.filepath)


@check("empty_mask")
def _empty_mask(rec: ScanRecord, cfg: Config) -> Iterable[QCFinding]:
    """The mask, when present, contains at least ``qc.min_tumor_voxels`` labelled voxels.

    An all-background mask trains the model to predict nothing and silently drags
    the Dice score down; it is worth failing loudly on.
    """
    if not rec.mask_available or rec.tumor_volume_mm3 is None:
        return
    voxel_volume = rec.voxel_volume_mm3
    if voxel_volume <= 0:
        return
    n_voxels = rec.tumor_volume_mm3 / voxel_volume
    if n_voxels < cfg.qc.min_tumor_voxels:
        yield QCFinding(rec.patient_id, rec.modality, "empty_mask", "ERROR",
                        f"mask has ~{n_voxels:.0f} labelled voxels, below minimum "
                        f"{cfg.qc.min_tumor_voxels}", rec.filepath)


def run_checks(
    records: Sequence[ScanRecord],
    cfg: Config,
    only: Sequence[str] | None = None,
) -> list[QCFinding]:
    """Run every registered check over every record.

    A check that raises is itself reported as an ERROR finding rather than
    aborting the run: one malformed study must not stop QC of the other 4,999.

    Args:
        records: Series to inspect.
        cfg: Run config supplying the thresholds.
        only: Optional subset of check names.

    Returns:
        All findings, in registry order.
    """
    names = list(only) if only else registered_checks()
    findings: list[QCFinding] = []

    for name in names:
        fn = _REGISTRY[name]
        for rec in records:
            try:
                findings.extend(fn(rec, cfg))
            except Exception as exc:
                log.exception("qc.check_failed", extra={"check": name, "patient": rec.patient_id})
                findings.append(
                    QCFinding(rec.patient_id, rec.modality, name, "ERROR",
                              f"check raised {type(exc).__name__}: {exc}", rec.filepath)
                )

    log.info("qc.complete", extra={
        "n_records": len(records),
        "n_checks": len(names),
        "n_findings": len(findings),
        "n_errors": sum(f.severity == "ERROR" for f in findings),
    })
    return findings


# --------------------------------------------------------------------------- #
# Cohort-level checks                                                          #
# --------------------------------------------------------------------------- #


def check_duplicates(records: Sequence[ScanRecord]) -> list[QCFinding]:
    """Flag content-identical files and repeated (patient, modality) pairs.

    Duplicate content across *different* patient IDs is the dangerous case: it
    leaks the same volume into train and test and inflates every metric.

    Policy: keep the first patient (in sorted order) and reject the later copies.
    Failing every copy would discard the underlying study entirely, which is
    over-correction -- the data is fine, the *duplication* is the defect. Sorting
    first makes the choice of survivor deterministic across runs.
    """
    findings: list[QCFinding] = []

    by_hash: dict[str, list[ScanRecord]] = defaultdict(list)
    for rec in records:
        if rec.sha256:
            by_hash[rec.sha256].append(rec)

    for digest, group in by_hash.items():
        if len(group) < 2:
            continue
        patients = sorted({r.patient_id for r in group})
        if len(patients) > 1:
            keep = patients[0]
            for rec in group:
                is_copy = rec.patient_id != keep
                findings.append(QCFinding(
                    rec.patient_id, rec.modality, "duplicate_content",
                    "ERROR" if is_copy else "INFO",
                    f"identical content (sha256 {digest[:12]}) shared by {patients}; "
                    + (f"rejected as copy of {keep}" if is_copy else "retained as canonical"),
                    rec.filepath,
                ))
        else:
            for rec in group:
                findings.append(QCFinding(
                    rec.patient_id, rec.modality, "duplicate_content", "WARN",
                    f"file repeated within subject (sha256 {digest[:12]})", rec.filepath,
                ))

    seen: dict[tuple[str, str], int] = defaultdict(int)
    for rec in records:
        key = (rec.patient_id, rec.modality)
        seen[key] += 1
        if seen[key] > 1:
            findings.append(QCFinding(
                rec.patient_id, rec.modality, "duplicate_series", "WARN",
                f"{seen[key]} series share patient/modality {key}", rec.filepath,
            ))
    return findings


def check_missing_modalities(records: Sequence[ScanRecord], cfg: Config) -> list[QCFinding]:
    """Flag subjects missing any modality the cohort schema requires."""
    expected = {m.lower() for m in cfg.qc.expected_modalities}
    if not expected:
        return []

    have: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        have[rec.patient_id].add(rec.modality)

    findings: list[QCFinding] = []
    for patient, modalities in sorted(have.items()):
        missing = expected - modalities
        if missing:
            findings.append(QCFinding(
                patient, ",".join(sorted(missing)), "missing_modalities", "ERROR",
                f"subject missing required modalities: {sorted(missing)}", "",
            ))
    return findings


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #


@dataclass
class QCReport:
    """Aggregate view of a QC run."""

    n_records: int
    n_subjects: int
    n_findings: int
    n_errors: int
    n_warnings: int
    failed_subjects: list[str] = field(default_factory=list)
    by_check: dict[str, int] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        """Fraction of subjects with no ERROR finding."""
        if self.n_subjects == 0:
            return 0.0
        return 1.0 - len(self.failed_subjects) / self.n_subjects


def summarise(findings: Sequence[QCFinding], records: Sequence[ScanRecord]) -> QCReport:
    """Roll findings up into a :class:`QCReport`."""
    by_check: dict[str, int] = defaultdict(int)
    for f in findings:
        by_check[f.check] += 1

    return QCReport(
        n_records=len(records),
        n_subjects=len({r.patient_id for r in records}),
        n_findings=len(findings),
        n_errors=sum(f.severity == "ERROR" for f in findings),
        n_warnings=sum(f.severity == "WARN" for f in findings),
        failed_subjects=sorted({f.patient_id for f in findings if f.severity == "ERROR"}),
        by_check=dict(by_check),
    )


def findings_to_frame(findings: Sequence[QCFinding]) -> pd.DataFrame:
    """Convert findings to a DataFrame with stable column order."""
    columns = ["patient_id", "modality", "check", "severity", "message", "filepath"]
    if not findings:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame([f.__dict__ for f in findings])[columns]


def write_csv_report(findings: Sequence[QCFinding], path: Path) -> Path:
    """Write findings to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    findings_to_frame(findings).to_csv(path, index=False)
    return path


_HTML_TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>QC Report - {name}</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:2rem auto;max-width:1100px;color:#1a1a1a}}
 h1{{margin-bottom:.2rem}} .sub{{color:#666;margin-bottom:1.5rem}}
 .cards{{display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:2rem}}
 .card{{flex:1;min-width:140px;border:1px solid #e3e3e3;border-radius:10px;padding:1rem}}
 .card .v{{font-size:2rem;font-weight:600}} .card .l{{color:#666;font-size:.8rem;text-transform:uppercase}}
 .err .v{{color:#c0392b}} .warn .v{{color:#d68910}} .ok .v{{color:#1e8449}}
 table{{border-collapse:collapse;width:100%;font-size:.88rem}}
 th,td{{border-bottom:1px solid #eee;padding:.5rem;text-align:left;vertical-align:top}}
 th{{background:#fafafa;position:sticky;top:0}}
 .sev-ERROR{{color:#c0392b;font-weight:600}} .sev-WARN{{color:#d68910;font-weight:600}}
 .sev-INFO{{color:#666}} code{{background:#f5f5f5;padding:.1rem .3rem;border-radius:3px}}
 .empty{{padding:2rem;text-align:center;color:#1e8449;border:1px dashed #cfe8d6;border-radius:10px}}
</style>
<h1>Quality Control Report</h1>
<div class="sub">Dataset <code>{name}</code> &middot; generated {ts}</div>
<div class="cards">
  <div class="card"><div class="l">Series</div><div class="v">{n_records}</div></div>
  <div class="card"><div class="l">Subjects</div><div class="v">{n_subjects}</div></div>
  <div class="card err"><div class="l">Errors</div><div class="v">{n_errors}</div></div>
  <div class="card warn"><div class="l">Warnings</div><div class="v">{n_warnings}</div></div>
  <div class="card ok"><div class="l">Pass rate</div><div class="v">{pass_rate:.0%}</div></div>
</div>
<h2>Findings by check</h2>
{by_check_table}
<h2>All findings</h2>
{findings_table}
"""


def _table(frame: pd.DataFrame, severity_column: str | None = None) -> str:
    """Render a DataFrame as an escaped HTML table."""
    if frame.empty:
        return '<div class="empty">No findings &mdash; all checks passed.</div>'
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in frame.columns)
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for col in frame.columns:
            value = html.escape(str(row[col]))
            css = f' class="sev-{value}"' if col == severity_column else ""
            cells.append(f"<td{css}>{value}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def write_html_report(
    report: QCReport,
    findings: Sequence[QCFinding],
    path: Path,
    name: str = "dataset",
) -> Path:
    """Render a standalone HTML QC report.

    Self-contained by design -- no CDN links, so it can be opened from a
    locked-down clinical workstation or attached to an email.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    by_check = pd.DataFrame(
        sorted(report.by_check.items(), key=lambda kv: -kv[1]), columns=["check", "count"]
    )
    html_text = _HTML_TEMPLATE.format(
        name=html.escape(name),
        ts=pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        n_records=report.n_records,
        n_subjects=report.n_subjects,
        n_errors=report.n_errors,
        n_warnings=report.n_warnings,
        pass_rate=report.pass_rate,
        by_check_table=_table(by_check),
        findings_table=_table(findings_to_frame(findings), severity_column="severity"),
    )
    path.write_text(html_text, encoding="utf-8")
    log.info("qc.report_written", extra={"path": str(path)})
    return path


def apply_findings(
    records: Sequence[ScanRecord], findings: Sequence[QCFinding]
) -> list[ScanRecord]:
    """Return copies of ``records`` with ``qc_status``/``qc_flags`` populated.

    Status is ``fail`` if any ERROR touched the subject, ``warn`` if only
    warnings, else ``pass``. Findings are matched at *subject* level for cohort
    checks (which carry a synthetic modality) and at series level otherwise.
    """
    by_subject: dict[str, list[QCFinding]] = defaultdict(list)
    for f in findings:
        by_subject[f.patient_id].append(f)

    updated: list[ScanRecord] = []
    for rec in records:
        relevant = [
            f for f in by_subject.get(rec.patient_id, [])
            if f.modality in (rec.modality, "") or f.check.startswith("missing_modalities")
        ]
        severities = {f.severity for f in relevant}
        status = "fail" if "ERROR" in severities else "warn" if "WARN" in severities else "pass"
        updated.append(
            rec.model_copy(update={
                "qc_status": status,
                "qc_flags": sorted({f.check for f in relevant}),
            })
        )
    return updated
