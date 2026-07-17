"""Radiology report parsing: sections, findings, negation, severity, measurements.

Why this is rule-based
----------------------
The obvious approach is to search the report for "pneumonia" and set a label.
That approach is wrong roughly a third of the time, because radiology prose is
dominated by *negation* and *uncertainty*:

    "No evidence of pneumonia."          -> keyword match, label must be 0
    "Pneumonia cannot be excluded."      -> keyword match, label is uncertain
    "No pleural effusion or pneumothorax." -> one 'no' negates two findings

This module implements a NegEx/NegBio-style algorithm: locate each finding
mention, then scan a bounded window for negation and uncertainty triggers,
respecting sentence boundaries and termination phrases ("but", "however").

It is deliberately dependency-free. spaCy and BioClinicalBERT are the natural
upgrades and the interface here is designed to be swapped for them
(:func:`extract_labels` returns the same shape either way), but a regex engine
that runs anywhere beats a 2 GB model that CI cannot install. The accuracy of
this approach on CheXpert-style labelling is well established -- the CheXpert
labeller itself is rule-based.

Label semantics follow CheXpert:
    ``1`` positive, ``0`` negative, ``-1`` uncertain, ``None`` not mentioned.

Example:
    >>> report = parse_report("FINDINGS: No pleural effusion.\\nIMPRESSION: Normal.")
    >>> report.labels["pleural_effusion"]
    0
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from common.logging import get_logger

log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Ontology                                                                     #
# --------------------------------------------------------------------------- #

# Finding -> surface forms. Kept here rather than in code so a clinician can
# review and extend the vocabulary without reading the algorithm.
FINDING_VOCAB: dict[str, list[str]] = {
    "pneumonia": [
        "pneumonia", "infectious process", "airspace opacity", "airspace disease",
        "consolidation", "infiltrate",
    ],
    "edema": [
        "edema", "pulmonary edema", "vascular congestion", "kerley", "interstitial edema",
    ],
    "cardiomegaly": [
        "cardiomegaly", "enlarged cardiac silhouette", "enlargement of the cardiac silhouette",
        "cardiac enlargement", "enlarged heart",
    ],
    "pleural_effusion": [
        "pleural effusion", "effusion", "pleural fluid",
    ],
    "atelectasis": [
        "atelectasis", "atelectatic", "volume loss",
    ],
    "pneumothorax": [
        "pneumothorax",
    ],
}

NEGATION_TRIGGERS: tuple[str, ...] = (
    "no evidence of", "no evidence for", "without evidence of", "no signs of",
    "no findings of", "free of", "resolved", "clear of", "rules out",
    "ruled out", "negative for", "absence of", "without", "no ", "not ",
)

UNCERTAIN_TRIGGERS: tuple[str, ...] = (
    "possible", "possibly", "may represent", "cannot be excluded", "can not be excluded",
    "cannot exclude", "suspicious for", "questionable", "concerning for", "suggestive of",
    "compatible with", "consistent with", "likely", "probable", "differential",
    "worrisome for", "versus", "vs.",
)

# Phrases that end a negation's scope: "no effusion, but consolidation is present"
TERMINATION: tuple[str, ...] = (
    "but", "however", "although", "though", "otherwise", "except", "aside from",
    "which", "yet",
)

SEVERITY_TERMS: dict[str, int] = {
    "trace": 1, "minimal": 1, "tiny": 1, "small": 2, "mild": 2, "slight": 2,
    "moderate": 3, "substantial": 4, "large": 4, "severe": 4, "extensive": 4,
    "massive": 5, "marked": 4,
}

RECOMMENDATION_CUES: tuple[str, ...] = (
    "recommend", "suggest", "advise", "consider", "follow-up", "follow up",
    "correlate", "further evaluation", "attention",
)

# Section headers as they appear in MIMIC-CXR reports.
_SECTION_RE = re.compile(
    r"^\s*(EXAMINATION|INDICATION|HISTORY|TECHNIQUE|COMPARISON|FINDINGS|IMPRESSION|"
    r"CONCLUSION|NOTIFICATION|WET READ)\s*:",
    re.IGNORECASE | re.MULTILINE,
)

# "3.2 cm", "16.1cm", "4 mm"
_MEASUREMENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(cm|mm)\b", re.IGNORECASE)

# Window (characters) before a mention searched for a negation trigger. 45 is the
# NegEx convention; long enough for "no evidence of", short enough that a
# negation two clauses back does not bleed forward.
_SCOPE_CHARS = 45


@dataclass
class ParsedReport:
    """Structured view of one radiology report."""

    study_id: str
    sections: dict[str, str] = field(default_factory=dict)
    labels: dict[str, int | None] = field(default_factory=dict)
    severity: dict[str, int] = field(default_factory=dict)
    measurements_cm: list[float] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    n_chars: int = 0

    @property
    def findings_text(self) -> str:
        """The FINDINGS section, or empty string."""
        return self.sections.get("findings", "")

    @property
    def impression_text(self) -> str:
        """The IMPRESSION/CONCLUSION section, or empty string."""
        return self.sections.get("impression", self.sections.get("conclusion", ""))

    @property
    def positive_findings(self) -> list[str]:
        """Findings labelled positive."""
        return sorted(k for k, v in self.labels.items() if v == 1)


def split_sections(text: str) -> dict[str, str]:
    """Split a report into lowercase-keyed sections.

    Text before the first recognised header is stored under ``preamble`` rather
    than dropped -- some MIMIC reports have no headers at all, and silently
    losing their content would zero out the labels.
    """
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return {"preamble": text.strip()}

    sections: dict[str, str] = {}
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections["preamble"] = preamble

    for i, match in enumerate(matches):
        name = match.group(1).lower()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[name] = text[match.end() : end].strip()
    return sections


def _scope_before(text: str, start: int) -> str:
    """Return the text preceding ``start``, clipped to the negation scope.

    Clipped at sentence boundaries and termination words, so a negation in a
    previous sentence cannot capture this mention.
    """
    window = text[max(0, start - _SCOPE_CHARS) : start]
    for boundary in (".", ";", ":"):
        if boundary in window:
            window = window.rsplit(boundary, 1)[1]
    for term in TERMINATION:
        pattern = rf"\b{re.escape(term)}\b"
        found = list(re.finditer(pattern, window))
        if found:
            window = window[found[-1].end() :]
    return window


def classify_mention(text: str, start: int, end: int) -> int:
    """Classify a single finding mention as positive, negative or uncertain.

    Args:
        text: Lowercased report text.
        start: Mention start offset.
        end: Mention end offset.

    Returns:
        ``1`` positive, ``0`` negative, ``-1`` uncertain.
    """
    before = _scope_before(text, start)
    after = text[end : end + 30]

    # Uncertainty wins over negation: "cannot be excluded" contains "not".
    if any(trigger in before for trigger in UNCERTAIN_TRIGGERS):
        return -1
    if any(trigger in after for trigger in UNCERTAIN_TRIGGERS):
        return -1
    if any(trigger in before for trigger in NEGATION_TRIGGERS):
        return 0
    return 1


def extract_labels(text: str) -> tuple[dict[str, int | None], dict[str, int]]:
    """Extract CheXpert-style labels and severity grades from report text.

    When a finding is mentioned more than once, the most clinically significant
    mention wins (positive > uncertain > negative). A report saying "no effusion"
    in FINDINGS and "small effusion" in IMPRESSION describes a patient who has an
    effusion.

    Returns:
        ``(labels, severity)`` where labels values are 1/0/-1/None and severity
        maps a positive finding to a 1-5 grade.
    """
    lowered = text.lower()
    labels: dict[str, int | None] = dict.fromkeys(FINDING_VOCAB)
    severity: dict[str, int] = {}
    priority = {1: 3, -1: 2, 0: 1}

    for finding, surface_forms in FINDING_VOCAB.items():
        for form in surface_forms:
            for match in re.finditer(rf"\b{re.escape(form)}\b", lowered):
                verdict = classify_mention(lowered, match.start(), match.end())
                current = labels[finding]
                if current is None or priority[verdict] > priority[current]:
                    labels[finding] = verdict

                if verdict == 1:
                    context = lowered[max(0, match.start() - 40) : match.start()]
                    for term, grade in SEVERITY_TERMS.items():
                        if re.search(rf"\b{term}\b", context):
                            severity[finding] = max(severity.get(finding, 0), grade)

    return labels, severity


def extract_measurements(text: str) -> list[float]:
    """Extract all size measurements, normalised to centimetres."""
    out: list[float] = []
    for value, unit in _MEASUREMENT_RE.findall(text):
        size = float(value)
        out.append(size / 10.0 if unit.lower() == "mm" else size)
    return out


def extract_recommendations(text: str) -> list[str]:
    """Extract sentences containing a follow-up or recommendation cue.

    These drive the agentic-AI use case: an agent that reads a report and decides
    whether the patient needs another scan is acting on exactly these sentences.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [
        s.strip()
        for s in sentences
        if any(cue in s.lower() for cue in RECOMMENDATION_CUES) and len(s.strip()) > 10
    ]


def parse_report(text: str, study_id: str = "") -> ParsedReport:
    """Parse a full radiology report into a :class:`ParsedReport`.

    Labels are extracted from FINDINGS + IMPRESSION only. Deliberately excluding
    INDICATION/HISTORY prevents the classic leak: "INDICATION: history of
    pneumonia" describes the referral question, not a finding on this film, and
    training on it teaches the model to read the clinician's suspicion.
    """
    sections = split_sections(text)
    label_source = " ".join([
        sections.get("findings", ""),
        sections.get("impression", ""),
        sections.get("conclusion", ""),
    ]).strip() or sections.get("preamble", text)

    labels, severity = extract_labels(label_source)

    return ParsedReport(
        study_id=study_id,
        sections=sections,
        labels=labels,
        severity=severity,
        measurements_cm=extract_measurements(label_source),
        recommendations=extract_recommendations(label_source),
        n_chars=len(text),
    )


def parse_report_file(path: Path) -> ParsedReport:
    """Parse a report from disk, using the filename stem as the study ID."""
    path = Path(path)
    return parse_report(path.read_text(encoding="utf-8", errors="replace"), study_id=path.stem)
