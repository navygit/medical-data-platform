"""Shared framework for the medical data platform.

Everything in ``common`` is dataset-agnostic. A pipeline under ``pipelines/``
composes these pieces; it never re-implements them. That is what keeps the three
case studies consistent rather than three separate codebases in one repo.

Modules:
    config: Typed, layered YAML configuration.
    logging: Structured JSON/human logging.
    metadata: Scan records and manifest I/O.
    qc: Pluggable quality-control check registry and reporting.
    storage: Filesystem access and content hashing.
    versioning: Immutable, content-addressed dataset releases and lineage.
    visualization: Headless PNG figure export.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "config",
    "logging",
    "metadata",
    "qc",
    "storage",
    "versioning",
    "visualization",
]
