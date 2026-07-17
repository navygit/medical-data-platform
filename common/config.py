"""Typed, layered configuration for pipeline runs.

Configuration is YAML on disk and Pydantic in memory. Every pipeline entrypoint
resolves a config the same way::

    base.yaml  <-  <pipeline>.yaml  <-  CLI overrides (--set key.path=value)

The resolved config is snapshotted into every dataset release, so a release can
always be traced back to the exact parameters that produced it. That traceability
is the reason config is typed rather than a loose dict.

Example:
    >>> cfg = load_config("configs/brats.yaml", overrides=["qc.min_tumor_voxels=50"])
    >>> cfg.qc.min_tumor_voxels
    50
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class PathsConfig(BaseModel):
    """Filesystem layout for one pipeline run."""

    raw: Path = Field(description="Immutable landing zone for source data.")
    interim: Path = Field(description="Intermediate artifacts; safe to delete.")
    processed: Path = Field(description="Model-ready outputs.")
    outputs: Path = Field(description="Reports, figures, cards.")
    releases: Path = Field(description="Versioned dataset manifests.")

    def mkdirs(self) -> None:
        """Create the output directories. Idempotent.

        ``raw`` is deliberately excluded. It is an *input* -- the immutable
        landing zone -- and creating it would mask the most common
        misconfiguration there is: a typo in the path. Instead of a clear "raw
        data not found", the pipeline would helpfully create an empty directory,
        ingest zero files, and publish an empty release without complaint.

        Inputs are asserted, not created.
        """
        for p in (self.interim, self.processed, self.outputs, self.releases):
            p.mkdir(parents=True, exist_ok=True)


class QCConfig(BaseModel):
    """Thresholds for the quality-control engine.

    These are deliberately explicit rather than hard-coded in ``qc.py``: QC
    thresholds are a clinical//governance decision, and reviewers need to see
    them in the release manifest.
    """

    min_tumor_voxels: int = Field(default=10, ge=0)
    expected_modalities: list[str] = Field(default_factory=list)
    expected_orientation: str = Field(default="RAS")
    max_spacing_mm: float = Field(default=5.0, gt=0)
    min_shape: list[int] = Field(default=[16, 16, 16])
    fail_on: Literal["error", "warn", "never"] = "error"

    @field_validator("expected_orientation")
    @classmethod
    def _valid_orientation(cls, v: str) -> str:
        if len(v) != 3 or not set(v.upper()) <= set("RASLPI"):
            raise ValueError(f"orientation must be 3 axis codes from RASLPI, got {v!r}")
        return v.upper()


class PreprocessConfig(BaseModel):
    """Volume normalisation parameters."""

    target_spacing: list[float] = Field(default=[1.0, 1.0, 1.0])
    target_orientation: str = "RAS"
    intensity_norm: Literal["zscore", "minmax", "none"] = "zscore"
    clip_percentiles: list[float] = Field(default=[0.5, 99.5])


class SplitConfig(BaseModel):
    """Train/val/test partition parameters."""

    train: float = 0.7
    val: float = 0.15
    test: float = 0.15
    seed: int = 42
    stratify_by: str | None = None
    group_by: str | None = Field(
        default="patient_id",
        description="Column that must never straddle two splits (leakage guard).",
    )

    @field_validator("test")
    @classmethod
    def _sums_to_one(cls, v: float, info: Any) -> float:
        train = info.data.get("train", 0.0)
        val = info.data.get("val", 0.0)
        if abs(train + val + v - 1.0) > 1e-6:
            raise ValueError(f"splits must sum to 1.0, got {train + val + v}")
        return v


class TrainConfig(BaseModel):
    """Baseline model hyperparameters."""

    epochs: int = 2
    batch_size: int = 2
    lr: float = 1e-3
    roi_size: list[int] = Field(default=[64, 64, 64])
    num_workers: int = 0
    device: str = "auto"
    mlflow_uri: str | None = None
    experiment: str = "default"


class Config(BaseModel):
    """Root configuration object handed to every pipeline stage."""

    name: str
    seed: int = 42
    log_level: str = "INFO"
    json_logs: bool = False
    paths: PathsConfig
    qc: QCConfig = Field(default_factory=QCConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    extra: dict[str, Any] = Field(default_factory=dict)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new dict.

    Nested dicts merge key-by-key; every other type (including lists) is replaced
    wholesale. Replacing lists is intentional -- appending to a list of expected
    modalities across config layers would be surprising behaviour.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce_scalar(raw: str) -> Any:
    """Parse a CLI override value using YAML scalar rules.

    Gives ``true`` -> bool, ``3`` -> int, ``[1,2]`` -> list for free, and falls
    back to the raw string for anything YAML cannot parse.
    """
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _apply_override(tree: dict[str, Any], dotted: str) -> None:
    """Apply a single ``a.b.c=value`` override in place."""
    if "=" not in dotted:
        raise ValueError(f"override must look like key.path=value, got {dotted!r}")
    path, _, raw = dotted.partition("=")
    node = tree
    parts = path.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"cannot descend into non-mapping at {part!r} in {path!r}")
    node[parts[-1]] = _coerce_scalar(raw)


def load_config(
    path: str | Path,
    overrides: list[str] | None = None,
    base: str | Path | None = None,
) -> Config:
    """Load, merge and validate a pipeline configuration.

    Args:
        path: Pipeline-specific YAML file.
        overrides: Dotted ``key.path=value`` strings, applied last.
        base: Optional base YAML. Defaults to ``configs/base.yaml`` next to
            ``path`` when that file exists.

    Returns:
        A validated :class:`Config`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        pydantic.ValidationError: If the merged tree violates the schema.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")

    if base is None:
        candidate = path.parent / "base.yaml"
        base = candidate if candidate.exists() and candidate != path else None

    tree: dict[str, Any] = {}
    if base is not None:
        tree = yaml.safe_load(Path(base).read_text(encoding="utf-8")) or {}

    tree = _deep_merge(tree, yaml.safe_load(path.read_text(encoding="utf-8")) or {})

    for override in overrides or []:
        _apply_override(tree, override)

    return Config.model_validate(tree)


def dump_config(cfg: Config) -> dict[str, Any]:
    """Serialise a config to plain JSON-able types for manifest snapshotting."""
    return yaml.safe_load(
        yaml.safe_dump(cfg.model_dump(mode="json"), default_flow_style=False)
    )
