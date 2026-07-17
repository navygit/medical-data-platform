"""Shared pytest fixtures.

Fixtures build a tiny synthetic dataset in a temp directory once per session.
Tests exercise the real pipeline code against real files on disk -- no mocking of
nibabel or pydicom. Mocking the I/O layer would test the mocks, and the bugs
that actually occur in medical data pipelines are I/O bugs: a truncated gzip, an
unexpected affine, a tag that is present but empty.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from common.config import Config, load_config
from common.metadata import ScanRecord


@pytest.fixture(scope="session")
def synth_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate the synthetic corpus once for the whole session.

    Small volumes (16^3) and few subjects keep the suite fast; the defect plan is
    fixed, so the assertions stay deterministic regardless of size.
    """
    from scripts.generate_synthetic_data import generate_brats, generate_lits, generate_mimic

    root = tmp_path_factory.mktemp("synth")
    generate_brats(root / "brats", n_subjects=10, shape=(16, 16, 16), seed=7)
    generate_mimic(root / "mimic", n_studies=6, size=64, seed=7)
    generate_lits(root / "lits", n_subjects=10, shape=(16, 16, 16), seed=7)
    return root


@pytest.fixture
def brats_cfg(synth_root: Path, tmp_path: Path) -> Config:
    """A BraTS config pointed at the synthetic corpus and a temp output tree."""
    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs" / "brats.yaml")
    cfg.paths.raw = synth_root / "brats"
    cfg.paths.interim = tmp_path / "interim"
    cfg.paths.processed = tmp_path / "processed"
    cfg.paths.outputs = tmp_path / "outputs"
    cfg.paths.releases = tmp_path / "releases"
    cfg.paths.mkdirs()
    return cfg


@pytest.fixture
def lits_cfg(synth_root: Path, tmp_path: Path) -> Config:
    """A LiTS config pointed at the synthetic corpus."""
    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs" / "lits.yaml")
    cfg.paths.raw = synth_root / "lits"
    cfg.paths.interim = tmp_path / "interim"
    cfg.paths.processed = tmp_path / "processed"
    cfg.paths.outputs = tmp_path / "outputs"
    cfg.paths.releases = tmp_path / "releases"
    cfg.paths.mkdirs()
    return cfg


@pytest.fixture
def mimic_cfg(synth_root: Path, tmp_path: Path) -> Config:
    """A MIMIC config pointed at the synthetic corpus."""
    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs" / "mimic.yaml")
    cfg.paths.raw = synth_root / "mimic"
    cfg.paths.interim = tmp_path / "interim"
    cfg.paths.processed = tmp_path / "processed"
    cfg.paths.outputs = tmp_path / "outputs"
    cfg.paths.releases = tmp_path / "releases"
    cfg.paths.mkdirs()
    return cfg


@pytest.fixture
def sample_record() -> ScanRecord:
    """A well-formed record that passes every QC check."""
    return ScanRecord(
        patient_id="subj-001",
        modality="t1",
        filepath="subj-001/subj-001_t1.nii.gz",
        shape=[64, 64, 64],
        voxel_spacing=[1.0, 1.0, 1.0],
        orientation="RAS",
        intensity_min=0.0,
        intensity_max=1000.0,
        intensity_mean=350.0,
        intensity_std=120.0,
        mask_available=True,
        mask_path="subj-001/subj-001_seg.nii.gz",
        tumor_volume_mm3=5000.0,
        sha256="a" * 64,
        file_size_bytes=1024,
    )
