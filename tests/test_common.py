"""Tests for the shared framework: config, logging, metadata, storage, versioning."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from common.config import Config, load_config
from common.logging import configure_logging, get_logger
from common.metadata import ScanRecord, decode_list_column, read_manifest, write_manifest
from common.storage import human_size, relative_to, sha256_bytes, sha256_file
from common.versioning import (
    build_file_entries,
    compute_dataset_hash,
    create_release,
    node,
    read_release,
    verify_release,
    write_release,
)

# --------------------------------------------------------------------------- #
# config                                                                       #
# --------------------------------------------------------------------------- #


def test_load_config_merges_base_and_override(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "base",
                "seed": 1,
                "paths": {
                    "raw": "a",
                    "interim": "b",
                    "processed": "c",
                    "outputs": "d",
                    "releases": "e",
                },
                "qc": {"min_tumor_voxels": 10, "max_spacing_mm": 5.0},
            }
        )
    )
    (tmp_path / "child.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "child",
                "qc": {"min_tumor_voxels": 99},
            }
        )
    )

    cfg = load_config(tmp_path / "child.yaml")

    assert cfg.name == "child"  # child overrides base
    assert cfg.seed == 1  # inherited from base
    assert cfg.qc.min_tumor_voxels == 99
    assert cfg.qc.max_spacing_mm == 5.0  # nested merge preserves sibling keys


def test_cli_overrides_are_type_coerced(tmp_path: Path) -> None:
    (tmp_path / "c.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "x",
                "paths": {
                    "raw": "a",
                    "interim": "b",
                    "processed": "c",
                    "outputs": "d",
                    "releases": "e",
                },
            }
        )
    )

    cfg = load_config(
        tmp_path / "c.yaml",
        overrides=[
            "qc.min_tumor_voxels=50",
            "json_logs=true",
            "preprocess.target_spacing=[2.0, 2.0, 2.0]",
        ],
    )

    assert cfg.qc.min_tumor_voxels == 50 and isinstance(cfg.qc.min_tumor_voxels, int)
    assert cfg.json_logs is True
    assert cfg.preprocess.target_spacing == [2.0, 2.0, 2.0]


def test_split_ratios_must_sum_to_one(tmp_path: Path) -> None:
    (tmp_path / "c.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "x",
                "paths": {
                    "raw": "a",
                    "interim": "b",
                    "processed": "c",
                    "outputs": "d",
                    "releases": "e",
                },
                "split": {"train": 0.6, "val": 0.3, "test": 0.3},  # sums to 1.2
            }
        )
    )
    with pytest.raises(ValidationError, match=r"sum to 1\.0"):
        load_config(tmp_path / "c.yaml")


def test_invalid_orientation_rejected(tmp_path: Path) -> None:
    (tmp_path / "c.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "x",
                "paths": {
                    "raw": "a",
                    "interim": "b",
                    "processed": "c",
                    "outputs": "d",
                    "releases": "e",
                },
                "qc": {"expected_orientation": "XYZ"},
            }
        )
    )
    with pytest.raises(ValidationError):
        load_config(tmp_path / "c.yaml")


def test_missing_config_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_malformed_override_raises(tmp_path: Path) -> None:
    (tmp_path / "c.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "x",
                "paths": {
                    "raw": "a",
                    "interim": "b",
                    "processed": "c",
                    "outputs": "d",
                    "releases": "e",
                },
            }
        )
    )
    with pytest.raises(ValueError, match=r"key\.path=value"):
        load_config(tmp_path / "c.yaml", overrides=["no_equals_sign"])


# --------------------------------------------------------------------------- #
# logging                                                                      #
# --------------------------------------------------------------------------- #


def test_logger_survives_reserved_extra_keys() -> None:
    """A reserved key in `extra` must not raise (regression guard)."""
    configure_logging()
    log = get_logger("test.reserved")
    log.info("event", extra={"message": "collides", "module": "collides", "safe": 1})


# --------------------------------------------------------------------------- #
# metadata                                                                     #
# --------------------------------------------------------------------------- #


def test_modality_and_sex_are_normalised() -> None:
    rec = ScanRecord(patient_id="p", modality="  T1CE  ", filepath="f", sex="male")
    assert rec.modality == "t1ce"
    assert rec.sex == "M"

    assert ScanRecord(patient_id="p", modality="t1", filepath="f", sex="Female").sex == "F"
    assert ScanRecord(patient_id="p", modality="t1", filepath="f", sex="other").sex == "U"


def test_voxel_volume_handles_missing_spacing() -> None:
    assert ScanRecord(patient_id="p", modality="t1", filepath="f").voxel_volume_mm3 == 0.0
    rec = ScanRecord(patient_id="p", modality="t1", filepath="f", voxel_spacing=[2.0, 2.0, 2.0])
    assert rec.voxel_volume_mm3 == pytest.approx(8.0)


def test_manifest_roundtrips_through_every_format(
    tmp_path: Path, sample_record: ScanRecord
) -> None:
    paths = write_manifest([sample_record], tmp_path / "m")

    for fmt in ("parquet", "csv", "json"):
        frame = read_manifest(paths[fmt])
        assert len(frame) == 1
        assert frame.iloc[0]["patient_id"] == "subj-001"
        # List columns must survive every format identically.
        assert decode_list_column(frame, "shape").iloc[0] == [64, 64, 64]


def test_empty_manifest_has_schema_columns(tmp_path: Path) -> None:
    paths = write_manifest([], tmp_path / "empty")
    frame = read_manifest(paths["csv"])
    assert frame.empty
    assert "patient_id" in frame.columns


def test_unsupported_format_rejected(tmp_path: Path, sample_record: ScanRecord) -> None:
    with pytest.raises(ValueError, match="unsupported manifest format"):
        write_manifest([sample_record], tmp_path / "m", formats=["xml"])


# --------------------------------------------------------------------------- #
# storage                                                                      #
# --------------------------------------------------------------------------- #


def test_sha256_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    a.write_bytes(b"identical")
    b.write_bytes(b"identical")
    c.write_bytes(b"different")

    assert sha256_file(a) == sha256_file(b)
    assert sha256_file(a) != sha256_file(c)
    assert sha256_file(a) == sha256_bytes(b"identical")


def test_sha256_chunking_matches_whole_file(tmp_path: Path) -> None:
    """A chunk boundary must not change the digest."""
    path = tmp_path / "big"
    path.write_bytes(b"x" * 5000)
    assert sha256_file(path, chunk_size=7) == sha256_file(path, chunk_size=1 << 20)


def test_relative_to_uses_posix_separators(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "dir" / "f.nii.gz"
    target.parent.mkdir(parents=True)
    target.touch()
    assert relative_to(target, tmp_path) == "sub/dir/f.nii.gz"


def test_relative_to_outside_root_falls_back(tmp_path: Path) -> None:
    assert "/" in relative_to(Path("/somewhere/else/f.nii"), tmp_path)


def test_human_size() -> None:
    assert human_size(512) == "512.0 B"
    assert human_size(1536) == "1.5 KB"
    assert human_size(1024**3) == "1.0 GB"


# --------------------------------------------------------------------------- #
# versioning                                                                   #
# --------------------------------------------------------------------------- #


def test_dataset_hash_is_order_invariant_but_content_sensitive(sample_record: ScanRecord) -> None:
    other = sample_record.model_copy(update={"filepath": "b.nii.gz", "sha256": "b" * 64})
    forward = build_file_entries([sample_record, other])
    reverse = build_file_entries([other, sample_record])

    assert compute_dataset_hash(forward) == compute_dataset_hash(reverse)

    changed = build_file_entries([sample_record.model_copy(update={"sha256": "c" * 64}), other])
    assert compute_dataset_hash(forward) != compute_dataset_hash(changed)


def test_unhashed_records_are_excluded(sample_record: ScanRecord) -> None:
    entries = build_file_entries([sample_record, sample_record.model_copy(update={"sha256": None})])
    assert len(entries) == 1


def test_release_is_immutable(tmp_path: Path, brats_cfg: Config, sample_record: ScanRecord) -> None:
    release = create_release(
        "brats", "v1.0.0", [sample_record], brats_cfg, splits={"train": ["subj-001"]}
    )
    write_release(release, tmp_path)

    with pytest.raises(FileExistsError, match="immutable"):
        write_release(release, tmp_path)


def test_release_roundtrip_and_index(
    tmp_path: Path, brats_cfg: Config, sample_record: ScanRecord
) -> None:
    release = create_release(
        "brats",
        "v1.0.0",
        [sample_record],
        brats_cfg,
        splits={"train": ["subj-001"]},
        lineage=[node("ingest", "scanned", root="x")],
        qc_summary={"n_errors": 0},
    )
    manifest = write_release(release, tmp_path)
    loaded = read_release(manifest)

    assert loaded.dataset_hash == release.dataset_hash
    assert loaded.n_files == 1
    assert loaded.lineage[0].stage == "ingest"
    assert loaded.config_snapshot["name"] == "brats"  # config is snapshotted

    index = json.loads((tmp_path / "releases.json").read_text())
    assert index[0]["version"] == "v1.0.0"
    assert (tmp_path / "v1.0.0" / "SHA256SUMS").exists()


def test_verify_release_detects_tampering(tmp_path: Path, brats_cfg: Config) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    target = data_root / "scan.nii.gz"
    target.write_bytes(b"original content")

    rec = ScanRecord(
        patient_id="p1",
        modality="t1",
        filepath="scan.nii.gz",
        sha256=sha256_file(target),
        file_size_bytes=target.stat().st_size,
    )
    release = create_release("brats", "v1.0.0", [rec], brats_cfg)

    assert verify_release(release, data_root) == []

    target.write_bytes(b"tampered content!")
    problems = verify_release(release, data_root)
    assert len(problems) == 1 and "hash mismatch" in problems[0]

    target.unlink()
    assert "missing" in verify_release(release, data_root)[0]
