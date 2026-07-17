"""Baseline 3D U-Net for BraTS tumour segmentation.

Scope
-----
This is a **baseline that consumes the data product**, not a competition entry.
Its purpose is to demonstrate that the released dataset loads, trains and
evaluates end to end -- that the manifest, the splits and the preprocessing are
actually usable by a model. A state-of-the-art nnU-Net would prove nothing extra
about the *data platform*, which is what this repository is about.

Design notes worth defending in review:

- **Reads the manifest, never the filesystem.** The loader consumes
  ``manifest_processed.parquet`` and the release splits. If the pipeline says a
  subject was rejected, the model cannot accidentally train on it. Globbing the
  processed directory would silently re-admit it.
- **Channels stacked in a fixed order** (T1, T1Gd, T2, FLAIR). Channel order is
  part of the data contract; a subject missing one is excluded upstream rather
  than zero-filled here.
- **BraTS label remap.** Raw labels are 1/2/4 (3 is unused). Training targets are
  the three clinically meaningful *nested* regions used by the challenge:
  TC (tumour core), WT (whole tumour), ET (enhancing tumour). Overlapping
  regions mean this is multi-label sigmoid, not multi-class softmax.

Requires the optional ``[ml]`` extra::

    pip install -e ".[ml]"
    python -m pipelines.brats.train --config configs/brats.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from common.config import Config, load_config
from common.logging import configure_logging, get_logger
from common.metadata import read_manifest

log = get_logger(__name__)

# Fixed channel order. Part of the data contract, not an implementation detail.
CHANNELS: tuple[str, ...] = ("t1", "t1ce", "t2", "flair")

# BraTS nested regions. Each is a union of raw labels, and they overlap by
# design: ET is inside TC is inside WT.
REGIONS: dict[str, tuple[int, ...]] = {
    "TC": (1, 4),      # tumour core: necrotic + enhancing
    "WT": (1, 2, 4),   # whole tumour: everything
    "ET": (4,),        # enhancing tumour
}


def _require_ml() -> None:
    """Fail with an actionable message when the ``[ml]`` extra is absent."""
    try:
        import monai  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Training requires the optional ML extra:\n"
            '    pip install -e ".[ml]"\n'
            f"(missing: {exc.name})"
        ) from exc


def to_regions(mask: np.ndarray) -> np.ndarray:
    """Convert a raw BraTS label map into a 3-channel multi-label target.

    Args:
        mask: Integer label map with values in {0, 1, 2, 4}.

    Returns:
        Float32 array of shape ``(3, *mask.shape)`` ordered (TC, WT, ET).
    """
    return np.stack(
        [np.isin(mask, labels).astype(np.float32) for labels in REGIONS.values()]
    )


class BratsDataset:
    """Loads subjects listed in the processed manifest for one split.

    Deliberately a plain class implementing ``__len__``/``__getitem__`` rather
    than a MONAI ``CacheDataset``: it is the *contract* (manifest-driven, fixed
    channel order) that matters here, and a plain class makes that contract
    readable without knowing MONAI's caching semantics.
    """

    def __init__(self, cfg: Config, split: str, roi: tuple[int, int, int]) -> None:
        self.cfg = cfg
        self.split = split
        self.roi = roi
        self.root = Path(cfg.paths.processed)

        manifest_path = Path(cfg.paths.outputs) / "metadata" / "manifest_processed.parquet"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"processed manifest not found: {manifest_path}\n"
                "Run the data pipeline first: python -m pipelines.brats.run"
            )

        frame = read_manifest(manifest_path)
        frame["_split"] = frame["extra"].apply(
            lambda e: (json.loads(e) if isinstance(e, str) else (e or {})).get("split")
        )
        frame = frame[frame["_split"] == split]

        self.subjects: list[str] = sorted(frame["patient_id"].unique().tolist())
        self._paths = {
            (row["patient_id"], row["modality"]): row["filepath"]
            for _, row in frame.iterrows()
        }
        self._masks = {
            row["patient_id"]: row["mask_path"]
            for _, row in frame.iterrows()
            if row.get("mask_path")
        }

        log.info("dataset.loaded", extra={
            "split": split, "n_subjects": len(self.subjects),
        })

    def __len__(self) -> int:
        return len(self.subjects)

    def _crop(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Centre-crop/pad to the configured ROI.

        Centre rather than random: with a baseline this small, a random crop often
        lands entirely on background and the model learns to predict zero.
        """
        out_img = np.zeros((image.shape[0], *self.roi), dtype=np.float32)
        out_msk = np.zeros((mask.shape[0], *self.roi), dtype=np.float32)

        slices_src, slices_dst = [], []
        for axis, want in enumerate(self.roi):
            have = image.shape[axis + 1]
            take = min(have, want)
            src_start = max(0, (have - take) // 2)
            dst_start = max(0, (want - take) // 2)
            slices_src.append(slice(src_start, src_start + take))
            slices_dst.append(slice(dst_start, dst_start + take))

        src = (slice(None), *slices_src)
        dst = (slice(None), *slices_dst)
        out_img[dst] = image[src]
        out_msk[dst] = mask[src]
        return out_img, out_msk

    def __getitem__(self, index: int) -> dict[str, Any]:
        import nibabel as nib

        subject = self.subjects[index]

        channels = []
        for modality in CHANNELS:
            path = self._paths.get((subject, modality))
            if path is None:
                raise KeyError(
                    f"subject {subject} is missing modality {modality!r}. "
                    "The pipeline should have excluded it; the manifest is inconsistent."
                )
            channels.append(
                np.asanyarray(nib.load(str(self.root / path)).dataobj, dtype=np.float32)
            )
        image = np.stack(channels)

        mask_path = self._masks.get(subject)
        raw_mask = (
            np.asanyarray(nib.load(str(self.root / mask_path)).dataobj)
            if mask_path else np.zeros(image.shape[1:], dtype=np.uint8)
        )
        target = to_regions(raw_mask)

        image, target = self._crop(image, target)
        return {"image": image, "label": target, "subject": subject}


def build_model() -> Any:
    """Construct the MONAI 3D U-Net.

    Instance norm rather than batch norm: batch size is 2, and batch statistics
    over two volumes are noise.
    """
    _require_ml()
    from monai.networks.nets import UNet

    return UNet(
        spatial_dims=3,
        in_channels=len(CHANNELS),
        out_channels=len(REGIONS),
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
        norm="instance",
    )


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    return {
        "image": torch.from_numpy(np.stack([b["image"] for b in batch])),
        "label": torch.from_numpy(np.stack([b["label"] for b in batch])),
        "subject": [b["subject"] for b in batch],
    }


def resolve_device(requested: str) -> Any:
    """Resolve ``"auto"`` to CUDA when available, else CPU."""
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def train(cfg: Config) -> dict[str, Any]:
    """Train the baseline and return the run summary.

    Logs to MLflow when ``train.mlflow_uri`` is set; otherwise metrics go to the
    structured log and ``outputs/brats/training_metrics.json``. Not requiring a
    tracking server for a smoke run is deliberate.
    """
    _require_ml()
    import torch
    from monai.losses import DiceLoss
    from torch.utils.data import DataLoader

    device = resolve_device(cfg.train.device)
    roi = tuple(cfg.train.roi_size)  # type: ignore[assignment]

    train_ds = BratsDataset(cfg, "train", roi)
    val_ds = BratsDataset(cfg, "val", roi)
    if len(train_ds) == 0:
        raise RuntimeError("training split is empty; check the pipeline output")

    train_dl = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                          num_workers=cfg.train.num_workers, collate_fn=_collate)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False,
                        num_workers=cfg.train.num_workers, collate_fn=_collate)

    model = build_model().to(device)
    # sigmoid, not softmax: the three regions are nested and overlap.
    loss_fn = DiceLoss(sigmoid=True, include_background=True)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    tracker = _mlflow_run(cfg)
    history: list[dict[str, float]] = []

    for epoch in range(cfg.train.epochs):
        model.train()
        train_loss = 0.0
        for batch in train_dl:
            optimiser.zero_grad()
            loss = loss_fn(model(batch["image"].to(device)), batch["label"].to(device))
            loss.backward()
            optimiser.step()
            train_loss += float(loss.item())

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                val_loss += float(
                    loss_fn(model(batch["image"].to(device)), batch["label"].to(device)).item()
                )

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss / max(len(train_dl), 1),
            "val_loss": val_loss / max(len(val_dl), 1),
        }
        history.append(row)
        log.info("train.epoch", extra=row)
        if tracker:
            tracker.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)

    out_dir = Path(cfg.paths.outputs)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = out_dir / "model_unet3d.pt"
    torch.save({"state_dict": model.state_dict(),
                "channels": CHANNELS, "regions": list(REGIONS)}, checkpoint)
    (out_dir / "training_metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    if tracker:
        tracker.end()

    summary = {
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "epochs": cfg.train.epochs,
        "device": str(device),
        "final_train_loss": round(history[-1]["train_loss"], 4) if history else None,
        "final_val_loss": round(history[-1]["val_loss"], 4) if history else None,
        "checkpoint": str(checkpoint),
    }
    log.info("train.complete", extra=summary)
    return summary


class _MlflowRun:
    """Thin MLflow wrapper so the training loop has no hard dependency on it."""

    def __init__(self, cfg: Config) -> None:
        import mlflow

        self._mlflow = mlflow
        mlflow.set_tracking_uri(cfg.train.mlflow_uri)
        mlflow.set_experiment(cfg.train.experiment)
        mlflow.start_run()
        mlflow.log_params({
            "epochs": cfg.train.epochs,
            "batch_size": cfg.train.batch_size,
            "lr": cfg.train.lr,
            "roi_size": str(cfg.train.roi_size),
            "dataset_version": cfg.extra.get("dataset_version"),
        })

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self._mlflow.log_metrics(metrics, step=step)

    def end(self) -> None:
        self._mlflow.end_run()


def _mlflow_run(cfg: Config) -> _MlflowRun | None:
    """Start an MLflow run when configured; degrade gracefully when not."""
    if not cfg.train.mlflow_uri:
        return None
    try:
        return _MlflowRun(cfg)
    except Exception as exc:
        log.warning("mlflow.unavailable", extra={"error": str(exc)})
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the BraTS 3D U-Net baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/brats.yaml"))
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=args.overrides)
    configure_logging(level=cfg.log_level, json_logs=cfg.json_logs)

    summary = train(cfg)
    print("\n" + "=" * 62)
    print("  Training complete")
    print("=" * 62)
    for key, value in summary.items():
        print(f"  {key:<18} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
