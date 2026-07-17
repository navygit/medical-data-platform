"""Evaluation: Dice, IoU, Hausdorff-95, plus prediction figures.

Metric choices and their reasons:

- **Dice** -- the field standard for segmentation overlap.
- **IoU** -- monotonic with Dice but penalises errors harder; reported because
  reviewers from a detection background expect it.
- **Hausdorff-95** -- a *boundary* metric. Dice is dominated by the bulk of a
  large lesion and can look excellent while the boundary is badly wrong, which is
  precisely what matters for surgical planning and volumetry. The 95th percentile
  variant is used rather than the maximum because the maximum is decided by a
  single outlier voxel.

The empty-mask convention is explicit and worth stating: when both prediction and
ground truth are empty, Dice is **1.0** (a correct negative), and when exactly one
is empty it is **0.0**. Libraries disagree on this, and silently averaging NaNs is
how a model that predicts nothing ends up reporting a great score.

Metrics are reported **per region** (TC/WT/ET), never as a single averaged number.
ET is the hardest region and averaging hides that.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common.config import Config, load_config
from common.logging import configure_logging, get_logger
from common.visualization import plot_slice_grid
from pipelines.brats.train import CHANNELS, REGIONS, BratsDataset, build_model, resolve_device

log = get_logger(__name__)


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Dice coefficient for one binary region.

    Returns 1.0 when both are empty (a correct negative) and 0.0 when only one
    is. Returning NaN here would let a do-nothing model average its way to a
    respectable score.
    """
    pred_sum, target_sum = float(pred.sum()), float(target.sum())
    if pred_sum == 0 and target_sum == 0:
        return 1.0
    if pred_sum == 0 or target_sum == 0:
        return 0.0
    return float(2.0 * np.logical_and(pred, target).sum() / (pred_sum + target_sum))


def iou_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Intersection over union, with the same empty-mask convention as Dice."""
    union = float(np.logical_or(pred, target).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, target).sum() / union)


def _surface_points(mask: np.ndarray) -> np.ndarray:
    """Coordinates of surface voxels (foreground voxels touching background)."""
    from scipy.ndimage import binary_erosion

    if not mask.any():
        return np.empty((0, mask.ndim))
    surface = mask & ~binary_erosion(mask, border_value=0)
    return np.argwhere(surface)


def hausdorff95(
    pred: np.ndarray, target: np.ndarray, spacing: tuple[float, ...] = (1, 1, 1)
) -> float:
    """Symmetric 95th-percentile Hausdorff distance in millimetres.

    Args:
        pred: Binary prediction.
        target: Binary ground truth.
        spacing: Voxel spacing, so the result is in mm rather than voxels.

    Returns:
        The distance, ``0.0`` if both masks are empty, or ``inf`` if exactly one
        is (an infinite boundary error is the honest answer -- there is no
        boundary to compare).
    """
    from scipy.spatial import cKDTree

    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return float("inf")

    scale = np.asarray(spacing, dtype=float)
    a = _surface_points(pred) * scale
    b = _surface_points(target) * scale
    if len(a) == 0 or len(b) == 0:
        return float("inf")

    forward = cKDTree(b).query(a)[0]
    backward = cKDTree(a).query(b)[0]
    return float(max(np.percentile(forward, 95), np.percentile(backward, 95)))


def evaluate_subject(
    pred: np.ndarray, target: np.ndarray, spacing: tuple[float, ...] = (1, 1, 1)
) -> dict[str, float]:
    """Compute every metric for every region of one subject.

    Args:
        pred: Binary prediction, shape ``(n_regions, D, H, W)``.
        target: Binary ground truth, same shape.
        spacing: Voxel spacing in mm.

    Returns:
        Flat mapping like ``{"dice_TC": 0.8, "hd95_ET": 3.2, ...}``.
    """
    out: dict[str, float] = {}
    for i, region in enumerate(REGIONS):
        p, t = pred[i].astype(bool), target[i].astype(bool)
        out[f"dice_{region}"] = dice_score(p, t)
        out[f"iou_{region}"] = iou_score(p, t)
        out[f"hd95_{region}"] = hausdorff95(p, t, spacing)
    return out


def evaluate(cfg: Config, split: str = "test", threshold: float = 0.5) -> pd.DataFrame:
    """Evaluate the trained checkpoint on a split and export figures.

    Returns:
        One row per subject with every metric.

    Raises:
        FileNotFoundError: If no checkpoint exists.
    """
    import torch

    checkpoint_path = Path(cfg.paths.outputs) / "model_unet3d.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"no checkpoint at {checkpoint_path}\n"
            "Train first: python -m pipelines.brats.train --config configs/brats.yaml"
        )

    device = resolve_device(cfg.train.device)
    roi = tuple(cfg.train.roi_size)  # type: ignore[assignment]

    dataset = BratsDataset(cfg, split, roi)
    if len(dataset) == 0:
        log.warning("evaluate.empty_split", extra={"split": split})
        return pd.DataFrame()

    model = build_model().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["state_dict"])
    model.eval()

    spacing = tuple(cfg.preprocess.target_spacing)
    figures_dir = Path(cfg.paths.outputs) / "figures"
    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            image = torch.from_numpy(sample["image"]).unsqueeze(0).to(device)

            logits = model(image)
            pred = (torch.sigmoid(logits).cpu().numpy()[0] > threshold).astype(np.uint8)
            target = sample["label"].astype(np.uint8)

            metrics = evaluate_subject(pred, target, spacing)
            rows.append({"subject": sample["subject"], "split": split, **metrics})
            log.info(
                "evaluate.subject",
                extra={
                    "subject": sample["subject"],
                    **{k: round(v, 3) for k, v in metrics.items() if k.startswith("dice")},
                },
            )

            # Figures for the first few subjects: WT is the most legible region.
            if i < 3:
                plot_slice_grid(
                    volume=sample["image"][CHANNELS.index("t1ce")],
                    mask=target[1],  # WT ground truth
                    prediction=pred[1],  # WT prediction
                    out_path=figures_dir / f"pred_{sample['subject']}.png",
                    title=f"{sample['subject']} - whole tumour",
                )

    frame = pd.DataFrame(rows)
    out_dir = Path(cfg.paths.outputs)
    frame.to_csv(out_dir / f"metrics_{split}.csv", index=False)

    # Hausdorff can be inf when a region is absent; excluded from the mean so one
    # empty region does not render the whole summary NaN.
    summary = {
        column: round(float(frame[column].replace([np.inf, -np.inf], np.nan).mean()), 4)
        for column in frame.columns
        if column not in ("subject", "split")
    }
    (out_dir / f"metrics_{split}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    log.info("evaluate.complete", extra={"split": split, "n_subjects": len(frame), **summary})
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the BraTS baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/brats.yaml"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=args.overrides)
    configure_logging(level=cfg.log_level, json_logs=cfg.json_logs)

    frame = evaluate(cfg, split=args.split, threshold=args.threshold)
    if frame.empty:
        print("No subjects to evaluate.")
        return 1

    print("\n" + "=" * 62)
    print(f"  Evaluation on '{args.split}' ({len(frame)} subjects)")
    print("=" * 62)
    for region in REGIONS:
        dice = frame[f"dice_{region}"].mean()
        iou = frame[f"iou_{region}"].mean()
        hd = frame[f"hd95_{region}"].replace([np.inf, -np.inf], np.nan).mean()
        print(f"  {region:<4} Dice {dice:.3f}   IoU {iou:.3f}   HD95 {hd:.2f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
