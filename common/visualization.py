"""Figure export for QC and model review.

Every figure is written to disk as a PNG by a pipeline stage. Nothing here needs
a notebook or a display -- the Agg backend is forced at import, so the same code
produces the same artifacts on a laptop and in CI.

That constraint is the point: a reviewer should be able to look at
``outputs/figures/`` after a run and see what the model saw, without re-executing
anyone's notebook.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from common.logging import get_logger

log = get_logger(__name__)

# Discrete colours for BraTS-style label maps: 1=necrotic, 2=edema, 4=enhancing.
_LABEL_COLORS: dict[int, tuple[float, float, float]] = {
    1: (0.85, 0.20, 0.20),
    2: (0.20, 0.75, 0.30),
    3: (0.20, 0.45, 0.90),
    4: (0.95, 0.75, 0.15),
}


def _normalise(slice_2d: np.ndarray) -> np.ndarray:
    """Scale a 2D slice to [0, 1] for display, tolerating constant input."""
    finite = slice_2d[np.isfinite(slice_2d)]
    if finite.size == 0:
        return np.zeros_like(slice_2d, dtype=float)
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        return np.zeros_like(slice_2d, dtype=float)
    return np.clip((slice_2d - lo) / (hi - lo), 0, 1)


def best_slice(mask: np.ndarray, axis: int = 2) -> int:
    """Index of the slice carrying the most labelled voxels.

    Picking the middle slice often shows an empty plane for small lesions, which
    makes review figures useless; picking the densest slice shows the finding.
    Falls back to the middle slice for an empty mask.
    """
    if mask is None or not np.any(mask):
        return mask.shape[axis] // 2 if mask is not None else 0
    axes = tuple(i for i in range(mask.ndim) if i != axis)
    return int(np.argmax((mask > 0).sum(axis=axes)))


def _overlay(ax: plt.Axes, image: np.ndarray, mask: np.ndarray | None, alpha: float) -> None:
    """Draw ``image`` in grey with ``mask`` colour-coded on top."""
    ax.imshow(_normalise(image).T, cmap="gray", origin="lower")
    if mask is None or not np.any(mask):
        return
    rgba = np.zeros((*mask.shape, 4))
    for label, colour in _LABEL_COLORS.items():
        hit = mask == label
        if hit.any():
            rgba[hit, :3] = colour
            rgba[hit, 3] = alpha
    ax.imshow(np.transpose(rgba, (1, 0, 2)), origin="lower")


def plot_slice_grid(
    volume: np.ndarray,
    mask: np.ndarray | None,
    out_path: Path,
    title: str = "",
    prediction: np.ndarray | None = None,
    axis: int = 2,
    alpha: float = 0.45,
) -> Path:
    """Render an MRI/CT slice with mask, prediction and overlay panels.

    Produces the four-panel figure the case-study README references: the raw
    slice, ground-truth mask, prediction (when supplied) and a colour overlay.

    Args:
        volume: 3D image array.
        mask: Optional 3D ground-truth label map.
        out_path: PNG destination.
        title: Figure suptitle.
        prediction: Optional 3D predicted label map.
        axis: Slice axis (0=sagittal, 1=coronal, 2=axial).
        alpha: Overlay opacity.

    Returns:
        The written path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    index = best_slice(mask, axis) if mask is not None else volume.shape[axis] // 2
    take = lambda arr: np.take(arr, index, axis=axis)  # noqa: E731

    img = take(volume)
    gt = take(mask) if mask is not None else None
    pred = take(prediction) if prediction is not None else None

    panels: list[tuple[str, np.ndarray, np.ndarray | None]] = [("Image", img, None)]
    if gt is not None:
        panels.append(("Ground truth", img, gt))
    if pred is not None:
        panels.append(("Prediction", img, pred))
        panels.append(("GT vs pred overlay", img, gt))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.4))
    axes = np.atleast_1d(axes)

    for ax, (label, image, overlay_mask) in zip(axes, panels, strict=True):
        _overlay(ax, image, overlay_mask, alpha)
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    fig.suptitle(f"{title}  (slice {index} along axis {axis})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    log.info("viz.slice_grid", extra={"path": str(out_path), "slice": index})
    return out_path


def plot_distribution(
    values: Sequence[float],
    out_path: Path,
    title: str,
    xlabel: str,
    bins: int = 30,
) -> Path:
    """Histogram with a median marker, for QC and bias reports."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))

    if data.size:
        ax.hist(data, bins=bins, color="#4C78A8", edgecolor="white")
        median = float(np.median(data))
        ax.axvline(median, color="#C0392B", linestyle="--",
                   label=f"median = {median:.2f}")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_categorical(
    counts: dict[str, int],
    out_path: Path,
    title: str,
    xlabel: str = "",
) -> Path:
    """Bar chart of category counts, sorted descending."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    if counts:
        items = sorted(counts.items(), key=lambda kv: -kv[1])
        labels = [str(k) for k, _ in items]
        values = [v for _, v in items]
        ax.bar(labels, values, color="#72B7B2", edgecolor="white")
        for i, v in enumerate(values):
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
        ax.tick_params(axis="x", rotation=30)
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
