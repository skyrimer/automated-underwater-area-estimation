import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Dict, Tuple


# Processing tensor
def to_mask2d(m_or_path: Union[str, torch.Tensor, np.ndarray]) -> np.ndarray:
    """Accepts a .pt path OR a torch.Tensor OR a NumPy array. Returns 2D binary mask {0,1}."""
    if isinstance(m_or_path, str):
        t = torch.load(m_or_path, map_location="cpu")
        m = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    elif isinstance(m_or_path, torch.Tensor):
        m = m_or_path.detach().cpu().numpy()
    else:
        m = np.asarray(m_or_path)
    m = (np.squeeze(m) > 0).astype(np.uint8)
    if m.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {m.shape}")
    return m


# Finding corners
def point_bottom_right(m: np.ndarray) -> Tuple[int, int]:
    """Bottom-right by maximizing x*y."""
    ys, xs = np.where(m == 1)
    if ys.size == 0:
        raise ValueError("Mask is empty.")
    i = (ys.astype(np.int64) * xs.astype(np.int64)).argmax()
    return int(ys[i]), int(xs[i])


def point_top_left(m: np.ndarray) -> Tuple[int, int]:
    """Top-left by minimizing (x+y) with x>0,y>0."""
    ys, xs = np.where(m == 1)
    keep = (ys > 0) & (xs > 0)
    ys, xs = ys[keep], xs[keep]
    if ys.size == 0:
        raise ValueError("No white pixels with x>0 and y>0.")
    sums = ys.astype(np.int64) + xs.astype(np.int64)
    min_sum = sums.min()
    idx = np.where(sums == min_sum)[0]
    y_sel = int(ys[idx].min())
    x_sel = int(xs[idx][ys[idx] == y_sel].min())
    return y_sel, x_sel


def point_top_right(m: np.ndarray) -> Tuple[int, int]:
    """Top-right by maximizing (x - y); tie-break largest x, then smallest y."""
    ys, xs = np.where(m == 1)
    diffs = xs.astype(np.int64) - ys.astype(np.int64)
    best = diffs.max()
    keep = diffs == best
    xs_t, ys_t = xs[keep], ys[keep]
    x_sel = int(xs_t.max())
    y_sel = int(ys_t[xs_t == x_sel].min())
    return y_sel, x_sel


def point_bottom_left(m: np.ndarray) -> Tuple[int, int]:
    """Bottom-left by maximizing (y - x); tie-break largest y, then smallest x."""
    ys, xs = np.where(m == 1)
    diffs = ys.astype(np.int64) - xs.astype(np.int64)
    best = diffs.max()
    keep = diffs == best
    ys_t, xs_t = ys[keep], xs[keep]
    y_sel = int(ys_t.max())
    x_sel = int(xs_t[ys_t == y_sel].min())
    return y_sel, x_sel


def find_four_points(
    m_or_path: Union[str, torch.Tensor, np.ndarray],
) -> Dict[str, Tuple[int, int]]:
    """Return dict with TL, TR, BR, BL as (y,x)."""
    m = to_mask2d(m_or_path)
    return {
        "TL": point_top_left(m),
        "TR": point_top_right(m),
        "BR": point_bottom_right(m),
        "BL": point_bottom_left(m),
    }


# Computing distance
def compute_distances(m_or_path):
    """
    Given points dict with keys {'TL','TR','BR','BL'} -> (y,x),
    return Euclidean distances (pixels) for:
      TR–TL, TR–BR, TL–BL, BR–BL, TL–BR, TR–BL.
    """
    points = find_four_points(m_or_path)

    def d(p, q):
        (y1, x1), (y2, x2) = points[p], points[q]
        return float(np.hypot(x2 - x1, y2 - y1))

    return {
        "TR_TL": d("TR", "TL"),
        "TR_BR": d("TR", "BR"),
        "TL_BL": d("TL", "BL"),
        "BR_BL": d("BR", "BL"),
        "TL_BR": d("TL", "BR"),
        "TR_BL": d("TR", "BL"),
    }


# Plot
def plot_points_and_distances(
    m_or_path: Union[str, torch.Tensor, np.ndarray], points: Dict[str, Tuple[int, int]]
) -> None:
    """Overlay TL/TR/BR/BL and requested segments with labels on the mask."""
    m = to_mask2d(m_or_path)
    H, W = m.shape
    fig, ax = plt.subplots(figsize=(8, 8 * H / W))
    ax.imshow(
        m,
        cmap="gray",
        interpolation="nearest",
        origin="upper",
        extent=(-0.5, W - 0.5, H - 0.5, -0.5),
    )

    # plot points
    for k, (y, x) in points.items():
        ax.scatter([x], [y], s=80)
        ax.annotate(
            f"{k} ({y},{x})",
            xy=(x, y),
            xytext=(x + 12, y + 12),
            arrowprops=dict(arrowstyle="->"),
        )

    # draw requested pairs
    pairs = [
        ("TR", "TL"),
        ("TR", "BR"),
        ("TL", "BL"),
        ("BR", "BL"),
        ("TL", "BR"),
        ("TR", "BL"),
    ]
    for a, b in pairs:
        (y1, x1), (y2, x2) = points[a], points[b]
        ax.plot([x1, x2], [y1, y2], linewidth=2)
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(xm + 5, ym + 5, f"{a}-{b}: {np.hypot(x2-x1,y2-y1):.2f}px")

    # grid
    major, minor = 50, 10
    ax.set_xticks(np.arange(0, W, major))
    ax.set_xticks(np.arange(0, W, minor), minor=True)
    ax.set_yticks(np.arange(0, H, major))
    ax.set_yticks(np.arange(0, H, minor), minor=True)
    ax.grid(which="major", linestyle="-", linewidth=0.6, alpha=0.35)
    ax.grid(which="minor", linestyle=":", linewidth=0.4, alpha=0.20)

    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_title("Four points & distances")
    plt.tight_layout()
    plt.show()


def pix_to_cm_square(m_or_path, width_side=53, height_side=53):
    distances = compute_distances(m_or_path).copy()

    # Calculate diagonal using Pythagorean theorem
    diagonal = float(np.hypot(width_side, height_side))

    for k, l in distances.items():
        # If the distance is a diagonal
        if k == "TL_BR" or k == "TR_BL":
            distances[k] = (diagonal / l) ** 2
        # If it is a width side
        elif k == "TR_TL" or k == "BR_BL":
            distances[k] = (width_side / l) ** 2
        # If it is a height side
        elif k == "TL_BL" or k == "TR_BR":
            distances[k] = (height_side / l) ** 2

    return distances


def estimate_area_using_quadrant(m_or_path, quadrant_width, quadrant_height, pct=0.08):
    """
    Return the average of values within [median - pct*median, median + pct*median].
    m_or_path: mask path or tensor.
    pct: fraction for tolerance (e.g., 0.08 for 8%).
    """
    distances = pix_to_cm_square(m_or_path, quadrant_width, quadrant_height)
    vals = list(distances.values())
    if len(vals) != 6:
        raise ValueError(f"Expected 6 values, got {len(vals)}")

    s = sorted(vals)
    median = 0.5 * (s[2] + s[3])
    tol = pct * median
    lo, hi = median - tol, median + tol

    # All the kept values within the band
    kept = [v for v in vals if lo <= v <= hi]
    return (sum(kept) / len(kept)) if kept else None
