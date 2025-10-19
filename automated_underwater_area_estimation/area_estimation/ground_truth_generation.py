import re
from pathlib import Path
from typing import List, Tuple, Optional

from PIL import Image
import numpy as np
import pandas as pd

# -------------------
# CONFIG
# -------------------
BASE = Path("./automated_underwater_area_estimation/data_preprocessed/IBF/")
IMAGES_DIR = BASE / "images"
CPCS_DIR = BASE / "cpcs"

OUTPUT_CSV = Path(__file__).parent / "quadrant_points.csv"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

QUADRANT_WIDTH, QUADRANT_HEIGHT = 52, 52

# -------------------
# CPC parsing helpers
# -------------------
FLOAT = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
PAIR_RE = re.compile(rf'^\s*"?\s*({FLOAT})\s*"?\s*,\s*"?\s*({FLOAT})\s*"?\s*$')
NUMS_RE = re.compile(rf"{FLOAT}")


def _read_text(path: Path) -> str:
    # tolerant file reader (handles UTF-8 BOM / latin-1)
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=enc, errors="strict")
        except Exception:
            pass
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_cpc(
    path: Path,
) -> tuple[float, float, list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Returns: work_w, work_h, roi_points[4], sample_points[N]
    - work_w/work_h: CPC 'working' coordinate space from header line
    - roi_points: 4 ROI corners (C1..C4) in working coords (order = file order)
    - sample_points: CPCe points in working coords
    """
    text = _read_text(path)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() != ""]
    if not lines:
        raise ValueError(f"Empty CPC: {path}")

    # Header: last 4 numbers contain working width/height in typical CPCe files
    header = lines[0]
    nums = [float(s) for s in NUMS_RE.findall(header)]
    if len(nums) < 4:
        raise ValueError(f"Unexpected CPC header format in {path}")
    work_w = float(nums[-4])
    work_h = float(nums[-3])

    # Next: collect first 4 float-pair lines as ROI corners
    roi: List[Tuple[float, float]] = []
    idx = 1
    while idx < len(lines) and len(roi) < 4:
        m = PAIR_RE.match(lines[idx])
        if m:
            roi.append((float(m.group(1)), float(m.group(2))))
        idx += 1
    if len(roi) != 4:
        raise ValueError(f"Could not find 4 ROI corners in {path}")

    # Next integer line = number of CPCe points
    n_points = 0
    while idx < len(lines):
        ln = lines[idx]
        if ln.isdigit():
            n_points = int(ln)
            idx += 1
            break
        idx += 1

    # Then that many float pairs as CPCe points
    pts: List[Tuple[float, float]] = []
    for _ in range(n_points):
        if idx >= len(lines):
            break
        m = PAIR_RE.match(lines[idx])
        if m:
            pts.append((float(m.group(1)), float(m.group(2))))
        idx += 1

    return work_w, work_h, roi, pts


# -------------------
# Image lookup
# -------------------
def find_image_for_stem(stem: str) -> Optional[Path]:
    # try common extensions (case-insensitive)
    for ext in IMAGE_EXTS:
        p = IMAGES_DIR / f"{stem}{ext}"
        if p.exists():
            return p
        p2 = IMAGES_DIR / f"{stem}{ext.upper()}"
        if p2.exists():
            return p2
    # fallback: scan directory (handles weird cases)
    for p in IMAGES_DIR.iterdir():
        if (
            p.is_file()
            and p.suffix.lower() in IMAGE_EXTS
            and p.stem.lower() == stem.lower()
        ):
            return p
    return None


# -------------------
# Build DF + overlays
# -------------------
def build_quadrant_df_and_overlays() -> pd.DataFrame:
    if not IMAGES_DIR.is_dir():
        raise SystemExit(f"Missing images dir: {IMAGES_DIR}")
    if not CPCS_DIR.is_dir():
        raise SystemExit(f"Missing cpcs dir: {CPCS_DIR}")

    cpc_files = sorted(
        [p for p in CPCS_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".cpc"]
    )
    print(f"Found {len(cpc_files)} CPC files")

    rows = []
    made, skipped = 0, 0

    for cpc in cpc_files:
        stem = cpc.stem
        img_path = find_image_for_stem(stem)
        if img_path is None:
            print(f"[WARN] No matching image for {cpc.name} — skipping.")
            skipped += 1
            continue

        try:
            work_w, work_h, roi, pts = parse_cpc(cpc)
        except Exception as e:
            print(f"[WARN] Parse failed for {cpc.name}: {e}")
            skipped += 1
            continue

        img = Image.open(img_path)
        w, h = img.size
        sx = w / work_w
        sy = h / work_h

        # Scale ROI corners into pixel space (C1..C4)
        roi_px = [(x * sx, y * sy) for (x, y) in roi]
        # Scale CPCe points too (optional overlay)

        # Row for the DataFrame: file name + four corners (pixel coords)
        (c1x, c1y), (c2x, c2y), (c3x, c3y), (c4x, c4y) = roi_px
        rows.append(
            {
                "stem": stem,
                "image_path": str(img_path),
                "cpc_path": str(cpc),
                "c1x": c1x,
                "c1y": c1y,
                "c2x": c2x,
                "c2y": c2y,
                "c3x": c3x,
                "c3y": c3y,
                "c4x": c4x,
                "c4y": c4y,
            }
        )
        made += 1

    df = pd.DataFrame(
        rows,
        columns=[
            "stem",
            "image_path",
            "cpc_path",
            "c1x",
            "c1y",
            "c2x",
            "c2y",
            "c3x",
            "c3y",
            "c4x",
            "c4y",
        ],
    )

    # Save CSV and return DF
    if len(df):
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved table to: {OUTPUT_CSV}")
    print(f"Overlays written: {made}, skipped: {skipped}")
    return df


def dist_px(x1, y1, x2, y2):
    """
    Euclidean distance in *pixels* between points (x1,y1) and (x2,y2).
    Works with scalars, NumPy arrays, or pandas Series (vectorized).
    Uses np.hypot for numerical stability: sqrt(dx**2 + dy**2).
    """
    dx = np.asarray(x2) - np.asarray(x1)
    dy = np.asarray(y2) - np.asarray(y1)
    return np.hypot(dx, dy)


if __name__ == "__main__":
    df = build_quadrant_df_and_overlays()

    df["width_lower"] = dist_px(df["c1x"], df["c1y"], df["c2x"], df["c2y"])
    df["width_upper"] = dist_px(df["c3x"], df["c3y"], df["c4x"], df["c4y"])
    df["quadrant_width_px"] = df[["width_lower", "width_upper"]].mean(axis=1)

    df["height_left"] = dist_px(df["c1x"], df["c1y"], df["c4x"], df["c4y"])
    df["height_right"] = dist_px(df["c2x"], df["c2y"], df["c3x"], df["c3y"])
    df["quadrant_height_px"] = df[["height_left", "height_right"]].mean(axis=1)

    df["aspect_ratio"] = df["quadrant_width_px"] / df["quadrant_height_px"]
    df["quadrant_width_irl_cm"] = QUADRANT_WIDTH
    df["quadrant_height_irl_cm"] = QUADRANT_WIDTH

    df["pixel_width_cm"] = df["quadrant_width_irl_cm"] / df["quadrant_width_px"]
    df["pixel_height_cm"] = df["quadrant_height_irl_cm"] / df["quadrant_height_px"]
    df["pixel_area_gt_cm^2"] = df["pixel_width_cm"] * df["pixel_height_cm"]
    df.to_csv(OUTPUT_CSV)
