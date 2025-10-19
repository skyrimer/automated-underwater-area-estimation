from typing import Any
from torch.utils.data import Dataset
from PIL import Image
from transformers import (
    AutoImageProcessor,
    SegformerForSemanticSegmentation,
    TrainingArguments,
    Trainer,
)

import torch, torch.nn.functional as F
import numpy as np

import re, random
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict

EPS = 1e-6


def load_mask_pt(path: str) -> torch.Tensor:
    """Return torch.bool (H,W)."""
    data = torch.load(path, map_location="cpu")
    t = data["mask"] if isinstance(data, dict) and "mask" in data else data
    if not isinstance(t, torch.Tensor):
        raise ValueError(f"Unsupported mask in {path}")
    if t.ndim == 3 and t.shape[0] in (1, 3):
        t = t.max(dim=0).values
    elif t.ndim == 3 and t.shape[2] == 1:
        t = t[..., 0]
    t = t > 0.5 if t.dtype.is_floating_point else t.bool()
    return t


def discover_pairs(img_dir: Path, mask_dir: Path) -> List[Tuple[Path, Path]]:
    pairs = []
    for p in img_dir.iterdir():
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = p.stem  # e.g., foo_aug00
        m = mask_dir / f"{stem}.pt"
        if m.exists():
            pairs.append((p, m))
    return sorted(pairs, key=lambda x: x[0].name)


def _original_id(stem: str) -> str:
    """ID that all augmented variants of the same original share."""
    augmentation_regex = re.compile(r"^(.*?)(?:_rot-?\d+_aug\d+)$")
    m = augmentation_regex.match(stem)
    return m.group(1) if m else stem


def _dataset_id(stem: str) -> str:
    """Prefix before the first underscore is the dataset label."""
    return stem.split("_", 1)[0] if "_" in stem else stem


def split_by_dataset_and_group(
    pairs: List[Tuple[Path, Path]],
    val_ratio: float = 0.2,
    seed: int = 42,
    holdout_datasets: set[str] | None = None,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """
    One-shot train/val split with NO CV:
      - Groups by original image so augmentations can't leak across splits.
      - Respects dataset prefixes; optionally holds out entire datasets.
      - Balances per dataset by taking ~val_ratio of groups from each dataset.
    """
    rng = random.Random(seed)
    holdout_datasets = set(holdout_datasets or [])

    # 1) Build group registry: one entry per ORIGINAL (not per augmented file)
    group_to_indices: Dict[str, list[int]] = defaultdict(list)
    group_dataset: Dict[str, str] = {}
    stems = [p[0].stem for p in pairs]

    for i, stem in enumerate(stems):
        gid = _original_id(stem)
        ds = _dataset_id(stem)
        group_to_indices[gid].append(i)
        group_dataset[gid] = ds  # consistent per group

    # 2) Partition group IDs per dataset
    ds_to_groups: Dict[str, list[str]] = defaultdict(list)
    for gid, ds in group_dataset.items():
        ds_to_groups[ds].append(gid)

    # 3) Assign groups to VAL/ TRAIN
    val_group_ids, train_group_ids = set(), set()

    # 3a) Hard holdout of entire datasets (optional)
    for ds in holdout_datasets:
        for gid in ds_to_groups.get(ds, []):
            val_group_ids.add(gid)

    # 3b) For the remaining datasets, sample ~val_ratio groups into VAL
    for ds, gids in ds_to_groups.items():
        if ds in holdout_datasets:
            continue
        gids = gids[:]  # copy
        rng.shuffle(gids)
        n_val = max(1, round(len(gids) * val_ratio)) if len(gids) > 0 else 0
        val_group_ids.update(gids[:n_val])
        train_group_ids.update(gids[n_val:])

    # 4) Materialize file-level splits from group IDs
    train_pairs, val_pairs = [], []
    for gid, idxs in group_to_indices.items():
        bucket = val_pairs if gid in val_group_ids else train_pairs
        for i in idxs:
            bucket.append(pairs[i])

    # 5) Safety checks
    assert train_group_ids.isdisjoint(
        val_group_ids
    ), "Group leakage detected (same original in train & val)."
    if holdout_datasets:
        # ensure no held-out dataset slipped into train
        bad = [g for g in train_group_ids if group_dataset[g] in holdout_datasets]
        assert not bad, f"Held-out dataset groups in train: {bad[:5]}..."

    # (optional) print dataset mix
    from collections import Counter

    print("Train datasets:", Counter(_dataset_id(p[0].stem) for p in train_pairs))
    print("Val datasets  :", Counter(_dataset_id(p[0].stem) for p in val_pairs))

    return train_pairs, val_pairs


def save_split(stems: List[str], path: Path):
    path.write_text("\n".join(stems) + "\n", encoding="utf-8")


class QuadPreAugDataset(Dataset):
    """
    Dataset that reads pre-augmented files from disk (no runtime Albumentations).
    Uses HF processor for normalization and (optional) resize.
    """

    def __init__(
        self,
        items: List[Tuple[Path, Path]],
        processor,
        target_size: Tuple[int, int] | None,
        train: bool,
    ):
        self.items = items
        self.processor = processor
        self.target_size = target_size
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx) -> Dict[str, Any]:
        img_path, mask_path = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        mask = load_mask_pt(str(mask_path)).to(torch.int8)  # (H,W) 0/1

        # Prepare inputs via processor; if target_size is set, resize here
        if self.target_size is not None:
            w, h = self.target_size
            encoded = self.processor(
                images=img,
                segmentation_maps=mask.numpy(),
                return_tensors="pt",
                size={"height": h, "width": w},
            )
        else:
            encoded = self.processor(
                images=img, segmentation_maps=mask.numpy(), return_tensors="pt"
            )

        return {k: v.squeeze(0) for k, v in encoded.items()}


# --- Dice + CE (foreground-aware) ---
def soft_dice_loss(logits, labels, eps=1e-6):
    # logits: (B,2,H,W), labels: (B,H,W)
    probs = logits.softmax(dim=1)[:, 1]  # class=1
    target = (labels == 1).float()
    num = 2 * (probs * target).sum(dim=(1, 2))
    den = (probs.pow(2) + target.pow(2)).sum(dim=(1, 2)) + eps
    return (1 - (num + eps) / (den + eps)).mean()


class DiceCETrainer(Trainer):
    """
    Custom Trainer that uses weighted CE + Dice loss.
    """

    def __init__(self, *args, ce_weight=(0.3, 0.7), dice_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        # Store as regular attributes - move to device when computing loss
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # (B, num_classes, H, W)

        # Upsample logits to match label size
        upsampled_logits = F.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )

        # Move ce_weight to the same device as logits
        device = upsampled_logits.device
        ce_class_weight = torch.tensor(
            self.ce_weight, dtype=torch.float32, device=device
        )

        # Cross-entropy loss with class weights
        ce_loss = F.cross_entropy(
            upsampled_logits, labels, weight=ce_class_weight, reduction="mean"
        )

        # Dice loss
        dice_loss = soft_dice_loss(upsampled_logits, labels)

        total_loss = ce_loss + self.dice_weight * dice_loss

        return (total_loss, outputs) if return_outputs else total_loss


def _dice_iou_from_counts(tp, fp, fn):
    iou = tp / (tp + fp + fn + EPS)
    dice = (2 * iou) / (1 + iou + EPS)
    return dice, iou


def _dilate_bool(x_bool, k: int):
    # x_bool: (B,1,H,W) bool -> bool; dilation via max_pool2d
    pad = k // 2
    return F.max_pool2d(x_bool.float(), kernel_size=k, stride=1, padding=pad) > 0.5


def _erode_bool(x_bool, k: int):
    # erosion via dilation of the inverse
    return ~_dilate_bool(~x_bool, k)


@torch.inference_mode()  # cheaper than no_grad for eval-time ops
def compute_metrics(eval_pred, chunk_size: int = 8, num_classes: int = 2, tol: int = 2):
    logits, labels = eval_pred  # logits: (B,C,h,w) np, labels: (B,H,W) np
    B, H, W = labels.shape[0], labels.shape[-2], labels.shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Accumulators kept on GPU (avoid .item() in the loop)
    tp = torch.zeros((), device=device, dtype=torch.float32)
    fp = torch.zeros((), device=device, dtype=torch.float32)
    fn = torch.zeros((), device=device, dtype=torch.float32)

    biou_vals = []  # tensors on GPU
    bf_vals = []

    for s in range(0, B, chunk_size):
        e = min(s + chunk_size, B)

        # --- move both logits and labels to GPU ---
        chunk = torch.from_numpy(logits[s:e]).to(
            device=device, dtype=torch.float32
        )  # (b,C,h,w)
        g_lab = torch.from_numpy(labels[s:e]).to(
            device=device, dtype=torch.long
        )  # (b,H,W)

        up = F.interpolate(chunk, size=(H, W), mode="bilinear", align_corners=False)
        probs = up.softmax(1)[
            :, 1
        ].float()  # (b,H,W) foreground prob (float32 for stability)
        pred1 = probs > 0.5
        gt1 = g_lab == 1

        # --- confusion counts on GPU ---
        tp += torch.count_nonzero(pred1 & gt1)
        fp += torch.count_nonzero(pred1 & ~gt1)
        fn += torch.count_nonzero(~pred1 & gt1)

        # --- Boundary IoU (1-px band via XOR with eroded mask) on GPU ---
        # edges = mask XOR erode(mask)
        p = pred1.unsqueeze(1)  # (b,1,H,W) bool
        g = gt1.unsqueeze(1)

        p_edge = p ^ _erode_bool(p, k=3)
        g_edge = g ^ _erode_bool(g, k=3)

        inter = (p_edge & g_edge).flatten(1).sum(-1).float()
        union = (p_edge | g_edge).flatten(1).sum(-1).float()
        biou = inter / (union + EPS)
        biou_vals.append(biou)

        # --- Boundary F1 with tolerance 'tol' px using dilations on GPU ---
        k = 2 * tol + 1
        g_dil = _dilate_bool(g_edge, k)
        p_dil = _dilate_bool(p_edge, k)

        tp_b = (p_edge & g_dil).flatten(1).sum(-1).float()
        fp_b = (p_edge & ~g_dil).flatten(1).sum(-1).float()
        fn_b = (g_edge & ~p_dil).flatten(1).sum(-1).float()

        prec = tp_b / (tp_b + fp_b + EPS)
        rec = tp_b / (tp_b + fn_b + EPS)
        bf = 2 * prec * rec / (prec + rec + EPS)
        bf_vals.append(bf)

        del chunk, g_lab, up, probs, pred1, gt1, p, g, p_edge, g_edge, g_dil, p_dil

    dice, iou = _dice_iou_from_counts(tp, fp, fn)

    # reduce at the very end
    boundary_iou = torch.cat(biou_vals).mean().item()
    boundary_f1 = torch.cat(bf_vals).mean().item()

    return {
        "miou": float(iou.item()),
        "dice": float(dice.item()),
        "boundary_iou": boundary_iou,
        "boundary_f1": boundary_f1,
    }


def compute_mfb_weights(mask_paths: list[Path]) -> tuple[float, float]:
    counts = {0: 0, 1: 0}
    totals_when_present = {0: 0, 1: 0}
    for mp in mask_paths:
        m = torch.load(str(mp), map_location="cpu")
        m = m["mask"] if isinstance(m, dict) and "mask" in m else m
        if m.ndim == 3:
            m = m.max(dim=0).values
        m = (m > 0.5).cpu().numpy().astype(np.uint8)
        H, W = m.shape
        for c in (0, 1):
            present = (m == c).any()
            if present:
                counts[c] += (m == c).sum()
                totals_when_present[c] += H * W

    freq = {c: counts[c] / max(totals_when_present[c], 1) for c in (0, 1)}
    median_freq = np.median(list(freq.values()))
    w = {c: median_freq / max(freq[c], 1e-9) for c in (0, 1)}
    return float(w[0]), float(w[1])


if __name__ == "__main__":
    base_images = "./automated_underwater_area_estimation/data_preprocessed/IBF/"
    base_checkpoints = (
        "./automated_underwater_area_estimation/segmentation_quadrant/checkpoints/"
    )
    OUT_IMG_DIR = Path(f"{base_images}out_images")
    OUT_MASK_DIR = Path(f"{base_images}out_masks")
    SPLITS_DIR = Path(f"{base_images}splits")
    SPLITS_DIR.mkdir(exist_ok=True, parents=True)

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
    CHECKPOINT = "nvidia/mit-b0"

    VAL_RATIO = 0.2
    SEED = 42
    IGNORE_IDX = 255
    # If you want the processor to resize to a fixed train size, set this:
    TARGET_SIZE: Tuple[int, int] | None = (800, 600)

    pairs = discover_pairs(OUT_IMG_DIR, OUT_MASK_DIR)
    print(f"Found {len(pairs)} samples in {OUT_IMG_DIR} / {OUT_MASK_DIR}")

    # Programmatic (stratified) split
    train_pairs, val_pairs = split_by_dataset_and_group(
        pairs, val_ratio=VAL_RATIO, seed=SEED
    )
    print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    # Save split files (optional, for bookkeeping)
    train_stems = [p[0].stem for p in train_pairs]
    val_stems = [p[0].stem for p in val_pairs]
    save_split(train_stems, SPLITS_DIR / "train.txt")
    save_split(val_stems, SPLITS_DIR / "val.txt")

    # Build processor & datasets
    processor = AutoImageProcessor.from_pretrained(
        CHECKPOINT, reduce_labels=False, use_fast=True
    )
    train_ds = QuadPreAugDataset(train_pairs, processor, TARGET_SIZE, train=True)
    val_ds = QuadPreAugDataset(val_pairs, processor, TARGET_SIZE, train=False)

    id2label = {0: "background", 1: "quadrant"}
    label2id = {v: k for k, v in id2label.items()}

    model = SegformerForSemanticSegmentation.from_pretrained(
        CHECKPOINT,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )
    model.config.semantic_loss_ignore_index = IGNORE_IDX
    model.config.dropout = 0.1
    model.config.classifier_dropout = 0.1

    batch_size = 24

    args = TrainingArguments(
        output_dir=f"{base_checkpoints}segformer_quad_preaug",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1,
        eval_accumulation_steps=1,
        learning_rate=6e-5,
        num_train_epochs=100,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=1_000,
        load_best_model_at_end=True,
        metric_for_best_model="miou",
        seed=SEED,
        bf16=torch.cuda.is_available(),
        save_safetensors=True,
    )

    ce_w = compute_mfb_weights([m for _, m in train_pairs])
    print("CE class weights (bg, quad):", ce_w)

    trainer = DiceCETrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor,
        compute_metrics=compute_metrics,
        ce_weight=ce_w,
        dice_weight=1.0,
    )

    trainer.train()
    final_dir = f"{base_checkpoints}segformer_best"
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
