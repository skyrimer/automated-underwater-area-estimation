from pathlib import Path
from typing import List, Dict, Any, Tuple
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from transformers import AutoImageProcessor, SegformerForSemanticSegmentation, TrainingArguments, Trainer
import torch.nn.functional as F
import evaluate

# ----------------------
# CONFIG
# ----------------------
base = "./automated_underwater_area_estimation/data_preprocessed/IBF/"
OUT_IMG_DIR = Path(f"{base}out_images")
OUT_MASK_DIR = Path(f"{base}out_masks")
SPLITS_DIR   = Path(f"{base}splits")   # will be created; txt files saved here
SPLITS_DIR.mkdir(exist_ok=True, parents=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
# CHECKPOINT = "nvidia/segformer-b5-finetuned-ade-640-640"  # change to B4/B3 if VRAM is tight
CHECKPOINT = "nvidia/mit-b0"  # change to B4/B3 if VRAM is tight

VAL_RATIO   = 0.2
SEED        = 42
IGNORE_IDX  = 255
# If you want the processor to resize to a fixed train size, set this:
TARGET_SIZE: Tuple[int,int] | None = (800, 600)  # (W,H) or None to let processor decide


def load_mask_pt(path: str) -> torch.Tensor:
    """Return torch.bool (H,W)."""
    data = torch.load(path, map_location="cpu")
    t = data["mask"] if isinstance(data, dict) and "mask" in data else data
    if not isinstance(t, torch.Tensor):
        raise ValueError(f"Unsupported mask in {path}")
    if t.ndim == 3 and t.shape[0] in (1,3):
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


def stratified_split_on_foreground(pairs: List[Tuple[Path,Path]], val_ratio=0.2, seed=42):
    """
    Simple stratification: split by whether mask has any foreground (True).
    """
    rng = random.Random(seed)
    pos, neg = [], []
    for img_p, mask_p in pairs:
        mask = load_mask_pt(str(mask_p))
        (pos if mask.any().item() else neg).append((img_p, mask_p))

    def split_bucket(bucket):
        rng.shuffle(bucket)
        n_val = max(1, int(round(len(bucket)*val_ratio))) if len(bucket) > 0 else 0
        return bucket[n_val:], bucket[:n_val]

    train_pos, val_pos = split_bucket(pos)
    train_neg, val_neg = split_bucket(neg)

    train = train_pos + train_neg
    val   = val_pos + val_neg
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def save_split(stems: List[str], path: Path):
    path.write_text("\n".join(stems) + "\n", encoding="utf-8")


class QuadPreAugDataset(Dataset):
    """
    Dataset that reads pre-augmented files from disk (no runtime Albumentations).
    Uses HF processor for normalization and (optional) resize.
    """
    def __init__(self, items: List[Tuple[Path,Path]], processor, target_size: Tuple[int,int] | None, train: bool):
        self.items = items
        self.processor = processor
        self.target_size = target_size
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx) -> Dict[str, Any]:
        img_path, mask_path = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        mask = load_mask_pt(str(mask_path)).to(torch.int32)  # (H,W) 0/1

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
            encoded = self.processor(images=img, segmentation_maps=mask.numpy(), return_tensors="pt")

        return {k: v.squeeze(0) for k, v in encoded.items()}

# Discover paired samples
pairs = discover_pairs(OUT_IMG_DIR, OUT_MASK_DIR)
print(f"Found {len(pairs)} samples in {OUT_IMG_DIR} / {OUT_MASK_DIR}")

# Programmatic (stratified) split
train_pairs, val_pairs = stratified_split_on_foreground(pairs, val_ratio=VAL_RATIO, seed=SEED)
print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)}")

# Save split files (optional, for bookkeeping)
train_stems = [p[0].stem for p in train_pairs]
val_stems   = [p[0].stem for p in val_pairs]
save_split(train_stems, SPLITS_DIR / "train.txt")
save_split(val_stems,   SPLITS_DIR / "val.txt")

# Build processor & datasets
processor = AutoImageProcessor.from_pretrained(CHECKPOINT, reduce_labels=False, use_fast=True)
train_ds  = QuadPreAugDataset(train_pairs, processor, TARGET_SIZE, train=True)
val_ds    = QuadPreAugDataset(val_pairs,   processor, TARGET_SIZE, train=False)

id2label = {0: "background", 1: "quadrant"}
label2id = {v:k for k,v in id2label.items()}

model = SegformerForSemanticSegmentation.from_pretrained(
    CHECKPOINT,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)
model.config.semantic_loss_ignore_index = IGNORE_IDX
model.config.dropout = 0.1
model.config.classifier_dropout = 0.1

# --- Dice + CE (foreground-aware) ---
def soft_dice_loss(logits, labels, eps=1e-6):
    # logits: (B,2,H,W), labels: (B,H,W)
    probs = logits.softmax(dim=1)[:, 1]  # class=1
    target = (labels == 1).float()
    num = 2 * (probs * target).sum(dim=(1,2))
    den = (probs.pow(2) + target.pow(2)).sum(dim=(1,2)) + eps
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

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # (B, num_classes, H, W)

        # Upsample logits to match label size
        upsampled_logits = F.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # Move ce_weight to the same device as logits
        device = upsampled_logits.device
        ce_class_weight = torch.tensor(self.ce_weight, dtype=torch.float32, device=device)

        # Cross-entropy loss with class weights
        ce_loss = F.cross_entropy(
            upsampled_logits,
            labels,
            weight=ce_class_weight,
            reduction="mean"
        )

        # Dice loss
        dice_loss = soft_dice_loss(upsampled_logits, labels)

        total_loss = ce_loss + self.dice_weight * dice_loss

        return (total_loss, outputs) if return_outputs else total_loss
# --- Metrics: mIoU / mF1 ---
metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits_t = torch.from_numpy(logits).float()
    # Upsample to label size in case processor resized differently
    preds = F.interpolate(
        logits_t, size=labels.shape[-2:], mode="bilinear", align_corners=False
    ).argmax(1).cpu().numpy().astype(np.int32)

    if res := metric.compute(
        predictions=list(preds),
        references=list(labels.astype(np.int32)),
        num_labels=2,
        ignore_index=IGNORE_IDX,
        reduce_labels=False,
    ):
        return {
            "miou": res["mean_iou"],
            "maccuracy": res["mean_accuracy"],
            "iou_bg": res["per_category_iou"][0],
            "iou_quad": res["per_category_iou"][1],
            "accuracy_bg": res["per_category_accuracy"][0],
            "accuracy_quad": res["per_category_accuracy"][1],
        }
    else:
        raise ValueError("Empty metric result")

args = TrainingArguments(
    output_dir="segformer_quad_preaug",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,  # effective batch 4
    eval_accumulation_steps = 1,
    learning_rate=6e-5,
    num_train_epochs=5,           # pre-aug data typically needs fewer epochs; tune as needed
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="miou",
    seed=SEED,
    bf16=torch.cuda.is_available(),  # or fp16=True
    save_safetensors=True,
)

trainer = DiceCETrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=processor,   # satisfies Trainer API
    compute_metrics=compute_metrics,
    ce_weight=(0.3, 0.7),  # ↑ weight for quadrant if it’s small
    dice_weight=1.0,
)

trainer.train()
final_dir = "segformer_best"
trainer.save_model(final_dir)          # saves the *current* model = best
processor.save_pretrained(final_dir)