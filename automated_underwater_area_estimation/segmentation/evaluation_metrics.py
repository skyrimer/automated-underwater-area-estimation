import torch
from typing import Dict, Union


def _prepare_masks(
    pred_mask: torch.Tensor, gt_mask: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Helper function to prepare masks for computation."""
    return pred_mask.bool().to(device), gt_mask.bool().to(device)


def compute_segmentation_metrics(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    device: torch.device,
    smooth: float = 1e-6,
    metrics: Union[str, list[str]] = "all",
) -> Union[float, Dict[str, float]]:
    """
    Compute segmentation metrics for binary masks efficiently.

    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        device: Device to perform computations on
        smooth: Smoothing factor to avoid division by zero
        metrics: Either "all" for all metrics, single metric name, or list of metric names
                Available: ["dice", "iou", "precision", "recall", "pixel_accuracy"]

    Returns:
        float if single metric requested, dict if multiple metrics
    """
    # Validate metrics parameter
    available_metrics = {"dice", "iou", "precision", "recall", "pixel_accuracy"}
    if isinstance(metrics, str):
        if metrics == "all":
            requested_metrics = available_metrics
        elif metrics in available_metrics:
            requested_metrics = {metrics}
        else:
            raise ValueError(
                f"Invalid metric '{metrics}'. Available: {available_metrics}"
            )
    elif isinstance(metrics, list):
        requested_metrics = set(metrics)
        invalid = requested_metrics - available_metrics
        if invalid:
            raise ValueError(
                f"Invalid metrics {invalid}. Available: {available_metrics}"
            )
    else:
        raise ValueError("metrics must be 'all', a string, or a list of strings")

    # Prepare masks
    pred_mask, gt_mask = _prepare_masks(pred_mask, gt_mask, device)

    # Compute base components only once
    tp = torch.logical_and(pred_mask, gt_mask).sum().float()  # True Positives

    results = {}

    # Compute only requested metrics
    if "dice" in requested_metrics:
        pred_sum = pred_mask.sum().float()
        gt_sum = gt_mask.sum().float()
        results["dice"] = ((2.0 * tp + smooth) / (pred_sum + gt_sum + smooth)).item()

    if "iou" in requested_metrics:
        union = torch.logical_or(pred_mask, gt_mask).sum().float()
        results["iou"] = ((tp + smooth) / (union + smooth)).item()

    if "precision" in requested_metrics:
        fp = torch.logical_and(pred_mask, ~gt_mask).sum().float()  # False Positives
        results["precision"] = ((tp + smooth) / (tp + fp + smooth)).item()

    if "recall" in requested_metrics:
        fn = torch.logical_and(~pred_mask, gt_mask).sum().float()  # False Negatives
        results["recall"] = ((tp + smooth) / (tp + fn + smooth)).item()

    if "pixel_accuracy" in requested_metrics:
        correct = torch.eq(pred_mask, gt_mask).sum().float()
        total = pred_mask.numel()
        results["pixel_accuracy"] = (correct / total).item()

    # Return single value if only one metric requested
    if len(results) == 1:
        return next(iter(results.values()))

    return results


# Convenience functions for backward compatibility and specific use cases
def compute_dice(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    device: torch.device,
    smooth: float = 1e-6,
) -> float:
    """Compute Dice coefficient (F1-score) for binary masks."""
    return compute_segmentation_metrics(pred_mask, gt_mask, device, smooth, "dice")


def compute_iou(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    device: torch.device,
    smooth: float = 1e-6,
) -> float:
    """Compute Intersection over Union (IoU) for binary masks."""
    return compute_segmentation_metrics(pred_mask, gt_mask, device, smooth, "iou")


def compute_precision(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    device: torch.device,
    smooth: float = 1e-6,
) -> float:
    """Compute Precision for binary masks."""
    return compute_segmentation_metrics(pred_mask, gt_mask, device, smooth, "precision")


def compute_recall(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    device: torch.device,
    smooth: float = 1e-6,
) -> float:
    """Compute Recall (Sensitivity) for binary masks."""
    return compute_segmentation_metrics(pred_mask, gt_mask, device, smooth, "recall")


def compute_pixel_accuracy(
    pred_mask: torch.Tensor, gt_mask: torch.Tensor, device: torch.device
) -> float:
    """Compute Pixel Accuracy for binary masks."""
    return compute_segmentation_metrics(
        pred_mask, gt_mask, device, metrics="pixel_accuracy"
    )


def compute_all_metrics(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    device: torch.device,
    smooth: float = 1e-6,
) -> Dict[str, float]:
    """Compute all segmentation metrics at once for efficiency."""
    return compute_segmentation_metrics(pred_mask, gt_mask, device, smooth, "all")
