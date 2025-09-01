import torch


def _to_probs(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 4 and logits.size(1) == 1:
        return torch.sigmoid(logits)
    elif logits.ndim == 4 and logits.size(1) > 1:
        return torch.softmax(logits, dim=1)
    else:
        return torch.sigmoid(logits)


def binary_iou(
    pred_logits: torch.Tensor, target_mask: torch.Tensor, threshold: float = 0.5
) -> float:
    """Compute IoU for binary segmentation with batched support.

    - pred_logits: shape [B, 1, H, W] or [B, H, W]
    - target_mask: shape [B, H, W] or [H, W] with {0,1}
    - When union == 0 (no positive in pred and target), returns 1.0 by convention.
    """
    if pred_logits.ndim == 3:  # [B, H, W]
        probs = torch.sigmoid(pred_logits)
    elif pred_logits.ndim == 4 and pred_logits.size(1) == 1:
        probs = torch.sigmoid(pred_logits[:, 0])
    else:
        raise ValueError("binary_iou expects logits with channel dim=1 or none")

    if target_mask.ndim == 2:
        target = target_mask.unsqueeze(0)
    else:
        target = target_mask

    preds = (probs >= threshold).to(torch.long)
    target = target.to(torch.long)

    # Flatten per-batch
    preds = preds.reshape(preds.size(0), -1)
    target = target.reshape(target.size(0), -1)

    intersection = (preds & target).sum(dim=1).to(torch.float)
    union = (preds | target).sum(dim=1).to(torch.float)

    iou = torch.where(union == 0, torch.ones_like(union), intersection / union)
    return float(iou.mean().item())
