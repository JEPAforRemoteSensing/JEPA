"""
Evaluation utilities for self-supervised learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast


class LinearProbe(nn.Module):
    """Linear probe classifier for evaluating representations."""

    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)


def evaluate_linear_probe(model, linear_probe, val_loader, device, use_amp=True):
    """Evaluate linear probe on validation set."""
    model.eval()
    linear_probe.eval()

    total_loss = 0
    metrics_totals = None
    num_batches = 0

    with torch.no_grad():
        for img_s1, img_s2, target in val_loader:
            img_s2 = img_s2.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with autocast("cuda", enabled=use_amp):
                features, _ = model(img_s2)
            outputs = linear_probe(features.float())
            loss = F.binary_cross_entropy_with_logits(outputs, target.float())

            total_loss += loss.item()
            
            batch_metrics = get_multilabel_metrics(outputs, target)
            if metrics_totals is None:
                metrics_totals = {k: 0.0 for k in batch_metrics.keys()}
            for k, v in batch_metrics.items():
                metrics_totals[k] += v
            num_batches += 1

    avg_metrics = {k: v / num_batches for k, v in metrics_totals.items()}
    avg_loss = total_loss / len(val_loader)

    return avg_metrics, avg_loss


def get_multilabel_metrics(outputs, targets, threshold=0.0):
    """
    Compute multi-label classification metrics.
    outputs: logits (B, C)
    targets: labels (B, C) - 0 or 1
    """
    preds = (outputs > threshold).float()
    targets = targets.float()

    # Exact Match Ratio
    # y_pred == y_true for all classes in a sample
    exact_match = (preds == targets).all(dim=1).float().mean().item()

    # Hamming Loss
    # Proportion of incorrectly predicted labels
    hamming_loss = (preds != targets).float().mean().item()

    # Hamming Score (Accuracy)
    # |yi ∩ yi^| / |yi ∪ yi^|
    intersection = (preds * targets).sum(dim=1)
    union = (preds + targets).clamp(0, 1).sum(dim=1)
    # Avoid division by zero
    accuracy = (intersection / (union + 1e-10)).mean().item()

    # Precision, Recall, F1 (samples average)
    tp = (preds * targets).sum(dim=1)
    fp = (preds * (1 - targets)).sum(dim=1)
    fn = ((1 - preds) * targets).sum(dim=1)

    precision = (tp / (tp + fp + 1e-10)).mean().item()
    recall = (tp / (tp + fn + 1e-10)).mean().item()
    f1 = (2 * tp / (2 * tp + fp + fn + 1e-10)).mean().item()

    return {
        "exact_match": exact_match,
        "hamming_loss": hamming_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
