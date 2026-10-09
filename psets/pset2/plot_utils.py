"""
Utilities to save and plot training histories (loss / accuracy / IoU curves).
"""

import json
import os

import matplotlib
matplotlib.use('Agg')  # write image files without opening a window
import matplotlib.pyplot as plt


def save_history(history, path):
    """Save a training history dict (lists of floats) to a JSON file."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"History saved to {path}")


def plot_history(history, path, title=''):
    """
    Plot train/val curves side by side: loss on the left, the metric on the right.
    Works for classification ('train_acc'/'val_acc') and segmentation ('train_iou'/'val_iou').
    """
    metric = 'acc' if 'train_acc' in history else 'iou'
    metric_label = 'Accuracy' if metric == 'acc' else 'IoU'
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(epochs, history['train_loss'], 'o-', label='Train')
    ax_loss.plot(epochs, history['val_loss'], 's-', label='Validation')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Loss')
    ax_loss.grid(alpha=0.3)
    ax_loss.legend()

    ax_metric.plot(epochs, history[f'train_{metric}'], 'o-', label='Train')
    ax_metric.plot(epochs, history[f'val_{metric}'], 's-', label='Validation')
    best = max(history[f'val_{metric}'])
    ax_metric.axhline(best, color='gray', linestyle='--', linewidth=1,
                      label=f'Best val = {best:.4f}')
    ax_metric.set_xlabel('Epoch')
    ax_metric.set_ylabel(metric_label)
    ax_metric.set_title(metric_label)
    ax_metric.grid(alpha=0.3)
    ax_metric.legend()

    if title:
        fig.suptitle(title)
    fig.tight_layout()

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Curves saved to {path}")
