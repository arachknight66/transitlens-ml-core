"""Detected-ephemeris dual-view representation and optional compact 1-D CNN."""
from __future__ import annotations
import numpy as np

def phase_view(time, flux, period, epoch, length: int, phase_min=-0.5, phase_max=0.5):
    if period <= 0 or len(time) != len(flux):
        raise ValueError("a positive detected period and aligned arrays are required")
    phase = ((np.asarray(time) - epoch + period / 2) % period) / period - 0.5
    edges = np.linspace(phase_min, phase_max, length + 1)
    index = np.digitize(phase, edges) - 1
    values, mask = np.zeros(length, np.float32), np.zeros(length, np.float32)
    for i in range(length):
        selected = np.asarray(flux)[index == i]
        selected = selected[np.isfinite(selected)]
        if selected.size:
            values[i] = np.median(selected) - 1.0
            mask[i] = 1.0
    return np.stack([values, mask])

def build_dual_views(time, flux, period, epoch, duration, global_length=2001, local_length=201):
    width = max(2.5 * duration / period, 0.02)
    return {
        "global": phase_view(time, flux, period, epoch, global_length),
        "local": phase_view(time, flux, period, epoch, local_length, -width, width),
    }

def create_torch_model(num_classes: int = 4):
    """Create the dual-branch CNN lazily so tabular-only installs remain usable."""
    import torch
    from torch import nn
    class Branch(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Conv1d(2, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(4),
                                     nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(8))
        def forward(self, x): return self.net(x).flatten(1)
    class DualViewCNN(nn.Module):
        def __init__(self):
            super().__init__(); self.global_branch = Branch(); self.local_branch = Branch()
            self.head = nn.Sequential(nn.Linear(512, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, num_classes))
        def forward(self, global_view, local_view):
            return self.head(torch.cat([self.global_branch(global_view), self.local_branch(local_view)], dim=1))
    return DualViewCNN()
