"""Chronological splits, train-only scaling and rolling windows. No shuffling across time."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def generate_synthetic_weather(num_hours: int = 720, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """SYNTHETIC hourly series: temperature (24 h cycle + noise), humidity (anti-correlated), pressure (random walk).

    Returns features (n, 3) and the target (n,), which is the temperature itself.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(num_hours)
    temp = 20.0 + 8.0 * np.sin(2 * np.pi * t / 24.0) + rng.normal(0.0, 1.0, num_hours)
    humidity = np.clip(60.0 - 15.0 * np.sin(2 * np.pi * t / 24.0) + rng.normal(0.0, 3.0, num_hours), 10, 100)
    pressure = 1013.25 + np.cumsum(rng.normal(0.0, 0.1, num_hours))
    return np.column_stack([temp, humidity, pressure]), temp


def chronological_split(n: int, train: float = 0.7, val: float = 0.15) -> tuple[slice, slice, slice]:
    a, b = int(n * train), int(n * (train + val))
    return slice(0, a), slice(a, b), slice(b, n)


class Scaler:
    """Standardisation fitted on the training slice only, saved with the model."""

    def fit(self, x: np.ndarray) -> "Scaler":
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0) + 1e-8
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_

    def state(self) -> dict:
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist()}


class WindowDataset(Dataset):
    """Window [i, i+L) of features predicts the target at i+L, for every i whose target index lies in `targets`."""

    def __init__(self, features: np.ndarray, target: np.ndarray, seq_len: int, targets: range):
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(target, dtype=torch.float32)
        self.seq_len = seq_len
        self.idx = [j for j in targets if j - seq_len >= 0]

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, k: int):
        j = self.idx[k]
        return self.x[j - self.seq_len : j], self.y[j : j + 1]


def make_loaders(features_s: np.ndarray, target_s: np.ndarray, seq_len: int, splits, batch_size: int = 32, seed: int = 0):
    """Windows may look back across a split boundary (that is past data), but every target stays inside its split."""
    g = torch.Generator().manual_seed(seed)
    tr, va, te = (range(s.start, s.stop) for s in splits)
    return (
        DataLoader(WindowDataset(features_s, target_s, seq_len, tr), batch_size=batch_size, shuffle=True, generator=g),
        DataLoader(WindowDataset(features_s, target_s, seq_len, va), batch_size=256),
        DataLoader(WindowDataset(features_s, target_s, seq_len, te), batch_size=256),
    )
