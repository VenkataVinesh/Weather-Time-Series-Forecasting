"""Train the LSTM on the SYNTHETIC series and compare it with baselines on a held-out test split.

    python train.py                  # 3 seeds, writes loss_curve.png, forecast_test.png and results.json
    python train.py --seeds 1 --epochs 5
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from baselines import persistence, sarima_one_step, seasonal_naive
from data_loader import Scaler, chronological_split, generate_synthetic_weather, make_loaders
from lstm_model import WeatherForecastingLSTM

SEQ_LEN = 24


def train_one(seed: int, epochs: int, features_s, target_s, splits):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_dl, val_dl, test_dl = make_loaders(features_s, target_s, SEQ_LEN, splits, seed=seed)
    model = WeatherForecastingLSTM(input_dim=features_s.shape[1], hidden_dim=32, num_layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.L1Loss()
    hist, best, best_state = {"train": [], "val": []}, float("inf"), None
    for _ in range(epochs):
        model.train()
        tot = 0.0
        for x, y in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
        hist["train"].append(tot / len(train_dl.dataset))
        model.eval()
        with torch.no_grad():
            val = float(np.mean([loss_fn(model(x), y).item() for x, y in val_dl]))
        hist["val"].append(val)
        if val < best:  # keep the epoch with the best validation loss; the test split is never used to choose
            best, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = torch.cat([model(x) for x, _ in test_dl]).squeeze(1).numpy()
    return preds, hist, test_dl.dataset.idx, model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()

    features, target = generate_synthetic_weather()
    splits = chronological_split(len(target))
    fx, ty = Scaler().fit(features[splits[0]]), Scaler().fit(target[splits[0], None])
    features_s, target_s = fx.transform(features), ty.transform(target[:, None])[:, 0]

    runs = [train_one(s, args.epochs, features_s, target_s, splits) for s in range(args.seeds)]
    idx = np.array(runs[0][2])
    y_true = target[idx]
    unscale = lambda p: p * ty.std_[0] + ty.mean_[0]
    lstm_mae = [float(np.mean(np.abs(unscale(r[0]) - y_true))) for r in runs]

    results = {
        "data": "synthetic (data_loader.generate_synthetic_weather, seed 42)",
        "test_points": int(len(idx)),
        "mae_degC": {
            "persistence": float(np.mean(np.abs(persistence(target, idx) - y_true))),
            "seasonal_naive_24h": float(np.mean(np.abs(seasonal_naive(target, idx) - y_true))),
            "sarima_100_100_24": float(np.mean(np.abs(sarima_one_step(target, splits[0].stop, idx) - y_true))),
            "lstm_mean_over_seeds": float(np.mean(lstm_mae)),
            "lstm_std_over_seeds": float(np.std(lstm_mae)),
        },
        "noise_floor_degC": "temperature noise is N(0, 1), so even a perfect model has MAE ≈ 0.80",
        "seeds": args.seeds,
        "epochs": args.epochs,
    }
    print(json.dumps(results, indent=2))
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    torch.save({"state_dict": runs[0][3].state_dict(), "feature_scaler": fx.state(), "target_scaler": ty.state(), "seq_len": SEQ_LEN}, "lstm.pt")

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, r in enumerate(runs):
        ax.plot(r[1]["train"], color="#2b6cb0", alpha=0.6, label="train" if i == 0 else None)
        ax.plot(r[1]["val"], color="#c05621", alpha=0.6, label="validation" if i == 0 else None)
    ax.set_title(f"LSTM loss (standardised MAE), {args.seeds} seeds, synthetic data")
    ax.set_xlabel("epoch"); ax.set_ylabel("MAE (standardised)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("loss_curve.png", dpi=120)

    fig, ax = plt.subplots(figsize=(10, 4))
    k = slice(0, 72)
    ax.plot(y_true[k], color="black", lw=1.4, label="actual")
    ax.plot(unscale(runs[0][0])[k], color="#2b6cb0", label=f"LSTM (MAE {lstm_mae[0]:.2f})")
    ax.plot(seasonal_naive(target, idx)[k], color="#999", ls="--", label=f"seasonal naive (MAE {results['mae_degC']['seasonal_naive_24h']:.2f})")
    ax.set_title("First 72 test hours, one step ahead (synthetic temperature, °C)")
    ax.set_xlabel("test hour"); ax.set_ylabel("°C"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("forecast_test.png", dpi=120)


if __name__ == "__main__":
    main()
