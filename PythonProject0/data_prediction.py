"""
Module 3: Demand Prediction (Zone × Hour)
- Aggregate trips into (date, hour, zone) demand series
- Build features, split 8:2
- Train MLP (PyTorch) vs RandomForest
- Plot loss curve, report MAE / RMSE, compare
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_pipeline import run_pipeline   # Module 1

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)


# ============ 1. Build (zone, date, hour) demand table ============
def build_demand_table(df: pd.DataFrame, top_k_zones: int = 30) -> pd.DataFrame:
    """
    Aggregate trip records into demand counts per (zone, date, hour).
    Limit to top-K busiest zones to keep the problem tractable & balanced.
    """
    df = df.copy()
    df["date"] = df["tpep_pickup_datetime"].dt.date

    top_zones = df["PULocationID"].value_counts().head(top_k_zones).index
    df = df[df["PULocationID"].isin(top_zones)]

    demand = (df.groupby(["PULocationID", "date", "pickup_hour"])
                .size().reset_index(name="demand"))

    # Re-derive temporal features
    demand["date"] = pd.to_datetime(demand["date"])
    demand["weekday"]      = demand["date"].dt.weekday
    demand["is_weekend"]   = (demand["weekday"] >= 5).astype(int)
    demand["is_rush_hour"] = (
        (demand["is_weekend"] == 0) &
        (demand["pickup_hour"].isin([7, 8, 9, 17, 18, 19]))
    ).astype(int)
    # Cyclic encoding of hour & weekday (helps NN learn periodicity)
    demand["hour_sin"] = np.sin(2 * np.pi * demand["pickup_hour"] / 24)
    demand["hour_cos"] = np.cos(2 * np.pi * demand["pickup_hour"] / 24)
    demand["wday_sin"] = np.sin(2 * np.pi * demand["weekday"] / 7)
    demand["wday_cos"] = np.cos(2 * np.pi * demand["weekday"] / 7)
    return demand


# ============ 2. PyTorch MLP ============
class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),     nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_tr, y_tr, X_te, y_te, epochs=40, batch=512, lr=1e-3):
    """Train MLP, return predictions on test set + loss history."""
    tr_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                          torch.tensor(y_tr, dtype=torch.float32))
    te_ds = TensorDataset(torch.tensor(X_te, dtype=torch.float32),
                          torch.tensor(y_te, dtype=torch.float32))
    tr_dl = DataLoader(tr_ds, batch_size=batch, shuffle=True)
    te_dl = DataLoader(te_ds, batch_size=batch)

    model  = MLP(X_tr.shape[1]).to(DEVICE)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    loss_f = nn.MSELoss()

    hist = {"train": [], "test": []}
    for ep in range(epochs):
        model.train(); tr_loss = 0
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_f(model(xb), yb)
            loss.backward(); opt.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(tr_ds)

        model.eval(); te_loss = 0
        with torch.no_grad():
            for xb, yb in te_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                te_loss += loss_f(model(xb), yb).item() * len(xb)
        te_loss /= len(te_ds)

        hist["train"].append(tr_loss); hist["test"].append(te_loss)
        if (ep + 1) % 5 == 0:
            print(f"  epoch {ep+1:02d}  train MSE={tr_loss:.3f}  test MSE={te_loss:.3f}")

    # Predict
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_te, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    return preds, hist


# ============ 3. Plot loss curve ============
def plot_loss(hist: dict, name="05_mlp_loss_curve.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hist["train"], label="Train MSE", color="steelblue")
    ax.plot(hist["test"],  label="Test MSE",  color="tomato")
    ax.set(xlabel="Epoch", ylabel="MSE Loss",
           title="MLP Training & Test Loss Curve")
    ax.legend()
    fig.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {os.path.join(OUT_DIR, name)}")


# ============ 4. Comparison plot ============
def plot_comparison(y_true, mlp_pred, rf_pred,
                    name="05_pred_vs_true.png"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, pred, title, c in [
        (axes[0], mlp_pred, "MLP: Predicted vs Actual",          "steelblue"),
        (axes[1], rf_pred,  "RandomForest: Predicted vs Actual", "seagreen"),
    ]:
        ax.scatter(y_true, pred, alpha=0.3, s=8, color=c)
        m = max(y_true.max(), pred.max())
        ax.plot([0, m], [0, m], "r--", lw=1)
        ax.set(xlabel="Actual Demand", ylabel="Predicted Demand", title=title)
    fig.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {os.path.join(OUT_DIR, name)}")


def plot_metrics_bar(metrics: dict, name="05_metrics_bar.png"):
    models = list(metrics.keys())
    mae    = [metrics[m]["MAE"]  for m in models]
    rmse   = [metrics[m]["RMSE"] for m in models]
    x = np.arange(len(models)); w = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - w/2, mae,  w, label="MAE",  color="steelblue")
    ax.bar(x + w/2, rmse, w, label="RMSE", color="tomato")
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("Error"); ax.set_title("MLP vs RandomForest: MAE & RMSE")
    for i, (a, b) in enumerate(zip(mae, rmse)):
        ax.text(i - w/2, a, f"{a:.2f}", ha="center", va="bottom")
        ax.text(i + w/2, b, f"{b:.2f}", ha="center", va="bottom")
    ax.legend()
    fig.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {os.path.join(OUT_DIR, name)}")


# ============ 5. Main pipeline ============
def main():
    # ---- data ----
    df = run_pipeline("yellow_tripdata_2026-01.parquet")
    demand = build_demand_table(df)
    print(f"Demand samples: {len(demand)}")

    feat_cols = ["PULocationID", "pickup_hour", "weekday",
                 "is_weekend", "is_rush_hour",
                 "hour_sin", "hour_cos", "wday_sin", "wday_cos"]
    X = demand[feat_cols].values
    y = demand["demand"].values.astype(np.float32)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED)

    # ---- MLP needs scaling; RF does not ----
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    print("\n[Train MLP]")
    mlp_pred, hist = train_mlp(X_tr_s, y_tr, X_te_s, y_te, epochs=40)
    plot_loss(hist)

    print("\n[Train RandomForest]")
    rf = RandomForestRegressor(n_estimators=200, max_depth=None,
                               n_jobs=-1, random_state=SEED)
    rf.fit(X_tr, y_tr)
    rf_pred = rf.predict(X_te)

    # ---- metrics ----
    def evalp(name, pred):
        mae  = mean_absolute_error(y_te, pred)
        rmse = np.sqrt(mean_squared_error(y_te, pred))
        print(f"{name:>14s}  MAE={mae:.3f}  RMSE={rmse:.3f}")
        return {"MAE": mae, "RMSE": rmse}

    print("\n=== Test Set Performance ===")
    metrics = {"MLP": evalp("MLP", mlp_pred),
               "RandomForest": evalp("RandomForest", rf_pred)}

    plot_comparison(y_te, mlp_pred, rf_pred)
    plot_metrics_bar(metrics)

    # ---- discussion ----
    print("\n=== Comparison Discussion ===")
    print("""
RandomForest:
  + No feature scaling needed; handles non-linearity & feature interactions natively.
  + Robust to outliers; faster to tune (few hyper-params).
  + Usually wins on small/medium tabular data like this aggregated demand task.
  - Hard to extrapolate beyond training range; model size grows with data.

MLP (Neural Network):
  + Flexible; easily extended to sequence models (LSTM/Transformer) for time series.
  + Benefits from cyclic encodings & large data.
  - Needs scaling, more epochs, more hyper-parameter tuning.
  - With limited tabular features, often slightly worse than tree ensembles.

Conclusion:
  For this static (zone, hour) demand regression with few features,
  RandomForest typically achieves comparable or lower MAE/RMSE with less effort.
  MLP becomes more attractive when adding richer features (weather, lagged demand,
  embeddings of zone IDs) or moving to a temporal sequence model.
""")


if __name__ == "__main__":
    main()