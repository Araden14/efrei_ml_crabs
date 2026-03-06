import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# Data loading + feature engineering
# ============================================================
train_set = pd.read_csv("/home/arnaud/efrei/efrei_ml_crabs/data/train.csv")
test_set = pd.read_csv("/home/arnaud/efrei/efrei_ml_crabs/data/test.csv")


def add_features(df):
    df = df.copy()
    df["Volume"] = df["Length"] * df["Diameter"] * df["Height"]
    df["LxD"] = df["Length"] * df["Diameter"]
    df["Shell_ratio"] = df["Shell Weight"] / (df["Weight"] + 1e-6)
    df["Shucked_ratio"] = df["Shucked Weight"] / (df["Weight"] + 1e-6)
    df["Viscera_ratio"] = df["Viscera Weight"] / (df["Weight"] + 1e-6)
    df["Meat_weight"] = df["Weight"] - df["Shell Weight"] - df["Viscera Weight"]
    df["Weight_per_Vol"] = df["Weight"] / (df["Volume"] + 1e-6)
    df["Shell_per_Vol"] = df["Shell Weight"] / (df["Volume"] + 1e-6)
    df["Shucked_per_Length"] = df["Shucked Weight"] / (df["Length"] + 1e-6)
    df["Weight_sq"] = df["Weight"] ** 2
    df["Height_sq"] = df["Height"] ** 2
    df["Log_Weight"] = np.log1p(df["Weight"])
    df["Log_Shell"] = np.log1p(df["Shell Weight"])
    return df


train_feat = add_features(train_set)
test_feat = add_features(test_set)

x = train_feat.drop(columns=["id", "Age"])
x = pd.get_dummies(x, columns=["Sex"]).values
y = train_feat["Age"].values

x_test = test_feat.drop(columns=["id"])
x_test = pd.get_dummies(x_test, columns=["Sex"]).values

X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc = scaler.transform(X_val)
X_test_sc = scaler.transform(x_test)

print(f"Features: {X_train_sc.shape[1]}")

# ============================================================
# PyTorch net
# ============================================================
class CrabNet(nn.Module):
    def __init__(self, input_dim, hidden_sizes, dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_pytorch(hidden_sizes, name, epochs=300, lr=0.001, batch_size=256, dropout=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr = torch.tensor(X_train_sc, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_v = torch.tensor(X_val_sc, dtype=torch.float32).to(device)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    model = CrabNet(X_train_sc.shape[1], hidden_sizes, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.L1Loss()

    best_mae = float("inf")
    best_state = None

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Track best
        model.eval()
        with torch.no_grad():
            val_pred = model(X_v).cpu().numpy()
        val_mae = mean_absolute_error(y_val, val_pred)
        if val_mae < best_mae:
            best_mae = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.train()

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_pred = model(X_v).cpu().numpy()

    mae = mean_absolute_error(y_val, y_pred)
    print(f"  {name}: MAE = {mae:.4f}")
    return model


# ============================================================
# Train models
# ============================================================
# Multiple PyTorch with different architectures (diversity helps ensemble)
print("Training PyTorch models...")
py1 = train_pytorch([64, 32], "PyTorch_64_32", epochs=300, dropout=0.15)
py2 = train_pytorch([128, 64, 32], "PyTorch_128_64_32", epochs=300, dropout=0.15)
py3 = train_pytorch([128, 64], "PyTorch_128_64", epochs=300, dropout=0.2)
py4 = train_pytorch([256, 128, 64], "PyTorch_256_128_64", epochs=300, dropout=0.2)

# Tree models with tuned params
print("\nTraining tree models...")

hgb1 = HistGradientBoostingRegressor(max_iter=1000, max_depth=6, learning_rate=0.03, l2_regularization=0.1, min_samples_leaf=10)
hgb1.fit(X_train_sc, y_train)
print(f"  HGB_v1: MAE = {mean_absolute_error(y_val, hgb1.predict(X_val_sc)):.4f}")

hgb2 = HistGradientBoostingRegressor(max_iter=1200, max_depth=5, learning_rate=0.05, l2_regularization=0.05, min_samples_leaf=20)
hgb2.fit(X_train_sc, y_train)
print(f"  HGB_v2: MAE = {mean_absolute_error(y_val, hgb2.predict(X_val_sc)):.4f}")

xgb1 = XGBRegressor(n_estimators=800, max_depth=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, verbosity=0, n_jobs=-1)
xgb1.fit(X_train_sc, y_train)
print(f"  XGB_v1: MAE = {mean_absolute_error(y_val, xgb1.predict(X_val_sc)):.4f}")

# ============================================================
# Ensemble with optimized weights
# ============================================================
print("\nBuilding ensemble...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_v_tensor = torch.tensor(X_val_sc, dtype=torch.float32).to(device)

all_val_preds = {}
for name, model in [("py1", py1), ("py2", py2), ("py3", py3), ("py4", py4)]:
    model.eval()
    with torch.no_grad():
        all_val_preds[name] = model(X_v_tensor).cpu().numpy()

all_val_preds["hgb1"] = hgb1.predict(X_val_sc)
all_val_preds["hgb2"] = hgb2.predict(X_val_sc)
all_val_preds["xgb1"] = xgb1.predict(X_val_sc)

# Simple average
simple_avg = np.mean(list(all_val_preds.values()), axis=0)
print(f"  Simple average (7 models): MAE = {mean_absolute_error(y_val, simple_avg):.4f}")

# Optimized weights via scipy
from scipy.optimize import minimize

preds_matrix = np.array(list(all_val_preds.values()))  # (n_models, n_samples)


def ensemble_mae(weights):
    w = np.array(weights)
    w = w / w.sum()
    blended = (w[:, None] * preds_matrix).sum(axis=0)
    return mean_absolute_error(y_val, blended)


n_models = len(all_val_preds)
res = minimize(ensemble_mae, x0=np.ones(n_models) / n_models,
               bounds=[(0, 1)] * n_models, method="Nelder-Mead")

best_weights = np.array(res.x)
best_weights = best_weights / best_weights.sum()

weighted_avg = (best_weights[:, None] * preds_matrix).sum(axis=0)
print(f"  Weighted average (optimized): MAE = {mean_absolute_error(y_val, weighted_avg):.4f}")
print(f"  Weights: {dict(zip(all_val_preds.keys(), [f'{w:.3f}' for w in best_weights]))}")

# ============================================================
# Generate submission
# ============================================================
X_test_tensor = torch.tensor(X_test_sc, dtype=torch.float32).to(device)

all_test_preds = {}
for name, model in [("py1", py1), ("py2", py2), ("py3", py3), ("py4", py4)]:
    model.eval()
    with torch.no_grad():
        all_test_preds[name] = model(X_test_tensor).cpu().numpy()

all_test_preds["hgb1"] = hgb1.predict(X_test_sc)
all_test_preds["hgb2"] = hgb2.predict(X_test_sc)
all_test_preds["xgb1"] = xgb1.predict(X_test_sc)

test_matrix = np.array(list(all_test_preds.values()))
final_pred = (best_weights[:, None] * test_matrix).sum(axis=0)

submission = pd.DataFrame({"id": test_set["id"], "Age": final_pred.round().astype(int)})
submission.to_csv("/home/arnaud/efrei/efrei_ml_crabs/data/submission_v2.csv", index=False)
print(f"\nSubmission saved ({len(submission)} rows)")
