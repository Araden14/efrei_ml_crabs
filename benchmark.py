import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# Data loading
# ============================================================
train_set = pd.read_csv("/home/arnaud/efrei/efrei_ml_crabs/data/train.csv")
test_set = pd.read_csv("/home/arnaud/efrei/efrei_ml_crabs/data/test.csv")

x = train_set.drop(columns=["id", "Age"])
x = pd.get_dummies(x, columns=["Sex"]).values
y = train_set["Age"].values

x_test = test_set.drop(columns=["id"])
x_test = pd.get_dummies(x_test, columns=["Sex"]).values

X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc = scaler.transform(X_val)
X_test_sc = scaler.transform(x_test)

# ============================================================
# PyTorch neural net
# ============================================================
class CrabNet(nn.Module):
    def __init__(self, input_dim, hidden_sizes):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(0.2))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_pytorch(hidden_sizes, name, epochs=200, lr=0.001, batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr = torch.tensor(X_train_sc, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_v = torch.tensor(X_val_sc, dtype=torch.float32).to(device)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    model = CrabNet(X_train_sc.shape[1], hidden_sizes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred = model(X_v).cpu().numpy()

    mae = mean_absolute_error(y_val, y_pred)
    print(f"  {name}: MAE = {mae:.4f}")
    return model


# Train only PyTorch_small and PyTorch_medium
print("Training PyTorch_small...")
py_small = train_pytorch([64, 32], "PyTorch_small")

print("Training PyTorch_medium...")
py_medium = train_pytorch([128, 64, 32], "PyTorch_medium")

# Train HistGradientBoosting
print("Training HistGradientBoosting...")
hgb_pipe = Pipeline([("scaler", StandardScaler()), ("model", HistGradientBoostingRegressor(max_iter=800, max_depth=6, learning_rate=0.05))])
hgb_pipe.fit(X_train, y_train)
hgb_val_pred = hgb_pipe.predict(X_val)
print(f"  HistGradientBoosting: MAE = {mean_absolute_error(y_val, hgb_val_pred):.4f}")

# ============================================================
# Ensemble: PySmall + PyMed + HGB
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_v_tensor = torch.tensor(X_val_sc, dtype=torch.float32).to(device)

py_small.eval()
py_medium.eval()
with torch.no_grad():
    py_small_val = py_small(X_v_tensor).cpu().numpy()
    py_medium_val = py_medium(X_v_tensor).cpu().numpy()

ensemble_val = np.mean([py_small_val, py_medium_val, hgb_val_pred], axis=0)
mae = mean_absolute_error(y_val, ensemble_val)
print(f"\nEnsemble_PySmall+PyMed+HGB: MAE = {mae:.4f}")

# ============================================================
# Generate submission on test set
# ============================================================
X_test_tensor = torch.tensor(X_test_sc, dtype=torch.float32).to(device)

with torch.no_grad():
    py_small_test = py_small(X_test_tensor).cpu().numpy()
    py_medium_test = py_medium(X_test_tensor).cpu().numpy()

hgb_test = hgb_pipe.predict(x_test)
ensemble_test = np.mean([py_small_test, py_medium_test, hgb_test], axis=0)

submission = pd.DataFrame({"id": test_set["id"], "Age": ensemble_test})
submission.to_csv("/home/arnaud/efrei/efrei_ml_crabs/data/submission.csv", index=False)
print(f"\nSubmission saved ({len(submission)} rows)")
