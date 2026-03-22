import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# ── Data ──────────────────────────────────────────────────────────────────────
train = pd.read_csv("./data/train.csv")
test  = pd.read_csv("./data/test.csv")

# ── Feature engineering ───────────────────────────────────────────────────────
def safe_div(a, b, fill=0.0):
    result = np.where(b == 0, fill, a / b)
    return result

def engineer(df):
    df = df.copy()
    df['Shell_to_Weight']   = safe_div(df['Shell Weight'],   df['Weight'])
    df['Meat_Ratio']        = safe_div(df['Shucked Weight'], df['Weight'])
    df['Viscera_Ratio']     = safe_div(df['Viscera Weight'], df['Weight'])
    df['Volume']            = df['Length'] * df['Diameter'] * df['Height']
    df['Shell_per_Volume']  = safe_div(df['Shell Weight'],   df['Volume'])
    df['Weight_per_Length'] = safe_div(df['Weight'],         df['Length'])
    df['Length_Diam_Ratio'] = safe_div(df['Length'],         df['Diameter'])
    df['BMI_like']          = safe_div(df['Weight'],         df['Height'] ** 2)
    return df

train = engineer(train)
test  = engineer(test)

# One-hot encode Sex
train = pd.get_dummies(train, columns=["Sex"])
test  = pd.get_dummies(test,  columns=["Sex"])

FEATURES = [c for c in train.columns if c not in ("id", "Age")]
X = train[FEATURES].values
y = train["Age"].values
X_test = test[FEATURES].values

# ── Models (tuned params from Optuna benchmark) ───────────────────────────────
lgbm = LGBMRegressor(
    n_estimators=668,
    learning_rate=0.0074716584813256225,
    max_depth=5,
    num_leaves=110,
    min_child_samples=69,
    subsample=0.8894270564243927,
    colsample_bytree=0.6631403893424325,
    reg_alpha=0.14264129526401326,
    reg_lambda=0.050520349860097236,
    random_state=42,
    verbose=-1,
)

xgb = XGBRegressor(
    n_estimators=1960,
    learning_rate=0.0070750481126885005,
    max_depth=9,
    min_child_weight=18,
    subsample=0.635246439833654,
    colsample_bytree=0.6911230131567632,
    reg_alpha=0.14146472001394517,
    reg_lambda=0.21111820229878614,
    gamma=0.1450981803676941,
    device="cuda",
    random_state=42,
    verbosity=0,
)

et = ExtraTreesRegressor(
    n_estimators=800,
    max_depth=None,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
)

# Weights: Optuna found XGB>>LGBM for 2-model blend.
# Add ET with small weight to decorrelate errors.
MODELS = [
    ("lgbm", lgbm, 0.35),
    ("xgb",  xgb,  0.55),
    ("et",   et,   0.10),
]

# ── 5-fold OOF training ───────────────────────────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds   = {name: np.zeros(len(X))         for name, *_ in MODELS}
test_preds  = {name: np.zeros(len(X_test))    for name, *_ in MODELS}

for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    log_y_tr    = np.log1p(y_tr)

    for name, model, _ in MODELS:
        if name in ("lgbm", "xgb"):
            model.fit(X_tr, log_y_tr)
            oof_preds[name][val_idx] = np.expm1(model.predict(X_val))
            test_preds[name]        += np.expm1(model.predict(X_test)) / 5
        else:
            model.fit(X_tr, y_tr)
            oof_preds[name][val_idx] = model.predict(X_val)
            test_preds[name]        += model.predict(X_test) / 5

    # per-fold MAE for monitoring
    blend_fold = sum(
        w * oof_preds[n][val_idx] for n, _, w in MODELS
    )
    mae_fold = mean_absolute_error(y_val, np.round(blend_fold))
    print(f"Fold {fold+1} MAE (rounded): {mae_fold:.4f}")

# ── Blend & evaluate ──────────────────────────────────────────────────────────
oof_blend  = sum(w * oof_preds[n]  for n, _, w in MODELS)
test_blend = sum(w * test_preds[n] for n, _, w in MODELS)

mae_raw     = mean_absolute_error(y, oof_blend)
mae_rounded = mean_absolute_error(y, np.round(oof_blend))
print(f"\nOOF MAE (raw):     {mae_raw:.4f}")
print(f"OOF MAE (rounded): {mae_rounded:.4f}")

# Per-model MAE for reference
for name, _, _ in MODELS:
    m = mean_absolute_error(y, np.round(oof_preds[name]))
    print(f"  {name} alone: {m:.4f}")

# ── Submission ────────────────────────────────────────────────────────────────
predictions = np.clip(np.round(test_blend), 1, None).astype(int)
submission  = pd.DataFrame({"id": test["id"], "Age": predictions})
submission.to_csv("./submission_ensemble.csv", index=False)
print("\nSaved submission_ensemble.csv")
