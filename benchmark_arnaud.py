"""
Optuna benchmark — tunes LightGBM (CPU), XGBoost (CUDA GPU), and blend weights.
Saves best submission to submission_tuned.csv
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Data & features ───────────────────────────────────────────────────────────
train = pd.read_csv("./data/train.csv")
test  = pd.read_csv("./data/test.csv")

def safe_div(a, b):
    return np.where(b == 0, 0.0, a / b)

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

train = engineer(pd.get_dummies(train, columns=["Sex"]))
test  = engineer(pd.get_dummies(test,  columns=["Sex"]))

FEATURES = [c for c in train.columns if c not in ("id", "Age")]
X      = train[FEATURES].values.astype(np.float32)
y      = train["Age"].values.astype(np.float32)
X_test = test[FEATURES].values.astype(np.float32)

KF = KFold(n_splits=5, shuffle=True, random_state=42)

def cv_mae(model, use_log=True):
    oof = np.zeros(len(X))
    for tr_idx, val_idx in KF.split(X):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr        = np.log1p(y[tr_idx]) if use_log else y[tr_idx]
        model.fit(X_tr, y_tr)
        pred = np.expm1(model.predict(X_val)) if use_log else model.predict(X_val)
        oof[val_idx] = pred
    return mean_absolute_error(y, np.round(oof)), oof

# ── LightGBM study ────────────────────────────────────────────────────────────
def lgbm_objective(trial):
    params = dict(
        n_estimators      = trial.suggest_int  ("n_est",      500,  4000),
        learning_rate     = trial.suggest_float("lr",         0.003, 0.05, log=True),
        num_leaves        = trial.suggest_int  ("leaves",     31,   255),
        max_depth         = trial.suggest_int  ("depth",      4,    12),
        min_child_samples = trial.suggest_int  ("min_child",  5,    100),
        subsample         = trial.suggest_float("subsample",  0.5,  1.0),
        colsample_bytree  = trial.suggest_float("colsample",  0.5,  1.0),
        reg_alpha         = trial.suggest_float("alpha",      1e-3, 10.0, log=True),
        reg_lambda        = trial.suggest_float("lambda",     1e-3, 10.0, log=True),
        random_state=42, verbose=-1,
    )
    mae, _ = cv_mae(LGBMRegressor(**params))
    return mae

print("=" * 55)
print("Tuning LightGBM (CPU) — 60 trials …")
lgbm_study = optuna.create_study(direction="minimize",
                                  sampler=optuna.samplers.TPESampler(seed=42))
lgbm_study.optimize(lgbm_objective, n_trials=60,
                    show_progress_bar=True, n_jobs=1)

best_lgbm_params = {**lgbm_study.best_params, "random_state": 42, "verbose": -1}
print(f"Best LGBM MAE : {lgbm_study.best_value:.4f}")
print(f"Best LGBM params: {best_lgbm_params}")

# ── XGBoost study (GPU) ───────────────────────────────────────────────────────
def xgb_objective(trial):
    params = dict(
        n_estimators     = trial.suggest_int  ("n_est",      500,  4000),
        learning_rate    = trial.suggest_float("lr",         0.003, 0.05, log=True),
        max_depth        = trial.suggest_int  ("depth",      4,    10),
        min_child_weight = trial.suggest_int  ("min_child",  1,    20),
        subsample        = trial.suggest_float("subsample",  0.5,  1.0),
        colsample_bytree = trial.suggest_float("colsample",  0.5,  1.0),
        reg_alpha        = trial.suggest_float("alpha",      1e-3, 10.0, log=True),
        reg_lambda       = trial.suggest_float("lambda",     1e-3, 10.0, log=True),
        gamma            = trial.suggest_float("gamma",      0,    5.0),
        device="cuda", verbosity=0, random_state=42,
    )
    mae, _ = cv_mae(XGBRegressor(**params))
    return mae

print("=" * 55)
print("Tuning XGBoost (CUDA GPU) — 60 trials …")
xgb_study = optuna.create_study(direction="minimize",
                                  sampler=optuna.samplers.TPESampler(seed=42))
xgb_study.optimize(xgb_objective, n_trials=60,
                   show_progress_bar=True, n_jobs=1)

best_xgb_params = {**xgb_study.best_params,
                   "device": "cuda", "verbosity": 0, "random_state": 42}
print(f"Best XGB MAE : {xgb_study.best_value:.4f}")
print(f"Best XGB params: {best_xgb_params}")

# ── Blend weight study ────────────────────────────────────────────────────────
print("=" * 55)
print("Computing OOF predictions with best models …")
_, lgbm_oof = cv_mae(LGBMRegressor(**best_lgbm_params))
_, xgb_oof  = cv_mae(XGBRegressor(**best_xgb_params))

def blend_objective(trial):
    w = trial.suggest_float("w_lgbm", 0.0, 1.0)
    blend = w * lgbm_oof + (1 - w) * xgb_oof
    return mean_absolute_error(y, np.round(blend))

print("Tuning blend weights — 50 trials …")
blend_study = optuna.create_study(direction="minimize",
                                   sampler=optuna.samplers.TPESampler(seed=42))
blend_study.optimize(blend_objective, n_trials=50, show_progress_bar=True)

w_lgbm = blend_study.best_params["w_lgbm"]
w_xgb  = 1 - w_lgbm
print(f"Best blend: LGBM={w_lgbm:.3f}  XGB={w_xgb:.3f}")
print(f"Best blend MAE: {blend_study.best_value:.4f}")

# ── Final submission ──────────────────────────────────────────────────────────
print("=" * 55)
print("Training final models on full data …")

lgbm_test = np.zeros(len(X_test))
xgb_test  = np.zeros(len(X_test))

for tr_idx, _ in KF.split(X):
    lgbm_m = LGBMRegressor(**best_lgbm_params)
    lgbm_m.fit(X[tr_idx], np.log1p(y[tr_idx]))
    lgbm_test += np.expm1(lgbm_m.predict(X_test)) / 5

    xgb_m = XGBRegressor(**best_xgb_params)
    xgb_m.fit(X[tr_idx], np.log1p(y[tr_idx]))
    xgb_test += np.expm1(xgb_m.predict(X_test)) / 5

test_blend  = w_lgbm * lgbm_test + w_xgb * xgb_test
predictions = np.clip(np.round(test_blend), 1, None).astype(int)

submission = pd.DataFrame({"id": test["id"], "Age": predictions})
submission.to_csv("./submission_tuned.csv", index=False)

print(f"\n{'='*55}")
print(f"FINAL OOF MAE : {blend_study.best_value:.4f}")
print(f"Saved → submission_tuned.csv")
