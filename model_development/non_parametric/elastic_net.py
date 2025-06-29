import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

# -------- File Paths --------
DATA_PATH = Path("data/train_trials.csv")
MODEL_PATH = Path("models/cox_elastic_model.pkl")
SCALER_PATH = Path("models/scaler_elastic.pkl")
SUMMARY_PATH = Path("data/cox_elastic_summary.csv")
FIGURE_PATH = Path("figure/cox_elastic_forest_plot.png")
METRIC_PATH = Path("data/cox_elastic_model_metrics.txt")
FEATURES_PATH = Path("models/cox_elastic_features.pkl")

# -------- Load and Clean Data --------
def load_clean_data(path):
    df = pd.read_csv(path)
    df = df[df["trial_duration_days"].notnull()]
    df["trial_duration_days"] = df["trial_duration_days"].astype(int)
    df["status"] = df["status"].astype(int)
    return df.dropna()

def drop_non_predictors(df):
    drop_cols = ["nct_id", "condition_group_count"]
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

def drop_constant_cols(df):
    nunique = df.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    return df.drop(columns=constant_cols)

def group_rare_categories(df, threshold=0.01):
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        freq = df[col].value_counts(normalize=True)
        rare = freq[freq < threshold].index
        df[col] = df[col].replace(rare, 'Other')
    return pd.get_dummies(df, drop_first=True)

# -------- Prepare Data --------
def prepare_surv_data(df):
    y = Surv.from_dataframe("status", "trial_duration_days", df)
    X = df.drop(columns=["status", "trial_duration_days"])
    return X, y

# -------- Fit Elastic Net Cox --------
def run_elastic_net_cv(X, y):
    alphas = 10 ** np.linspace(-4, 1, 10)
    l1_ratios = [0.1, 0.5, 0.9]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", CoxnetSurvivalAnalysis(alpha_min_ratio=0.0001, max_iter=10000))
    ])

    param_grid = {
        "model__alphas": [alphas],
        "model__l1_ratio": l1_ratios
    }

    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=5,
        scoring=lambda est, X, y: concordance_index_censored(
            y["status"], y["trial_duration_days"], est.predict(X)
        )[0],
        n_jobs=-1
    )

    grid.fit(X, y)
    print(f"Best l1_ratio: {grid.best_params_['model__l1_ratio']}, Best alpha index: {grid.best_estimator_.named_steps['model'].alphas_.argmin()}")
    return grid.best_estimator_

# -------- Save Forest Plot --------
def plot_forest(model, X, path):
    coefs_2d = model.named_steps["model"].coef_
    best_alpha_index = model.named_steps["model"].alphas_.argmin()
    coefs = coefs_2d[:, best_alpha_index]
    names = X.columns
    selected = coefs != 0

    fig, ax = plt.subplots(figsize=(10, 0.4 * sum(selected)))
    ax.barh(names[selected], coefs[selected])
    ax.set_title("Elastic Net Cox Log(Hazard Ratios)")
    ax.axvline(x=0, color="gray", linestyle="--")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()

# -------- Save Metrics --------
def save_metrics(model, X, y):
    cindex = concordance_index_censored(y["status"], y["trial_duration_days"], model.predict(X))[0]
    n_events = np.sum(y["status"])
    METRIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRIC_PATH, "w") as f:
        f.write(f"C-index: {cindex:.4f}\n")
        f.write(f"Events: {n_events} / Observations: {len(y)}\n")
    print(f"Saved metrics to {METRIC_PATH}")

# -------- Main --------
if __name__ == "__main__":
    print("Loading and cleaning data...")
    df = load_clean_data(DATA_PATH)
    df = drop_non_predictors(df)
    df = group_rare_categories(df)
    df = drop_constant_cols(df)

    print("Preparing survival data...")
    X, y = prepare_surv_data(df)

    print("Training Elastic Net Cox model with CV...")
    best_model = run_elastic_net_cv(X, y)

    print("Saving model and scaler...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(best_model.named_steps["scaler"], f)

    print("Saving training features...")
    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(X.columns.tolist(), f)

    print("Saving coefficients...")
    model = best_model.named_steps["model"]
    best_alpha_index = model.alphas_.argmin()
    coefs = pd.Series(model.coef_[:, best_alpha_index], index=X.columns)
    coefs[coefs != 0].to_csv(SUMMARY_PATH)
    print(f"Saved summary to {SUMMARY_PATH}")

    print("Plotting forest plot...")
    plot_forest(best_model, X, FIGURE_PATH)

    print("Saving performance metrics...")
    save_metrics(best_model, X, y)
