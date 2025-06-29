import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
from sklearn.inspection import permutation_importance
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored

# -------- Paths --------
DATA_PATH = Path("data/train_trials.csv")
MODEL_PATH = Path("models/rsf_model.pkl")
SCALER_PATH = Path("models/rsf_scaler.pkl")
METRIC_PATH = Path("data/rsf_model_metrics.txt")
PARTIAL_RISK_FIGURE_PATH = Path("figure/rsf_partial_risk_plot.png")

# -------- Load & Preprocess --------
def load_clean_data(path):
    df = pd.read_csv(path)
    df = df[df["trial_duration_days"].notnull()]
    df["trial_duration_days"] = df["trial_duration_days"].astype(int)
    df["status"] = df["status"].astype(int)
    return df

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

def scale_continuous(df):
    print("Skipping scaling: data already normalized.")
    return df, None

# -------- Save Metrics --------
def save_metrics(model, X, y_struct):
    METRIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    pred = model.predict(X)
    ci = concordance_index_censored(y_struct["status"], y_struct["trial_duration_days"], pred)[0]
    with open(METRIC_PATH, "w") as f:
        f.write(f"C-index: {ci:.4f}\n")
        f.write(f"n_estimators: {model.n_estimators}\n")
        f.write(f"Min Samples Split: {model.min_samples_split}\n")
        f.write(f"Min Samples Leaf: {model.min_samples_leaf}\n")
        f.write(f"Max Features: {model.max_features}\n")
    print("\n RSF Model Metrics:")
    print(f"C-index: {ci:.4f}")
    print(f"Metrics saved to: {METRIC_PATH}")

# -------- Feature Importance --------
def select_top_predictors(model, X, y_struct, top_n=9):
    print("Downsampling and calculating permutation importances...")

    sample_idx = X.sample(n=5000, random_state=42).index
    X_small = X.loc[sample_idx]
    y_small = y_struct[sample_idx]

    result = permutation_importance(
        model, X_small, y_small,
        n_repeats=1,
        random_state=42,
        n_jobs=1
    )

    importances = pd.Series(result.importances_mean, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(top_n).index.tolist()
    print("Top Predictors:", top_features)
    return top_features


# -------- Partial Risk Plot --------
def plot_partial_risk_by_top_predictors(model, X_df, y_df, selected_vars):
    print("Generating RSF risk score plots...")
    partial_risk = model.predict(X_df)
    df = X_df.copy()
    df["partial_risk"] = partial_risk

    n_plots = len(selected_vars)
    cols = 3
    rows = int(np.ceil(n_plots / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows))
    axs = axs.flatten()
    colors = ['#1f77b4', '#ff7f0e']

    for i, var in enumerate(selected_vars):
        ax = axs[i]
        values = sorted(df[var].dropna().unique())
        is_binary = df[var].nunique() == 2

        if is_binary:
            for j, val in enumerate(values):
                group = df[df[var] == val]
                label = f"{var} = {int(val)} (n={len(group)})"
                ax.hist(group["partial_risk"], bins=30, alpha=0.6, label=label,
                        color=colors[j % len(colors)], edgecolor='black', linewidth=0.5)
            title = f"{var} (binary)"
        else:
            ax.hist(df["partial_risk"], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            title = f"{var} (continuous)"

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Predicted Risk", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True)

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.suptitle("RSF Partial Risk Score Distributions", fontsize=14, y=1.02)
    PARTIAL_RISK_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PARTIAL_RISK_FIGURE_PATH, bbox_inches="tight")
    plt.close()
    print(f"Partial risk plot saved to: {PARTIAL_RISK_FIGURE_PATH}")

# -------- Individual Survival Curve Plot --------
def plot_individual_survival_curves(model, X_sample, ids=None, n_curves=5):
    print("Plotting individual survival curves...")
    surv_fns = model.predict_survival_function(X_sample)
    time_grid = np.linspace(0, 1000, 100)

    plt.figure(figsize=(8, 6))
    for i, fn in enumerate(surv_fns[:n_curves]):
        plt.step(time_grid, fn(time_grid), where="post", label=f"Sample {ids[i] if ids is not None else i}")
    plt.title("Predicted Survival Curves")
    plt.xlabel("Time (days)")
    plt.ylabel("Survival Probability")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -------- Main --------
if __name__ == "__main__":
    print("Loading and preprocessing...")
    df = load_clean_data(DATA_PATH)
    df = drop_non_predictors(df)
    df = group_rare_categories(df)
    df = df.dropna()
    df = drop_constant_cols(df)
    df, scaler = scale_continuous(df)

    X = df.drop(columns=["trial_duration_days", "status"])
    y_struct = Surv.from_dataframe("status", "trial_duration_days", df)

    print("Training Random Survival Forest with best hyperparameters...")
    rsf = RandomSurvivalForest(
        n_estimators=100,
        max_features=0.5,
        min_samples_leaf=20,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42
    )
    rsf.fit(X, y_struct)

    print("Saving model and scaler...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rsf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("Saving metrics...")
    save_metrics(rsf, X, y_struct)

    try:
        print("Selecting top predictors...")
        top_vars = select_top_predictors(rsf, X, y_struct, top_n=9)
        print("Plotting partial risk distributions...")
        plot_partial_risk_by_top_predictors(rsf, X, df[["status", "trial_duration_days"]], top_vars)
    except Exception as e:
        print(f"Skipping top predictor plots due to error: {e}")

    print("Plotting individual survival curves...")
    plot_individual_survival_curves(rsf, X.iloc[:5], ids=X.index[:5], n_curves=5)

    print("All done.")
