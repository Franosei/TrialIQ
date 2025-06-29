import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.utils import k_fold_cross_validation
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler

# -------- File Paths --------
DATA_PATH = Path("data/train_trials.csv")
MODEL_PATH = Path("models/cox_model.pkl")
SCALER_PATH = Path("models/scaler.pkl")
SUMMARY_PATH = Path("data/cox_model.csv")
FIGURE_PATH = Path("figure/cox_forest_plot.png")
METRIC_PATH = Path("data/cox_model_metrics.txt")
PARTIAL_RISK_FIGURE_PATH = Path("figure/cox_partial_risk_plot.png")

# -------- Load and Clean Data --------
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
    return df

def scale_continuous(df):
    # Data is already normalized; skip scaling.
    print("Skipping scaling: data already normalized.")
    return df, None

# -------- Penalizer Tuning --------
def find_best_penalizer(df, candidate_penalties=[0.001, 0.01, 0.1, 0.5, 1.0]):
    best_score, best_penalty = -np.inf, None
    for p in candidate_penalties:
        cph = CoxPHFitter(penalizer=p)
        try:
            scores = k_fold_cross_validation(
                cph, df,
                duration_col="trial_duration_days",
                event_col="status",
                k=5,
                scoring_method="concordance_index"
            )
            avg_score = np.mean(scores)
            print(f" Penalizer={p:.3f}, C-index={avg_score:.4f}")
            if avg_score > best_score:
                best_score = avg_score
                best_penalty = p
        except Exception as e:
            print(f" Penalizer={p:.3f} failed: {e}")
    return best_penalty, best_score

# -------- Fit Cox Model --------
def run_cox_model(df, penalizer):
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df, duration_col="trial_duration_days", event_col="status", show_progress=True)
    return cph

# -------- Save Forest Plot --------
def plot_forest(cph):
    fig = plt.figure(figsize=(10, 0.3 * len(cph.params_) + 1))
    ax = fig.add_subplot(111)
    cph.plot(ax=ax)
    plt.title("Cox Proportional Hazards — Log(Hazard Ratios)")
    plt.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_PATH)
    plt.close()

# -------- Save Model Metrics --------
def save_metrics(cph, df):
    METRIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRIC_PATH, "w") as f:
        f.write(f"C-index: {cph.concordance_index_:.4f}\n")
        f.write(f"Partial log-likelihood: {cph.log_likelihood_:.2f}\n")
        f.write(f"AIC (partial): {cph.AIC_partial_:.2f}\n")
        f.write(f"Events: {df['status'].sum()} / Observations: {df.shape[0]}\n")

    print("\n Model Performance Metrics:")
    print(f"C-index: {cph.concordance_index_:.4f}")
    print(f"Partial log-likelihood: {cph.log_likelihood_:.2f}")
    print(f"AIC (partial): {cph.AIC_partial_:.2f}")
    print(f"Events: {df['status'].sum()} / Observations: {df.shape[0]}")
    print(f"Metrics saved to: {METRIC_PATH}")

# -------- Mixed Predictor Selector --------
def select_mixed_predictors(df, cph, top_cat=6, top_cont=3, min_group_size=100):
    summary = cph.summary
    all_vars = summary.index.tolist()
    categorical_vars = []
    continuous_vars = []

    for var in all_vars:
        if var not in df.columns:
            continue
        if df[var].nunique() == 2:
            counts = df[var].value_counts()
            if all(counts >= min_group_size):
                categorical_vars.append((var, abs(summary.loc[var, "coef"])))
        elif df[var].nunique() > 10:
            continuous_vars.append((var, abs(summary.loc[var, "coef"])))

    top_cats = [v for v, _ in sorted(categorical_vars, key=lambda x: x[1], reverse=True)[:top_cat]]
    top_conts = [v for v, _ in sorted(continuous_vars, key=lambda x: x[1], reverse=True)[:top_cont]]

    return top_cats + top_conts

# -------- Plot Partial Hazard Distributions --------
def plot_partial_risk_by_top_predictors(cph, df, selected_vars):
    print("Generating partial risk score plots...")
    df = df.copy()
    df["partial_risk"] = cph.predict_partial_hazard(df)

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
            counts = df[var].value_counts().to_dict()
            for j, val in enumerate(values):
                group = df[df[var] == val]
                label = f"{var} = {int(val)} (n={len(group)})"
                ax.hist(
                    group["partial_risk"],
                    bins=30,
                    alpha=0.6,
                    label=label,
                    color=colors[j % len(colors)],
                    edgecolor='black',
                    linewidth=0.5,
                    histtype='stepfilled'
                )
            title = f"{var} (0: {counts.get(0,0)}, 1: {counts.get(1,0)})"
        else:
            ax.hist(df["partial_risk"], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            title = f"{var} (Continuous)"

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Partial Hazard", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True)

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.suptitle("Partial Hazard Score Distributions for Top Predictors", fontsize=14, y=1.02)
    PARTIAL_RISK_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PARTIAL_RISK_FIGURE_PATH, bbox_inches="tight")
    plt.close()
    print(f"Balanced partial risk figure saved to: {PARTIAL_RISK_FIGURE_PATH}")

# -------- Main Pipeline --------
if __name__ == "__main__":
    print("Loading trial data...")
    df = load_clean_data(DATA_PATH)
    df = drop_non_predictors(df)

    print("Grouping rare categorical levels...")
    df = group_rare_categories(df)

    print("Dropping missing and constant columns...")
    df = df.dropna()
    df = drop_constant_cols(df)

    print("Scaling continuous variables...")
    df, scaler = scale_continuous(df)

    print("Tuning penalizer via cross-validation...")
    penalizer, best_score = find_best_penalizer(df)
    print(f"Best penalizer: {penalizer:.3f} with C-index: {best_score:.4f}")

    print(f"Training Cox model on shape: {df.shape}")
    cph = run_cox_model(df, penalizer)

    print("Saving model and results...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(cph, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    cph.summary.to_csv(SUMMARY_PATH)
    plot_forest(cph)
    save_metrics(cph, df)

    selected_vars = select_mixed_predictors(df, cph, top_cat=6, top_cont=3)
    plot_partial_risk_by_top_predictors(cph, df, selected_vars)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Summary saved to: {SUMMARY_PATH}")
    print(f"Forest plot saved to: {FIGURE_PATH}")
