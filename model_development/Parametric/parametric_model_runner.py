import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
from lifelines import WeibullAFTFitter, LogLogisticAFTFitter, LogNormalAFTFitter

# ---------------- File Paths ----------------
DATA_PATH = Path("data/train_trials.csv")
MODEL_SAVE = Path("models/parametric_model.pkl")
SUMMARY_PATH = Path("data/parametric_model_summary.csv")
COMPARISON_PATH = Path("data/parametric_model_comparison.csv")
FIGURE_PATH = Path("figure/parametric_stratified_grid.png")

# ---------------- Load & Preprocess ----------------
def load_and_preprocess(path):
    df = pd.read_csv(path)
    before_all = len(df)
    df = df.dropna(subset=["trial_duration_days", "status"])
    df = df[df["trial_duration_days"] > 0].copy()
    df = df.drop(columns=["nct_id"], errors="ignore")
    df["status"] = df["status"].astype(int)

    print(f"Initial rows after dropna & filtering: {len(df)} (from original {before_all})")
    return df

# ---------------- Model Training with Penalizer Tuning ----------------
def train_with_penalizers(df, model_class, name, penalizers=[0.001, 0.010, 0.100, 0.500, 1.000]):
    best_model = None
    best_result = None
    best_cindex = -np.inf

    for pen in penalizers:
        model = model_class(penalizer=pen)
        try:
            model.fit(df, duration_col="trial_duration_days", event_col="status")
            cindex = model.score(df, scoring_method='concordance_index')
            result = {
                "model": name,
                "penalizer": pen,
                "aic": model.AIC_,
                "log_likelihood": model.log_likelihood_,
                "concordance_index": cindex
            }
            print(f"{name} (penalizer={pen:.4f}): AIC={model.AIC_:.2f}, CI={cindex:.3f}")
            if cindex > best_cindex:
                best_model = model
                best_result = result
                best_cindex = cindex
        except Exception as e:
            print(f"{name} (penalizer={pen:.4f}) failed: {e}")
    
    return best_model, best_result

# ---------------- Train All Models ----------------
def train_aft_models(df):
    model_classes = {
        "weibull": WeibullAFTFitter,
        "loglogistic": LogLogisticAFTFitter,
        "lognormal": LogNormalAFTFitter
    }

    all_results = []
    fitted_models = {}

    for name, cls in model_classes.items():
        model, result = train_with_penalizers(df, cls, name)
        if model:
            fitted_models[name] = model
            all_results.append(result)
    
    result_df = pd.DataFrame(all_results)
    result_df.to_csv(COMPARISON_PATH, index=False)
    return fitted_models, result_df

# ---------------- Plot Stratified Survival Curves ----------------
def plot_stratified_survival_grid_parametric(model, df, duration_col, event_col, top_n=9):
    summary = model.summary.copy()
    try:
        lambda_df = summary.loc["lambda_"]
        lambda_df = lambda_df.drop(index="Intercept", errors="ignore")
        lambda_df["abs_coef"] = lambda_df["coef"].abs()
        top_features = lambda_df.sort_values("abs_coef", ascending=False).head(top_n).index.tolist()
    except:
        print("Failed to extract lambda_ terms from summary.")
        return

    rows, cols = 3, 3
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for i, feat in enumerate(top_features):
        if i >= rows * cols: break
        row, col = divmod(i, cols)
        ax = axs[row][col]

        try:
            if df[feat].nunique() <= 1:
                ax.set_title(f"{feat} (no variation)")
                ax.axis('off')
                continue

            if df[feat].nunique() <= 4:
                df['risk_group'] = df[feat].astype(str)
            else:
                df['risk_group'] = pd.qcut(df[feat], q=4, duplicates='drop')

            times = np.linspace(0, df[duration_col].max(), 1000)

            for group in df['risk_group'].unique():
                subset = df[df['risk_group'] == group]
                if len(subset) < 5: continue
                surv = model.predict_survival_function(subset, times=times)
                mean_surv = surv.mean(axis=1)
                ax.plot(mean_surv.index, mean_surv.values, label=str(group))

            ax.set_title(f"Partial Risk by '{feat}'")
            ax.set_xlabel("Time (days)")
            ax.set_ylabel("Survival Probability")
            ax.grid(True)
            ax.legend(fontsize=8)
        except Exception as e:
            ax.set_title(f"{feat} (Error)")
            ax.axis('off')
            print(f"Skipped {feat}: {e}")

    for j in range(i + 1, rows * cols):
        r, c = divmod(j, cols)
        fig.delaxes(axs[r][c])

    plt.tight_layout()
    plt.suptitle("Stratified Survival Curves by Top Predictors (Parametric)", fontsize=14, y=1.02)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_PATH, bbox_inches='tight')
    plt.close()

# ---------------- Main ----------------
if __name__ == "__main__":
    print("Loading and preprocessing data...")
    df = load_and_preprocess(DATA_PATH)

    print("Fitting AFT models with penalizer tuning...")
    fitted_models, result_df = train_aft_models(df)

    if result_df.empty:
        print("No model trained successfully. Exiting.")
        exit(1)

    best_model_name = result_df.sort_values("aic").iloc[0]['model']
    best_model = fitted_models[best_model_name]
    print(f"Best model: {best_model_name}")

    print("Saving model and summary...")
    MODEL_SAVE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_SAVE)
    best_model.summary.to_csv(SUMMARY_PATH)

    print("Generating stratified survival plots...")
    plot_stratified_survival_grid_parametric(best_model, df, "trial_duration_days", "status")

    print(f"Done. Best model: {best_model_name}, plot saved to: {FIGURE_PATH}")
