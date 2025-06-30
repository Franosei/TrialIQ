import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored

# -------- File Paths --------
TEST_PATH = Path("data/test_trials.csv")
MODEL_PATH = Path("models/rsf_model.pkl")
METRIC_PATH = Path("data/rsf_model_test_metrics.txt")

# -------- Load Data --------
def load_clean_data(path):
    df = pd.read_csv(path)
    df = df[df["trial_duration_days"].notnull()]
    df["trial_duration_days"] = df["trial_duration_days"].astype(int)
    df["status"] = df["status"].astype(int)
    df = df.drop(columns=["nct_id", "condition_group_count"], errors="ignore")
    return df

# -------- Rare Category Grouping and One-Hot Encoding --------
def group_rare_categories(df, threshold=0.01):
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        freq = df[col].value_counts(normalize=True)
        rare = freq[freq < threshold].index
        df[col] = df[col].replace(rare, "Other")
    return pd.get_dummies(df, drop_first=True)

# -------- Fairness Evaluation --------
def evaluate_c_index_by_group(df_raw, X, model, group_col, is_continuous=False, bins=3):
    print(f"\nFairness Evaluation by '{group_col}':")
    results = []

    if is_continuous:
        labels = ["Low", "Mid", "High"][:bins]
        try:
            df_raw[f"{group_col}_binned"] = pd.qcut(df_raw[group_col], q=bins, labels=labels, duplicates="drop")
        except Exception as e:
            print(f"  Skipping {group_col} — binning failed: {e}")
            return []
        eval_col = f"{group_col}_binned"
    else:
        eval_col = group_col

    for group in df_raw[eval_col].dropna().unique():
        mask = df_raw[eval_col] == group
        if mask.sum() < 10:
            continue
        y_group = Surv.from_dataframe("status", "trial_duration_days", df_raw[mask])
        X_group = X.loc[mask]
        preds = model.predict(X_group)
        ci_result = concordance_index_censored(
            y_group["status"],
            y_group["trial_duration_days"],
            preds
        )
        c_index = ci_result[0]
        results.append((group, mask.sum(), df_raw.loc[mask, "status"].sum(), c_index))
        print(f"  {str(group):20}  N={mask.sum():3}  Events={df_raw.loc[mask, 'status'].sum():3}  C-index={c_index:.4f}")
    return results

# -------- Main Evaluation --------
if __name__ == "__main__":
    print("Loading and preprocessing test data...")
    df_raw = load_clean_data(TEST_PATH)

    print("Applying rare category grouping and encoding...")
    df_model = group_rare_categories(df_raw.copy())

    print("Loading trained RSF model...")
    model = joblib.load(MODEL_PATH)
    model_features = model.feature_names_in_

    print("Aligning test features to model features...")
    for col in model_features:
        if col not in df_model.columns:
            df_model[col] = 0.0
    X_test = df_model[model_features]

    # Sync df_raw to match model input rows
    df_raw = df_raw.loc[df_model.index]

    print("Preparing structured labels...")
    y_test = Surv.from_dataframe("status", "trial_duration_days", df_raw)

    print("Making predictions and computing global C-index...")
    preds = model.predict(X_test)
    c_index_result = concordance_index_censored(
        y_test["status"],
        y_test["trial_duration_days"],
        preds
    )
    c_index = c_index_result[0]
    concordant = c_index_result[1]
    comparable = c_index_result[4]

    num_events = df_raw["status"].sum()
    num_total = df_raw.shape[0]

    print(f"\nGlobal C-index: {c_index:.4f}")
    print(f"Concordant pairs: {concordant} / Comparable pairs: {comparable}")
    print(f"Events: {num_events} / Observations: {num_total}")

    # -------- Save Global Metrics --------
    METRIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRIC_PATH, "w") as f:
        f.write(f"Global C-index: {c_index:.4f}\n")
        f.write(f"Concordant pairs: {concordant} / Comparable pairs: {comparable}\n")
        f.write(f"Events: {num_events} / Observations: {num_total}\n")

    # -------- Fairness Evaluation Lists --------
    categorical_group_cols = [
        "sponsor_class_category_industry",
        "is_fda_regulated_drug_true",
        "c08",
        "has_expanded_access_true",
        "intervention_model_category_parallel",
        "intervention_model_category_crossover",
        "intervention_grouped_category_behavioral/lifestyle",
        "is_fda_regulated_drug_none"
    ]

    continuous_group_cols = [
        "site_count", 
        "maximum_age",
        "collaborator_count",
        "arm_count",
        "eligibility_token_count",
        "exclusion_criteria_count",
        "secondary_outcome_count"
    ]

    # -------- Evaluate C-index by Categorical Groups --------
    for group_col in categorical_group_cols:
        if group_col in df_raw.columns:
            results = evaluate_c_index_by_group(df_raw, X_test, model, group_col, is_continuous=False)
            with open(METRIC_PATH, "a") as f:
                f.write(f"\nFairness Evaluation by '{group_col}':\n")
                for group, n, ev, ci in results:
                    f.write(f"  {group:20}  N={n:3}  Events={ev:3}  C-index={ci:.4f}\n")

    # -------- Evaluate C-index by Binned Continuous Groups --------
    for group_col in continuous_group_cols:
        if group_col in df_raw.columns:
            results = evaluate_c_index_by_group(df_raw, X_test, model, group_col, is_continuous=True, bins=3)
            with open(METRIC_PATH, "a") as f:
                f.write(f"\nFairness Evaluation by binned '{group_col}':\n")
                for group, n, ev, ci in results:
                    f.write(f"  {group:20}  N={n:3}  Events={ev:3}  C-index={ci:.4f}\n")

    print(f"\nAll results saved to: {METRIC_PATH}")
