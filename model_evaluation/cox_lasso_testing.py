import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sksurv.metrics import concordance_index_censored

# -------- File Paths --------
TEST_PATH = Path("data/test_trials.csv")
MODEL_PATH = Path("models/cox_lasso_model.pkl")
FEATURES_PATH = Path("models/cox_lasso_features.pkl")
METRIC_PATH = Path("data/cox_lasso_test_metrics.txt")

# -------- Load Data --------
def load_clean_data(path):
    df = pd.read_csv(path)
    df = df[df["trial_duration_days"].notnull()]
    df["trial_duration_days"] = df["trial_duration_days"].astype(int)
    df["status"] = df["status"].astype(int)
    return df.dropna()

def group_rare_categories(df, threshold=0.01):
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        freq = df[col].value_counts(normalize=True)
        rare = freq[freq < threshold].index
        df[col] = df[col].replace(rare, 'Other')
    return pd.get_dummies(df, drop_first=True)

# -------- Fairness Evaluation --------
def evaluate_c_index_by_group(df_raw, df_model, risks, group_col, is_continuous=False, bins=3):
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
        cidx = concordance_index_censored(
            df_raw.loc[mask, "status"].values.astype(bool),
            df_raw.loc[mask, "trial_duration_days"].values,
            -risks[mask]
        )[0]
        results.append((group, mask.sum(), df_raw.loc[mask, "status"].sum(), cidx))
        print(f"  {str(group):20}  N={mask.sum():3}  Events={df_raw.loc[mask, 'status'].sum():3}  C-index={cidx:.4f}")
    return results

# -------- Main --------
if __name__ == "__main__":
    print("Loading test data...")
    df_raw = load_clean_data(TEST_PATH)
    df_raw = df_raw.drop(columns=["nct_id", "condition_group_count"], errors="ignore")
    df_raw = group_rare_categories(df_raw)

    print("Loading trained LASSO Cox model and feature set...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(FEATURES_PATH, "rb") as f:
        trained_features = pickle.load(f)

    print("Aligning test features with training features...")
    for col in trained_features:
        if col not in df_raw.columns:
            df_raw[col] = 0.0

    keep_cols = trained_features + ["trial_duration_days", "status"]
    df_model = df_raw[[col for col in keep_cols if col in df_raw.columns]]

    X_test = df_model[trained_features]
    y_time = df_model["trial_duration_days"].values
    y_event = df_model["status"].values.astype(bool)

    print("Predicting partial hazard and computing global C-index...")
    risks = model.predict(X_test)
    c_index = concordance_index_censored(y_event, y_time, -risks)[0]

    METRIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRIC_PATH, "w") as f:
        f.write(f"Global C-index: {c_index:.4f}\n")
        f.write(f"Events: {y_event.sum()} / Observations: {len(y_event)}\n")

    print(f"\nGlobal C-index: {c_index:.4f}")
    print(f"Events: {y_event.sum()} / Observations: {len(y_event)}")

    # -------- Fairness Analysis --------
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

    for group_col in categorical_group_cols:
        if group_col in df_raw.columns:
            results = evaluate_c_index_by_group(df_raw, df_model, risks, group_col, is_continuous=False)
            with open(METRIC_PATH, "a") as f:
                f.write(f"\nFairness Evaluation by '{group_col}':\n")
                for group, n, ev, ci in results:
                    f.write(f"  {group:20}  N={n:3}  Events={ev:3}  C-index={ci:.4f}\n")

    for group_col in continuous_group_cols:
        if group_col in df_raw.columns:
            results = evaluate_c_index_by_group(df_raw, df_model, risks, group_col, is_continuous=True, bins=3)
            with open(METRIC_PATH, "a") as f:
                f.write(f"\nFairness Evaluation by binned '{group_col}':\n")
                for group, n, ev, ci in results:
                    f.write(f"  {group:20}  N={n:3}  Events={ev:3}  C-index={ci:.4f}\n")

    print(f"\nAll results saved to: {METRIC_PATH}")
