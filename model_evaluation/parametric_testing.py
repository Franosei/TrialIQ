import pandas as pd
import numpy as np
from lifelines.utils import concordance_index
from pathlib import Path
import joblib

# -------- File Paths --------
TEST_PATH = Path("data/test_trials.csv")
MODEL_PATH = Path("models/parametric_model.pkl")
FEATURES_PATH = Path("models/parametric_features.pkl")
METRIC_PATH = Path("data/parametric_model_test_metrics.txt")

# -------- Load Data --------
def load_clean_data(path):
    df = pd.read_csv(path)
    df = df[df["trial_duration_days"].notnull()]
    df["trial_duration_days"] = df["trial_duration_days"].astype(int)
    df["status"] = df["status"].astype(int)
    df = df.drop(columns=["nct_id", "condition_group_count"], errors="ignore")
    return df

# -------- Match Cox Preprocessing: Group Rare and One-Hot Encode --------
def group_rare_categories(df, threshold=0.01):
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        freq = df[col].value_counts(normalize=True)
        rare = freq[freq < threshold].index
        df[col] = df[col].replace(rare, 'Other')
    return pd.get_dummies(df, drop_first=True)

# -------- Align Test Features --------
def align_and_filter(df, model_features):
    for col in model_features:
        if col not in df.columns:
            df[col] = 0.0
    keep_cols = model_features + ["trial_duration_days", "status"]
    df = df[keep_cols]
    df = df.dropna()
    return df

# -------- Fairness Evaluation --------
def evaluate_c_index_by_group(df_raw, preds, group_col, is_continuous=False, bins=3):
    print(f"\nFairness Evaluation by '{group_col}':")
    results = []

    if is_continuous:
        labels = ["Low", "Mid", "High"][:bins]
        try:
            df_raw[f"{group_col}_binned"] = pd.qcut(df_raw[group_col], q=bins, labels=labels, duplicates='drop')
        except Exception as e:
            print(f"  Skipping {group_col} — binning failed: {e}")
            return []
        eval_col = f"{group_col}_binned"
    else:
        eval_col = group_col

    unique_groups = df_raw[eval_col].dropna().unique()
    for group in unique_groups:
        mask = df_raw[eval_col] == group
        if mask.sum() < 10:
            continue
        ci = concordance_index(
            df_raw.loc[mask, "trial_duration_days"],
            -preds[mask],
            df_raw.loc[mask, "status"]
        )
        results.append((group, mask.sum(), df_raw.loc[mask, "status"].sum(), ci))
        print(f"  {str(group):20}  N={mask.sum():3}  Events={df_raw.loc[mask, 'status'].sum():3}  C-index={ci:.4f}")
    return results

# -------- Main Evaluation --------
if __name__ == "__main__":
    print("Loading test data...")
    df_raw = load_clean_data(TEST_PATH)
    df_raw = group_rare_categories(df_raw)
    df_raw = df_raw.dropna()  # Same as in Cox pipeline

    print("Loading trained parametric model...")
    model = joblib.load(MODEL_PATH)

    print("Loading feature list used during training...")
    model_features = joblib.load(FEATURES_PATH)

    print("Aligning and filtering test data to match model features...")
    df_model = align_and_filter(df_raw.copy(), model_features)

    print("Predicting and evaluating concordance index...")
    try:
        preds = model.predict_median(df_model)
        c_index = concordance_index(
            df_model["trial_duration_days"],
            -preds,
            df_model["status"]
        )
    except Exception as e:
        print(f"Model prediction failed: {e}")
        exit(1)

    num_events = df_model["status"].sum()
    num_total = df_model.shape[0]

    print(f"\nGlobal C-index: {c_index:.4f}")
    print(f"Events: {num_events} / Observations: {num_total}")

    METRIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRIC_PATH, "w") as f:
        f.write(f"Global C-index: {c_index:.4f}\n")
        f.write(f"Events: {num_events} / Observations: {num_total}\n")

    # -------- Fairness Evaluation --------
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

    # Filter to match df_model rows
    df_eval = df_raw.loc[df_model.index]

    for group_col in categorical_group_cols:
        if group_col in df_eval.columns:
            results = evaluate_c_index_by_group(df_eval, preds, group_col, is_continuous=False)
            with open(METRIC_PATH, "a") as f:
                f.write(f"\nFairness Evaluation by '{group_col}':\n")
                for group, n, ev, ci in results:
                    f.write(f"  {group:20}  N={n:3}  Events={ev:3}  C-index={ci:.4f}\n")

    for group_col in continuous_group_cols:
        if group_col in df_eval.columns:
            results = evaluate_c_index_by_group(df_eval, preds, group_col, is_continuous=True, bins=3)
            with open(METRIC_PATH, "a") as f:
                f.write(f"\nFairness Evaluation by binned '{group_col}':\n")
                for group, n, ev, ci in results:
                    f.write(f"  {group:20}  N={n:3}  Events={ev:3}  C-index={ci:.4f}\n")

    print(f"\nAll results saved to: {METRIC_PATH}")
