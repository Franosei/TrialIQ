import pandas as pd
import numpy as np
from lifelines.utils import concordance_index
from pathlib import Path
import pickle

# -------- File Paths --------
TEST_PATH = Path("data/test_trials.csv")
MODEL_PATH = Path("models/cox_model.pkl")
METRIC_PATH = Path("data/cox_model_test_metrics.txt")

# -------- Load Data --------
def load_clean_data(path):
    df = pd.read_csv(path)
    df = df[df["trial_duration_days"].notnull()]
    df["trial_duration_days"] = df["trial_duration_days"].astype(int)
    df["status"] = df["status"].astype(int)
    return df

# -------- Group Rare Categories --------
def group_rare_categories(df, threshold=0.01):
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        freq = df[col].value_counts(normalize=True)
        rare = freq[freq < threshold].index
        df[col] = df[col].replace(rare, 'Other')
    return pd.get_dummies(df, drop_first=True)

# -------- Fairness Evaluation --------
def evaluate_c_index_by_group(df_raw, df_model, hazards, group_col, is_continuous=False, bins=3):
    print(f"\nFairness Evaluation by '{group_col}':")
    results = []

    if is_continuous:
        labels = [f"Low", f"Mid", f"High"][:bins]
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
            -hazards[mask],
            df_raw.loc[mask, "status"]
        )
        results.append((group, mask.sum(), df_raw.loc[mask, "status"].sum(), ci))
        print(f"  {str(group):20}  N={mask.sum():3}  Events={df_raw.loc[mask, 'status'].sum():3}  C-index={ci:.4f}")
    return results

# -------- Main Evaluation --------
if __name__ == "__main__":
    print("Loading test data...")
    df_raw = load_clean_data(TEST_PATH)
    df_raw = df_raw.drop(columns=["nct_id", "condition_group_count"], errors="ignore")
    df_raw = group_rare_categories(df_raw)
    df_raw = df_raw.dropna()

    print("Loading trained Cox model...")
    with open(MODEL_PATH, "rb") as f:
        cph = pickle.load(f)

    print("Aligning test features to model training columns...")
    model_cols = cph.params_.index.tolist()
    for col in model_cols:
        if col not in df_raw.columns:
            df_raw[col] = 0.0  # Add missing features as zeros

    keep_cols = model_cols + ['trial_duration_days', 'status']
    df_model = df_raw[[col for col in keep_cols if col in df_raw.columns]]

    print("Predicting risk and evaluating global C-index...")
    hazards = cph.predict_partial_hazard(df_model)
    c_index = concordance_index(
        df_model["trial_duration_days"],
        -hazards,
        df_model["status"]
    )

    num_events = df_model["status"].sum()
    num_total = df_model.shape[0]

    print(f"\nGlobal C-index: {c_index:.4f}")
    print(f"Events: {num_events} / Observations: {num_total}")

    METRIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRIC_PATH, "w") as f:
        f.write(f"Global C-index: {c_index:.4f}\n")
        f.write(f"Events: {num_events} / Observations: {num_total}\n")

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

    continuous_group_cols = ["site_count", 
                             "maximum_age",
                             "collaborator_count",
                             "arm_count",
                             "eligibility_token_count",
                             "exclusion_criteria_count",
                             "secondary_outcome_count"]

    # Evaluate categorical
    for group_col in categorical_group_cols:
        if group_col in df_raw.columns:
            results = evaluate_c_index_by_group(df_raw, df_model, hazards, group_col, is_continuous=False)
            with open(METRIC_PATH, "a") as f:
                f.write(f"\nFairness Evaluation by '{group_col}':\n")
                for group, n, ev, ci in results:
                    f.write(f"  {group:20}  N={n:3}  Events={ev:3}  C-index={ci:.4f}\n")

    # Evaluate binned continuous
    for group_col in continuous_group_cols:
        if group_col in df_raw.columns:
            results = evaluate_c_index_by_group(df_raw, df_model, hazards, group_col, is_continuous=True, bins=3)
            with open(METRIC_PATH, "a") as f:
                f.write(f"\nFairness Evaluation by binned '{group_col}':\n")
                for group, n, ev, ci in results:
                    f.write(f"  {group:20}  N={n:3}  Events={ev:3}  C-index={ci:.4f}\n")

    print(f"\nAll results saved to: {METRIC_PATH}")
