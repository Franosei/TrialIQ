import json
import pandas as pd
import numpy as np
from typing import Any, Dict
from sklearn.preprocessing import StandardScaler
import ast
import math

# File paths
INPUT_PATH = "data/processed_trials.json"
TRAIN_CSV_PATH = "data/train_trials.csv"
TEST_CSV_PATH = "data/test_trials.csv"
SUMMARY_TRAIN_PATH = "data/summary_statistics_train.json"
SUMMARY_TEST_PATH = "data/summary_statistics_test.json"
SUMMARY_FULL_PATH = "data/summary_statistics_full.json"

RELEVANT_COLUMNS = [
    "nct_id", "phase", "masking", "primary_purpose", "arm_count",
    "minimum_age", "maximum_age", "healthy_volunteers", "status",
    "collaborator_count", "site_count", "is_fda_regulated_drug", "is_fda_regulated_device",
    "oversight_has_dmc", "has_expanded_access", "trial_duration_days",
    "eligibility_token_count", "inclusion_criteria_count", "exclusion_criteria_count",
    "primary_outcome_count", "secondary_outcome_count", "masking_category",
    "randomization_category", "intervention_model_category", "sponsor_class_category",
    "intervention_type_count", "intervention_type_drug_ratio", "condition_groups",
    "intervention_grouped_category"
]

def load_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, data: Any):
    def convert(obj):
        if isinstance(obj, (np.integer, int)): return int(obj)
        if isinstance(obj, (np.floating, float)): return float(obj)
        if isinstance(obj, (np.bool_, bool)): return int(obj)
        if isinstance(obj, (pd.Timestamp, pd.Timedelta)): return str(obj)
        return str(obj)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=convert)

def summarize_continuous(series: pd.Series) -> Dict[str, Any]:
    clean_series = pd.to_numeric(series, errors="coerce").dropna()
    return {
        "count": int(clean_series.count()),
        "mean": float(clean_series.mean()),
        "std": float(clean_series.std()),
        "min": float(clean_series.min()),
        "25%": float(clean_series.quantile(0.25)),
        "median": float(clean_series.median()),
        "75%": float(clean_series.quantile(0.75)),
        "max": float(clean_series.max()),
        "missing": int(series.isna().sum())
    }

def summarize_categorical(series: pd.Series) -> Dict[str, Any]:
    normalized = series.fillna("MISSING").astype(str).str.strip().str.lower()
    normalized = normalized.replace({"missing": "OTHER", "none": "OTHER", "unknown": "OTHER"})
    freq = normalized.value_counts(dropna=False)
    total = len(series)
    return {
        label.upper(): {
            "count": int(count),
            "percent": round((count / total) * 100, 2)
        }
        for label, count in freq.items()
    }

def summarize_condition_groups(series: pd.Series) -> Dict[str, Any]:
    all_codes = []
    for val in series.dropna():
        if isinstance(val, list): all_codes.extend(val)
    freq = pd.Series(all_codes).value_counts()
    total_trials = len(series)
    return {
        str(code): {
            "count": int(count),
            "percent": round((count / total_trials) * 100, 2)
        } for code, count in freq.items()
    }

def generate_summary(df: pd.DataFrame) -> Dict[str, Any]:
    summary = {}
    for col in df.columns:
        if col == "nct_id":
            continue
        try:
            if col.startswith("C") and col[1:].isdigit():
                continue
            elif col == "condition_groups":
                summary[col] = summarize_condition_groups(df[col])
            elif df[col].dtype == "object" or df[col].nunique() <= 20:
                summary[col] = summarize_categorical(df[col])
            else:
                summary[col] = summarize_continuous(df[col])
        except Exception as e:
            summary[col] = {"error": str(e)}
    return summary

def get_primary_intervention_category(intervention_list):
    if not isinstance(intervention_list, list): return "Other"
    category_priority = {
        "Pharmacologic": 1, "Device/Diagnostic": 2, "Procedural": 3,
        "Behavioral/Lifestyle": 4, "Other": 5
    }
    mapping = {
        "drug": "Pharmacologic", "biological": "Pharmacologic",
        "device": "Device/Diagnostic", "diagnostic test": "Device/Diagnostic",
        "procedure": "Procedural", "radiation": "Procedural", "genetic": "Procedural",
        "behavioral": "Behavioral/Lifestyle", "dietary_supplement": "Behavioral/Lifestyle"
    }
    mapped = {mapping.get(i.lower().strip(), "Other") for i in intervention_list if isinstance(i, str)}
    return sorted(mapped, key=lambda c: category_priority[c])[0] if mapped else "Other"

def one_hot_condition_groups(train_df, test_df):
    def safe_parse(val):
        if isinstance(val, list): return val
        try: return ast.literal_eval(val)
        except: return []
    train_parsed = train_df["condition_groups"].apply(safe_parse)
    test_parsed = test_df["condition_groups"].apply(safe_parse)
    all_codes = sorted(set(code for group in pd.concat([train_parsed, test_parsed]) for code in group if isinstance(code, str)))
    for code in all_codes:
        train_df[code] = train_parsed.apply(lambda x: 1 if code in x else 0)
        test_df[code] = test_parsed.apply(lambda x: 1 if code in x else 0)
    train_df["condition_group_count"] = train_parsed.apply(lambda x: len(set(x)))
    test_df["condition_group_count"] = test_parsed.apply(lambda x: len(set(x)))
    train_df.drop(columns=["condition_groups"], inplace=True)
    test_df.drop(columns=["condition_groups"], inplace=True)
    return train_df, test_df

def encode_status_as_binary(df):
    df["status"] = df["status"].apply(lambda x: 0 if str(x).strip().upper() == "COMPLETED" else 1)
    return df

def encode_all_categoricals(train_df, test_df):
    exclude_cols = {"nct_id", "status", "condition_group_count"}
    exclude_cols.update(c for c in train_df.columns if c.startswith("C") and c[1:].isdigit())

    def clean_string(val):
        if pd.isna(val): return "OTHER"
        try:
            val = str(val).strip()
            if val.startswith("[") and val.endswith("]"):
                parsed = ast.literal_eval(val)
                return str(parsed[0]) if parsed else "OTHER"
            return val
        except:
            return "OTHER"

    for col in train_df.columns:
        if col not in exclude_cols and train_df[col].dtype == "object":
            train_df[col] = train_df[col].apply(clean_string)
            test_df[col] = test_df[col].apply(clean_string)

    categorical_cols = [
        col for col in train_df.columns
        if col not in exclude_cols and train_df[col].dtype == "object" and train_df[col].nunique() <= 50
    ]

    full_df = pd.concat([train_df, test_df], axis=0)
    full_encoded = pd.get_dummies(full_df, columns=categorical_cols, drop_first=False)

    rename_map = {}
    for col in full_encoded.columns:
        if "_OTHER" in col:
            for orig in categorical_cols:
                if col.startswith(orig + "_OTHER"):
                    rename_map[col] = f"{orig}_OTHER"
    full_encoded.rename(columns=rename_map, inplace=True)

    train_encoded = full_encoded.iloc[:len(train_df)].reset_index(drop=True)
    test_encoded = full_encoded.iloc[len(train_df):].reset_index(drop=True)

    return train_encoded, test_encoded

def convert_booleans_to_int(df):
    for col in df.columns:
        if df[col].dtype == "bool":
            df[col] = df[col].astype(int)
    return df

def standardize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_standardize = [
        "arm_count", "minimum_age", "maximum_age", "collaborator_count", "site_count",
        "eligibility_token_count", "inclusion_criteria_count", "exclusion_criteria_count",
        "primary_outcome_count", "secondary_outcome_count",
        "intervention_type_drug_ratio", "condition_group_count"
    ]
    existing_cols = [col for col in columns_to_standardize if col in df.columns]
    if existing_cols:
        scaler = StandardScaler()
        df[existing_cols] = scaler.fit_transform(df[existing_cols])
        print("Standardized columns:", existing_cols)
    return df


def drop_constant_columns(train_df, test_df):
    drop_cols = []
    for col in train_df.columns:
        if train_df[col].nunique(dropna=True) <= 1:
            drop_cols.append(col)

    if drop_cols:
        print("Dropped constant or all-NaN columns:", drop_cols)
    train_df.drop(columns=drop_cols, inplace=True)
    test_df.drop(columns=drop_cols, inplace=True)
    return train_df, test_df



def drop_highly_correlated_columns(train_df, test_df, threshold=0.95):
    numeric_df = train_df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = set()
    while True:
        # Find the highest correlation
        max_corr = upper.max().max()
        if max_corr <= threshold:
            break
        max_col = upper.stack().idxmax()
        col1, col2 = max_col

        # Drop the one with higher mean correlation
        col1_mean = upper[col1].mean()
        col2_mean = upper[col2].mean()
        drop = col1 if col1_mean > col2_mean else col2

        to_drop.add(drop)
        upper.drop(index=drop, columns=drop, inplace=True)

    if to_drop:
        print("Dropped highly correlated columns:", sorted(to_drop))
    train_df.drop(columns=to_drop, inplace=True)
    test_df.drop(columns=to_drop, inplace=True)
    return train_df, test_df

def summary_data():
    trials = load_json(INPUT_PATH)
    valid_trials = [t for t in trials if not t.get("duration_missing", True)]
    for trial in valid_trials:
        raw_types = trial.get("intervention_type_unique", [])
        trial["intervention_grouped_category"] = get_primary_intervention_category(raw_types)

    trimmed = [{k: t.get(k, None) for k in RELEVANT_COLUMNS} for t in valid_trials]
    df = pd.DataFrame(trimmed)
    indices = np.random.permutation(df.index)
    split_point = int(len(indices) * 0.8)
    train_df = df.iloc[indices[:split_point]].reset_index(drop=True)
    test_df = df.iloc[indices[split_point:]].reset_index(drop=True)

    train_df = train_df.loc[:, ~train_df.columns.str.lower().str.contains("unknown")]
    test_df = test_df.loc[:, ~test_df.columns.str.lower().str.contains("unknown")]

    train_df, test_df = one_hot_condition_groups(train_df, test_df)
    train_df, test_df = encode_all_categoricals(train_df, test_df)

    train_df = train_df.loc[:, ~train_df.columns.str.lower().str.contains("unknown")]
    test_df = test_df.loc[:, ~test_df.columns.str.lower().str.contains("unknown")]

    train_df.drop(columns=["nct_id"], inplace=True, errors="ignore")
    test_df.drop(columns=["nct_id"], inplace=True, errors="ignore")

    train_df, test_df = drop_constant_columns(train_df, test_df)
    train_df, test_df = drop_highly_correlated_columns(train_df, test_df, threshold=0.9)

    train_df = encode_status_as_binary(train_df)
    test_df = encode_status_as_binary(test_df)
    train_df = convert_booleans_to_int(train_df)
    test_df = convert_booleans_to_int(test_df)

    train_df.columns = train_df.columns.str.lower()
    test_df.columns = test_df.columns.str.lower()

    stats_train = generate_summary(train_df)
    stats_test = generate_summary(test_df)
    stats_full = generate_summary(df)
    
    # Standardize float columns separately for each dataset
    train_df = standardize_dataset(train_df)
    test_df = standardize_dataset(test_df)

    train_df.to_csv(TRAIN_CSV_PATH, index=False)
    test_df.to_csv(TEST_CSV_PATH, index=False)
    save_json(SUMMARY_TRAIN_PATH, stats_train)
    save_json(SUMMARY_TEST_PATH, stats_test)
    save_json(SUMMARY_FULL_PATH, stats_full)

    print(f"Train set saved to: {TRAIN_CSV_PATH}")
    print(f"Test set saved to: {TEST_CSV_PATH}")
    print(f"Train summary saved to: {SUMMARY_TRAIN_PATH}")
    print(f"Test summary saved to: {SUMMARY_TEST_PATH}")
    print(f"Full summary saved to: {SUMMARY_FULL_PATH}")

if __name__ == "__main__":
    summary_data()
