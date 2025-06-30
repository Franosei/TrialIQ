import json
from feature_pipeline import preprocess_trial
from utils import save_json, load_json
import sys
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def drop_high_missing_columns(data: list[dict], threshold: float = 0.02) -> list[dict]:
    df = pd.DataFrame(data)
    keep_columns = df.columns[df.isnull().mean() <= threshold]
    return df[keep_columns].to_dict(orient="records")


def parallel_preprocess(trials: list[dict], max_workers: int = 3) -> list[dict]:
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(preprocess_trial, trial) for trial in trials]
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print("Error processing trial:", e)
    return results


def main():
    # Load raw JSON from data file
    raw_data = load_json("data/raw_trials.json")

    # Remove duplicates based on `nct_id`
    seen_ids = set()
    deduped_data = []
    for trial in raw_data:
        trial_id = trial.get("nct_id")
        if trial_id and trial_id not in seen_ids:
            seen_ids.add(trial_id)
            deduped_data.append(trial)

    print(f"Loaded {len(raw_data)} raw trials.")
    print(f"Removed {len(raw_data) - len(deduped_data)} duplicate trials.")
    print(f"Proceeding with {len(deduped_data)} unique trials.")

    # Apply full preprocessing concurrently
    processed = parallel_preprocess(deduped_data, max_workers=3)

    # Drop columns with >2% missing values
    processed_cleaned = drop_high_missing_columns(processed)

    # Save output
    save_json("data/processed_trials.json", processed_cleaned)

    print(f"Preprocessed {len(processed_cleaned)} trials (after feature extraction and pruning).")
    print("Output saved to: data/processed_trials.json")


if __name__ == "__main__":
    main()
    