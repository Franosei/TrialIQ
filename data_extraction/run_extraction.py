from disease_list import DISEASES
from api_client import fetch_trials_by_condition
from trial_parser import extract_features
import json

def run():
    all_data = []

    for disease in DISEASES:
        print(f"Fetching trials for: {disease}")
        trials = fetch_trials_by_condition(disease)
        all_data.extend([extract_features(t) for t in trials])

    with open("data/raw_trials.json", "w") as f:
        json.dump(all_data, f, indent=2)

if __name__ == "__main__":
    run()
