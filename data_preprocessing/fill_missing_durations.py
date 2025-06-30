import json
from pathlib import Path
from dateutil import parser
from datetime import datetime

def parse_date(date_str):
    try:
        return parser.parse(date_str).date()
    except Exception:
        return None

def calculate_duration(start, end):
    if start and end:
        return (end - start).days
    return None

def build_duration_lookup(raw_trials):
    lookup = {}
    for trial in raw_trials:
        nct_id = trial.get("nct_id")
        start = parse_date(trial.get("start_date"))
        end = parse_date(trial.get("completion_date"))
        duration = calculate_duration(start, end)
        if nct_id and duration is not None:
            lookup[nct_id] = duration
    return lookup

def update_null_durations(null_trials, duration_lookup):
    updated = []
    for trial in null_trials:
        nct_id = trial.get("nct_id")
        duration = duration_lookup.get(nct_id)
        if duration is not None:
            trial["trial_duration_days"] = duration
        updated.append(trial)
    return updated

def main():
    raw_path = Path("data/raw_trials.json")
    null_path = Path("data/null_duration_trials.json")
    output_path = Path("data/null_duration_trials_filled.json")

    with open(raw_path, 'r', encoding='utf-8') as f:
        raw_trials = json.load(f)

    with open(null_path, 'r', encoding='utf-8') as f:
        null_trials = json.load(f)

    duration_lookup = build_duration_lookup(raw_trials)
    updated_trials = update_null_durations(null_trials, duration_lookup)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(updated_trials, f, indent=2)

    print(f"Updated trials saved to {output_path}")

if __name__ == "__main__":
    main()
