# preprocessing/outcome_features.py
from typing import Dict, Any

def extract_outcome_features(trial: Dict[str, Any]) -> Dict[str, Any]:
    features = {}
    status = trial.get("status", "").upper()

    features["trial_success"] = status == "COMPLETED"
    features["trial_failed"] = status in {"TERMINATED", "WITHDRAWN"}

    return features
