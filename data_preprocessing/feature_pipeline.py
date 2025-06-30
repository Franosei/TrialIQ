from typing import Dict, Any, List

from normalize_fields import normalize_trial_fields
from text_features import extract_text_features
from outcome_features import extract_outcome_features
from categorical_encoder import encode_trial_categories
from condition_grouper import assign_condition_groups


def preprocess_trial(trial: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run full preprocessing pipeline on a single trial:
    - normalization
    - text-based features
    - outcome features
    - categorical encoding
    - MeSH-based condition grouping
    - preserve all fields
    """
    # Step 1: Normalize fields
    trial = normalize_trial_fields(trial)

    # Step 2: Feature extraction
    features = {}
    features.update(trial)
    features.update(extract_text_features(trial))
    features.update(extract_outcome_features(trial))
    features.update(encode_trial_categories(trial))
    features["condition_groups"] = assign_condition_groups(trial.get("conditions", []))

    # Step 3: Optionally prune fields — here we preserve everything
    features = prune_fields(features)

    return features


def preprocess_all(trials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [preprocess_trial(trial) for trial in trials]


def prune_fields(trial: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preserve all available fields from the trial dictionary.
    This prevents loss of important data like has_expanded_access, etc.
    """
    return trial.copy()  # No pruning — keep everything
