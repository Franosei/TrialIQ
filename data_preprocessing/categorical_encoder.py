from typing import Dict, Any, List
from collections import Counter

def encode_trial_categories(trial: Dict[str, Any]) -> Dict[str, Any]:
    features = {}

    def normalize(val: str) -> str:
        return val.lower() if isinstance(val, str) else "unknown"

    # Normalize standard categories
    for field in ["masking", "randomization", "intervention_model", "sponsor_class"]:
        features[f"{field}_category"] = normalize(trial.get(field))

    # Group sponsor_class into 3 buckets
    sponsor_raw = normalize(trial.get("sponsor_class"))
    if sponsor_raw == "industry":
        sponsor_group = "industry"
    elif sponsor_raw in {"nih", "other_gov", "fed"}:
        sponsor_group = "government"
    else:
        sponsor_group = "non_industry"
    features["sponsor_class_grouped"] = sponsor_group

    # Intervention types
    interventions: List[Dict[str, Any]] = trial.get("interventions", [])
    types = [normalize(i.get("type")) for i in interventions]
    counts = Counter(types)

    for int_type, count in counts.items():
        features[f"intervention_type_{int_type}_count"] = count

    # Unique type list and ratio
    unique_types = sorted(set(types))
    features["intervention_type_unique"] = unique_types
    features["intervention_type_count"] = len(interventions)
    features["intervention_type_drug_ratio"] = (
        features.get("intervention_type_drug_count", 0) / features["intervention_type_count"]
        if features["intervention_type_count"] else 0
    )

    # Binary flags for intervention presence
    core_types = ["drug", "biological", "device", "procedure", "behavioral", "radiation", "other"]
    for t in core_types:
        features[f"has_{t}_intervention"] = t in unique_types

    # Intervention archetype binning
    if unique_types == ["drug"]:
        archetype = "drug_only"
    elif set(["drug", "biological"]).issubset(unique_types):
        archetype = "drug_biological_combo"
    elif "device" in unique_types and "drug" in unique_types:
        archetype = "drug_device_combo"
    elif "device" in unique_types:
        archetype = "device_only_or_combo"
    elif len(unique_types) == 1:
        archetype = f"{unique_types[0]}_only"
    elif not unique_types:
        archetype = "none"
    else:
        archetype = "mixed_other"
    features["intervention_archetype"] = archetype

    # Group into 5 broader categories
    if unique_types == ["drug"]:
        features["intervention_type_group"] = "drug_only"
    elif "drug" in unique_types:
        features["intervention_type_group"] = "drug_combo"
    elif "drug" not in unique_types and len(unique_types) == 1:
        features["intervention_type_group"] = "device_or_bio_only"
    elif "drug" not in unique_types and len(unique_types) > 1:
        features["intervention_type_group"] = "multi_combo_non_drug"
    else:
        features["intervention_type_group"] = "other_or_unknown"

    return features
