import re
from typing import Dict, Any
from dateutil import parser
from datetime import datetime

def compute_trial_duration(start: str, end: str) -> int | None:
    try:
        start_date = parser.parse(start, default=datetime(1900, 1, 1))
        end_date = parser.parse(end, default=datetime(1900, 1, 1))
        return (end_date - start_date).days
    except Exception:
        return None

def normalize_trial_fields(trial: Dict[str, Any]) -> Dict[str, Any]:
    normalized = trial.copy()

    # Normalize key string fields
    for key in ["masking", "randomization", "intervention_model", "sponsor_class"]:
        if key in normalized and isinstance(normalized[key], str):
            normalized[key] = normalized[key].strip().lower()

    # Normalize phase
    phase = normalized.get("phase")
    if phase and isinstance(phase, str):
        normalized["phase"] = phase.strip().upper()

    # Enhanced age normalization
    for key in ["minimum_age", "maximum_age"]:
        val = normalized.get(key)
        if isinstance(val, str):
            match = re.match(r"(\d+)\s*(week|weeks|month|months|year|years)?", val.lower())
            if match:
                num = int(match.group(1))
                unit = match.group(2)
                if unit in {"week", "weeks"}:
                    normalized[key] = round(num / 52, 2)  # weeks to years
                elif unit in {"month", "months"}:
                    normalized[key] = round(num / 12, 2)  # months to years
                else:
                    normalized[key] = float(num)  # assume years if no unit or unit is year
            else:
                normalized[key] = None
        elif isinstance(val, (int, float)):
            normalized[key] = float(val)
        else:
            normalized[key] = None

    # Set default age range if missing
    if normalized["minimum_age"] is None:
        normalized["minimum_age"] = 0
    if normalized["maximum_age"] is None:
        normalized["maximum_age"] = 100

    # Normalize sex
    sex = normalized.get("sex")
    if sex:
        normalized["sex"] = sex.strip().upper()

    # Normalize booleans
    for b in [
        "healthy_volunteers",
        "is_fda_regulated_drug",
        "is_fda_regulated_device",
        "has_expanded_access",
        "oversight_has_dmc"
    ]:
        val = normalized.get(b)
        if isinstance(val, str):
            normalized[b] = val.strip().lower() == "true"
        elif isinstance(val, bool):
            normalized[b] = val
        else:
            normalized[b] = "none"

    # Drop unused field
    normalized.pop("oversight_authorities", None)

    # Normalize remaining string fields
    for key in ["title", "description", "status", "study_type"]:
        if key in normalized and isinstance(normalized[key], str):
            normalized[key] = normalized[key].strip()

    # Trial duration calculation
    start = normalized.get("start_date")
    end = normalized.get("completion_date")
    if start and end:
        normalized["trial_duration_days"] = compute_trial_duration(start, end)
        normalized["duration_missing"] = False
    else:
        normalized["trial_duration_days"] = None
        normalized["duration_missing"] = True

    return normalized
