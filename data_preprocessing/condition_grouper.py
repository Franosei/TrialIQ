import json
import os
import re
from typing import List, Dict, Set
from difflib import get_close_matches
from functools import lru_cache

# Global mappings
CONDITION_TO_TOP_C: Dict[str, Set[str]] = {}
NORMALIZED_NAME_INDEX: Dict[str, str] = {}
TOP_LEVEL_C_CODES: Set[str] = set()


def normalize_string(name: str) -> str:
    """Lowercase, remove punctuation and excess whitespace."""
    return re.sub(r"\W+", " ", name.lower()).strip()


def load_mesh_disease_tree(json_path: str = "data/mesh/mesh_disease_tree.json") -> None:
    global CONDITION_TO_TOP_C, NORMALIZED_NAME_INDEX, TOP_LEVEL_C_CODES

    if CONDITION_TO_TOP_C:
        return  # Already loaded

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"MeSH JSON tree not found at: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    term_map = data.get("term_to_top_c", {})
    CONDITION_TO_TOP_C = {normalize_string(k): set(v) for k, v in term_map.items() if v}
    NORMALIZED_NAME_INDEX = {normalize_string(k): k for k in term_map.keys()}
    for codes in CONDITION_TO_TOP_C.values():
        TOP_LEVEL_C_CODES.update(codes)


@lru_cache(maxsize=4096)
def match_condition(clean: str) -> Set[str]:
    # Step 1: Full exact match
    if clean in CONDITION_TO_TOP_C:
        return CONDITION_TO_TOP_C[clean]

    # Step 2: Fuzzy match
    close = get_close_matches(clean, CONDITION_TO_TOP_C.keys(), n=3, cutoff=0.70)
    if close:
        return CONDITION_TO_TOP_C[close[0]]

    # Step 3: Token-based partial match
    for token in clean.split():
        if len(token) < 4:
            continue
        token_close = get_close_matches(token, CONDITION_TO_TOP_C.keys(), n=3, cutoff=0.70)
        if token_close:
            return CONDITION_TO_TOP_C[token_close[0]]

    # Step 4: Substring fallback
    for key in CONDITION_TO_TOP_C:
        if clean in key:
            return CONDITION_TO_TOP_C[key]

    # Step 5: Nothing matched
    return {"OTHER"}


def assign_condition_groups(conditions: List[str]) -> List[str]:
    load_mesh_disease_tree()
    groups: Set[str] = set()

    for cond in conditions:
        clean = normalize_string(cond)
        groups.update(match_condition(clean))

    return sorted(groups)
