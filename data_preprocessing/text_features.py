# preprocessing/text_features.py

from typing import Dict, Any
import re
import nltk
from nltk.tokenize import TreebankWordTokenizer, sent_tokenize

nltk.download('punkt', quiet=True)

def count_tokens(text: str) -> int:
    if not text:
        return 0
    tokenizer = TreebankWordTokenizer()
    return len(tokenizer.tokenize(text))

def count_sentences(text: str) -> int:
    """Count sentences using punctuation or newline structure."""
    if not text:
        return 0
    nltk_sents = sent_tokenize(text)
    newline_sents = re.split(r'\n+', text)
    clean_nltk = [s.strip() for s in nltk_sents if s.strip()]
    clean_newline = [s.strip() for s in newline_sents if s.strip()]
    return max(len(clean_nltk), len(clean_newline))

def split_eligibility_sections(text: str) -> dict:
    inclusion, exclusion = '', ''
    parts = re.split(r'(?i)exclusion criteria:?', text)
    if len(parts) == 2:
        inclusion = parts[0]
        exclusion = parts[1]
    elif 'inclusion criteria' in text.lower():
        inclusion = text
    return {
        "inclusion_criteria_count": count_sentences(inclusion),
        "exclusion_criteria_count": count_sentences(exclusion)
    }

def extract_text_features(trial: Dict[str, Any]) -> Dict[str, Any]:
    features = {}
    criteria = trial.get("eligibility_criteria", "")

    features["eligibility_token_count"] = count_tokens(criteria)
    features.update(split_eligibility_sections(criteria))

    features["primary_outcome_count"] = len(trial.get("primary_outcomes", []))
    features["secondary_outcome_count"] = len(trial.get("secondary_outcomes", []))
    features["has_primary_outcome"] = bool(trial.get("primary_outcomes"))
    features["has_secondary_outcome"] = bool(trial.get("secondary_outcomes"))

    return features
