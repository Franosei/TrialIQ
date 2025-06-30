# preprocessing/utils.py
import json
from datetime import datetime
from typing import Optional


def parse_iso_date(date_str: str) -> Optional[datetime]:
    """Parse a date string in ISO format, returning None if invalid."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        try:
            return datetime.strptime(date_str, "%Y-%m")
        except ValueError:
            return None


def safe_lower(s: Optional[str]) -> Optional[str]:
    if isinstance(s, str):
        return s.strip().lower()
    return None


def save_json(path: str, data: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
