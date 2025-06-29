import time
import requests
from datetime import datetime, timedelta
from config import BASE_API_URL, DEFAULT_PAGE_SIZE

def safe_request(url, params, retries=3, backoff=2):
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"Request failed (attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                wait = backoff * (2 ** attempt)
                print(f"Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

def fetch_trials_by_condition(condition, desired_valid_trials=2000):
    all_trials = []
    page_token = None
    cutoff_date = datetime.today() - timedelta(days=10 * 365)
    accepted_statuses = {"COMPLETED", "TERMINATED", "WITHDRAWN"}

    while len(all_trials) < desired_valid_trials:
        params = {
            "query.term": condition,
            "pageSize": DEFAULT_PAGE_SIZE
        }
        if page_token:
            params["pageToken"] = page_token

        try:
            response = safe_request(BASE_API_URL, params)
        except Exception as e:
            print(f"Final error for {condition}: {e}")
            break

        data = response.json()
        studies = data.get("studies", [])
        if not studies:
            break

        for trial in studies:
            ps = trial.get("protocolSection", {})
            phases = ps.get("designModule", {}).get("phases", [])
            start_date_str = ps.get("statusModule", {}).get("startDateStruct", {}).get("date")
            status = ps.get("statusModule", {}).get("overallStatus")

            # Skip trials not in accepted status set
            if status not in accepted_statuses:
                continue

            # Skip invalid phase
            if not phases or "NA" in phases:
                continue

            # Skip missing or old start date
            if not start_date_str:
                continue
            try:
                start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            except ValueError:
                continue
            if start_date < cutoff_date:
                continue

            all_trials.append(trial)
            if len(all_trials) >= desired_valid_trials:
                break

        page_token = data.get("nextPageToken")
        if not page_token:
            break

        time.sleep(2)

    return all_trials

