import os
import pandas as pd
from collections import Counter
from typing import Dict, List

from utils import (
    get_issue_text,
    get_issue_tags,
    get_pr_text,
    get_pr_tags,
    get_all_repo_issues,
    get_all_repo_prs,
)

SEARCH_DELAY = float(os.getenv("GITHUB_SEARCH_DELAY", "1.5"))

def get_all_repo_issues_to_binned_data(owner: str, repo: str, year: int) -> List[dict]:
    """Fetch issues for a given year and return per-issue text and tags."""
    issues = get_all_repo_issues(owner, repo, year=year, search_delay=SEARCH_DELAY)
    results: List[dict] = []
    for issue in issues:
        issue_text = get_issue_text(issue)
        issue_tags = get_issue_tags(issue_text)
        results.append(
            {
                "issue_id": issue.get("id"),
                "issue_title": issue.get("title"),
                "issue_text": issue_text,
                "issue_tags": issue_tags,
                "year": year,
            }
        )
    return results


def get_all_repo_prs_to_binned_data(owner: str, repo: str, year: int) -> List[dict]:
    """Fetch PRs for a given year and return per-PR text and tags."""
    prs = get_all_repo_prs(owner, repo, year=year, search_delay=SEARCH_DELAY)
    results: List[dict] = []
    for pr in prs:
        pr_text = get_pr_text(pr)
        pr_tags = get_pr_tags(pr_text)
        results.append(
            {
                "pr_id": pr.get("id"),
                "pr_title": pr.get("title"),
                "pr_text": pr_text,
                "pr_tags": pr_tags,
                "year": year,
            }
        )
    return results


def aggregate_year(owner: str, repo: str, year: int) -> Dict[str, Dict[str, int]]:
    """Aggregate tag counts for issues and PRs for a single year.

    Returns a mapping: {"issues": {tag: count}, "prs": {tag: count}, "total": {tag: count}}
    """
    issue_rows = get_all_repo_issues_to_binned_data(owner, repo, year)
    pr_rows = get_all_repo_prs_to_binned_data(owner, repo, year)

    issue_counter: Counter = Counter()
    for row in issue_rows:
        issue_counter.update(row.get("issue_tags", []))

    pr_counter: Counter = Counter()
    for row in pr_rows:
        pr_counter.update(row.get("pr_tags", []))

    total_counter: Counter = issue_counter + pr_counter

    return {
        "issues": dict(issue_counter),
        "prs": dict(pr_counter),
        "total": dict(total_counter),
    }


def aggregate_range(owner: str, repo: str, start_year: int, end_year: int) -> Dict[int, Dict[str, Dict[str, int]]]:
    """Aggregate tag counts for each year in [start_year, end_year]."""
    summary: Dict[int, Dict[str, Dict[str, int]]] = {}
    for year in range(start_year, end_year + 1):
        print(f"Aggregating {owner}/{repo} for {year}...")
        summary[year] = aggregate_year(owner, repo, year)
    return summary


def store_counts_per_year(owner: str, repo: str, year: int, data: Dict[str, Dict[str, int]]) -> None:
    """Store tag counts for a single year in CSV format.

    Args:
        owner: GitHub repository owner
        repo: GitHub repository name
        year: Year to store data for
        data: Dictionary with keys 'issues', 'prs', 'total' containing tag->count mappings
    """
    # Create data directory if it doesn't exist
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    # Convert nested dictionary to long-format DataFrame
    rows = []
    for source_type in ['issues', 'prs', 'total']:
        if source_type in data:
            for tag, count in data[source_type].items():
                rows.append({
                    'owner': owner,
                    'repo': repo,
                    'year': year,
                    'source_type': source_type,
                    'tag': tag,
                    'count': count
                })

    if rows:  # Only create file if there's data
        df = pd.DataFrame(rows)

        # Create filename based on repository and year
        filename = f"{data_dir}/{owner}_{repo}_{year}_tag_counts.csv"
        df.to_csv(filename, index=False)
        print(f"  → Saved {len(rows)} rows to {filename}")


def aggregate_all_to_datafile(owner: str, repo: str, start_year: int, end_year: int) -> None:
    """Aggregate tag counts for each year in [start_year, end_year] and store in a CSV file."""
    summary = aggregate_range(owner, repo, start_year, end_year)
    for year, data in summary.items():
        print(f"Storing {owner}/{repo} for {year}...")
        store_counts_per_year(owner, repo, year, data)

if __name__ == "__main__":
    # Configuration
    OWNER = "scipy"
    REPO = "scipy"
    YEAR = int(os.getenv("SCA_SAMPLE_YEAR", "2024"))
    START_YEAR = int(os.getenv("SCA_START_YEAR", "2023"))
    END_YEAR = int(os.getenv("SCA_END_YEAR", "2024"))

    # Choose what to run based on environment variables
    if os.getenv("SCA_CREATE_DATAFILES", "").lower() in ("1", "true", "yes"):
        print(f"Creating data files for {OWNER}/{REPO} from {START_YEAR} to {END_YEAR}")
        aggregate_all_to_datafile(OWNER, REPO, START_YEAR, END_YEAR)
        print("Done! Check the 'data/' directory for CSV files.")
    else:
        # Default: run single year analysis
        from pprint import pprint
        print(f"Running sample aggregation for {OWNER}/{REPO} in {YEAR}")
        year_summary = aggregate_year(OWNER, REPO, YEAR)
        pprint(year_summary)