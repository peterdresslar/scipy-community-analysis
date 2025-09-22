import os
import time
from typing import Dict, Iterable, List, Optional

import httpx
from dotenv import load_dotenv


def get_github_client_with_timeout(timeout: int = 30) -> httpx.Client:
    load_dotenv()

    GITHUB_API = "https://api.github.com"
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    DEFAULT_HEADERS: Dict[str, str] = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        DEFAULT_HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    _client = httpx.Client(base_url=GITHUB_API, headers=DEFAULT_HEADERS, timeout=timeout)
    return _client


def _get_with_backoff(path: str, params: Optional[Dict[str, object]] = None) -> httpx.Response:
    """HTTP GET with simple backoff and rate limit handling."""
    # set search delay from params or default to 1.5
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = get_github_client_with_timeout().get(path, params=params)

            if resp.status_code == 403:
                reset_epoch_str = resp.headers.get("X-RateLimit-Reset", "0") or "0"
                try:
                    reset_epoch = int(reset_epoch_str)
                except ValueError:
                    reset_epoch = 0
                now = int(time.time())
                wait_seconds = max(0, reset_epoch - now)
                if wait_seconds == 0:
                    wait_seconds = (attempt + 1) * 10
                print(f"Rate limited. Sleeping ~{wait_seconds}s for {path} ...")
                time.sleep(wait_seconds + 1)
                continue

            if resp.status_code == 422:
                print(f"Query error on {path}: {resp.text}")
                if "Search API" in resp.text and "1000" in resp.text:
                    print("Hit 1000 result limit. Consider splitting query into smaller chunks.")
                raise RuntimeError(f"Query error on {path}: {resp.status_code} {resp.text}")

            if resp.status_code in (500, 502, 503, 504):
                backoff = (attempt + 1) * 5
                print(f"Server {resp.status_code} on {path}. Backing off {backoff}s...")
                time.sleep(backoff)
                continue

            resp.raise_for_status()
            return resp
        except httpx.HTTPError as exc:
            backoff = (attempt + 1) * 5
            print(f"HTTP error on {path}: {exc}. Retrying in {backoff}s...")
            time.sleep(backoff)
    raise RuntimeError(f"Failed to GET {path} after {max_retries} retries")


def _iter_search_issues_prs(owner: str, repo: str, query_kind: str, search_delay: float, created_range: str) -> Iterable[dict]:
    """Iterate search results for issues or PRs for a given created date range.

    query_kind: "issue" or "pr"
    created_range: e.g., "2024-01-01..2024-12-31"
    Note: subject to Search API 1000-item cap per query.
    """
    assert query_kind in {"issue", "pr"}
    q = f"repo:{owner}/{repo} is:{query_kind} created:{created_range}"
    per_page = 100
    page = 1
    fetched = 0
    while True:
        resp = _get_with_backoff(
            "/search/issues",
            params={
                "q": q,
                "per_page": per_page,
                "page": page,
                # Sorting by created can help pagination be more deterministic
                "sort": "created",
                "order": "asc",
            },
        )
        data = resp.json()
        items = data.get("items", [])
        if not items:
            break
        for it in items:
            yield it
        fetched += len(items)
        page += 1
        # throttle between pages
        time.sleep(search_delay)
        # Guard against 1000-item window cap
        if fetched >= 1000:
            print("Reached 1000 search results cap; consider splitting the year into months.")
            break


def get_all_repo_issues(owner: str, repo: str, year: Optional[int] = None, search_delay: float = 1.5) -> List[dict]:
    """Fetch issues created in the given year (state=all) via Search API.

    If year is None, returns an empty list to avoid massive downloads by default.
    """
    if year is None:
        return []

    # Split year into monthly chunks to avoid 1000-item limit
    all_issues = []
    for month in range(1, 13):
        start_date = f"{year}-{month:02d}-01"
        if month == 12:
            end_date = f"{year}-12-31"
        else:
            end_date = f"{year}-{month+1:02d}-01"
        date_range = f"{start_date}..{end_date}"

        print(f"Fetching issues for {owner}/{repo} in {date_range}")
        month_issues = list(_iter_search_issues_prs(owner, repo, "issue", search_delay, date_range))
        all_issues.extend(month_issues)

        # Add small delay between months
        if month < 12:
            time.sleep(1)

    return all_issues


def get_all_repo_prs(owner: str, repo: str, year: Optional[int] = None, search_delay: float = 1.5) -> List[dict]:
    """Fetch pull requests created in the given year (state=all) via Search API.

    If year is None, returns an empty list to avoid massive downloads by default.
    """
    if year is None:
        return []

    # Split year into monthly chunks to avoid 1000-item limit
    all_prs = []
    for month in range(1, 13):
        start_date = f"{year}-{month:02d}-01"
        if month == 12:
            end_date = f"{year}-12-31"
        else:
            end_date = f"{year}-{month+1:02d}-01"
        date_range = f"{start_date}..{end_date}"

        print(f"Fetching PRs for {owner}/{repo} in {date_range}")
        month_prs = list(_iter_search_issues_prs(owner, repo, "pr", date_range))
        all_prs.extend(month_prs)

        # Add small delay between months
        if month < 12:
            time.sleep(1)

    return all_prs


def get_issue_text(issue: dict) -> str:
    """Compose semantically relevant text from an issue's title, body, and labels."""
    title = issue.get("title") or ""
    body = issue.get("body") or ""
    labels = " ".join([lbl.get("name", "") for lbl in issue.get("labels", [])])
    return f"{title}\n{body}\n{labels}"


def get_pr_text(pr: dict) -> str:
    """Compose semantically relevant text from a PR's title and body."""
    title = pr.get("title") or ""
    body = pr.get("body") or ""
    return f"{title}\n{body}"


_SCIENCE_KEYWORDS: Dict[str, List[str]] = {
    "physics": ["physics", "quantum", "relativ", "particle", "optics", "thermo"],
    "chemistry": ["chem", "molecule", "reaction", "spectroscop", "chromatograph", "nmr"],
    "biology": ["bio", "genom", "protein", "cell", "dna", "rna", "enzyme"],
    "neuroscience": ["neuro", "brain", "cortex", "synap", "eeg", "fmri"],
    "astronomy": ["astro", "cosmic", "galax", "stellar", "planet", "telescope"],
    "geoscience": ["geo", "earth", "seism", "climat", "atmos", "ocean", "hydro"],
    "statistics": ["stat", "regression", "bayes", "anova", "likelihood", "bootstrap"],
    "machine-learning": ["machine learning", "neural", "svm", "random forest", "xgboost", "deep"],
    "signal-processing": ["fft", "filter", "wavelet", "signal", "frequency", "spectral"],
    "optimization": ["optimiz", "gradient", "solver", "minimiz", "l-bfgs", "newton"],
    "numerical-methods": ["ode", "pde", "integrat", "interpol", "sparse", "eigen", "matrix"],
    "medical": ["medic", "clinic", "diagnos", "imaging", "ct", "mri", "health"],
    "economics": ["econom", "finance", "market", "econometric", "macro", "micro"],
    "computer-vision": ["image", "vision", "segmentation", "detection", "opencv"],
    "audio": ["audio", "speech", "sound", "acoustic", "phoneme"],
}


def _find_tags(text: str) -> List[str]:
    text_lc = (text or "").lower()
    present: List[str] = []
    for tag, keywords in _SCIENCE_KEYWORDS.items():
        if any(kw in text_lc for kw in keywords):
            present.append(tag)
    return present


def get_issue_tags(text: str) -> List[str]:
    return _find_tags(text)


def get_pr_tags(text: str) -> List[str]:
    return _find_tags(text)


