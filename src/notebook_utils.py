import dateparser
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional
import time
import httpx
from utils import get_github_client_with_timeout

# Initialize the GitHub client
client = get_github_client_with_timeout()

# Get the token for the print statement
TOKEN = os.getenv("GITHUB_TOKEN")

HEADERS = {
    "Accept": "application/vnd.github.cloak-preview+json",
}

def get_repo(owner: str, repo: str) -> dict:
    r = client.get(f"/repos/{owner}/{repo}")
    r.raise_for_status()
    print("GitHub client ready. Token:", "yes" if TOKEN else "no")
    return r.json()

def get_repo_code_size_python_lines(owner: str, repo: str) -> int:
    # todo use cloc to get the number of python lines
    return 0

def get_commit_activity(owner: str, repo: str, weeks: int = 52) -> list:
    # Returns last `weeks` weeks of commit counts per week
    r = client.get(f"/repos/{owner}/{repo}/stats/commit_activity")
    if r.status_code == 202:
        # GitHub may be computing stats; retry after short wait
        import time
        for _ in range(10):
            time.sleep(1.5)
            r = client.get(f"/repos/{owner}/{repo}/stats/commit_activity")
            if r.status_code != 202:
                break
    r.raise_for_status()
    return r.json()

def plot_commit_activity(activity: list):
    activity_df = pd.DataFrame(activity)
    if not activity_df.empty:
        # GitHub returns unix week timestamp (start of week)
        activity_df["week"] = pd.to_datetime(activity_df["week"], unit="s")
        activity_df.rename(columns={"total": "commits"}, inplace=True)
        activity_df.sort_values("week", inplace=True)
        activity_df["commits_4w_ma"] = activity_df["commits"].rolling(4, min_periods=1).mean()

        ax = activity_df.set_index("week")["commits_4w_ma"].plot(
            figsize=(9, 4), title="SciPy weekly commits (4-week moving average)", color="tab:blue"
        )
        activity_df.set_index("week")["commits"].plot(ax=ax, alpha=0.3, color="tab:gray")
        ax.set_ylabel("Commits per week")
        plt.tight_layout()
        plt.show()
    return


def to_datetime(s: str):
    return dateparser.parse(s)

def github_search_total(endpoint: str, q: str, extra_headers: Optional[dict] = None) -> Optional[int]:
    """Return total_count from GitHub Search API for the given endpoint and query.

    endpoint: one of {"issues", "commits", "code", ...}
    q: search query string
    """
    headers = HEADERS.copy()
    if extra_headers:
        headers.update(extra_headers)

    delay_seconds = float(os.getenv("GITHUB_SEARCH_DELAY", "1.5"))
    max_retries = 3

    for attempt in range(max_retries):
        try:
            r = client.get(
                f"/search/{endpoint}", params={"q": q, "per_page": 1}, headers=headers
            )

            if r.status_code == 403:
                # Rate limit; sleep until reset if header present, else exponential fallback
                reset_epoch = int(r.headers.get("X-RateLimit-Reset", "0") or 0)
                now = int(time.time())
                wait_s = max(0, reset_epoch - now)
                if wait_s == 0:
                    wait_s = (attempt + 1) * 10
                print(
                    f"Pausing to process additional requests. Sleeping ~{wait_s}s. Query: {endpoint} :: {q}"
                )
                time.sleep(wait_s + 1)
                continue

            if r.status_code in (500, 502, 503, 504):
                backoff = (attempt + 1) * 5
                print(f"Server {r.status_code}. Backing off {backoff}s...")
                time.sleep(backoff)
                continue

            r.raise_for_status()
            data = r.json()
            total = int(data.get("total_count", 0))
            # Throttle successful calls
            time.sleep(delay_seconds)
            return total
        except httpx.HTTPError as e:
            backoff = (attempt + 1) * 5
            print(f"HTTP error for {endpoint}: {e}. Retrying in {backoff}s...")
            time.sleep(backoff)

    print(f"Failed after {max_retries} attempts: {endpoint} :: {q}")
    return None


def yearly_counts(owner: str, repo: str, start_year: int, end_year: int) -> pd.DataFrame:
    """Compute yearly totals for commits, issues opened, PRs opened, and merged PRs.

    Uses Search API total_count to avoid heavy pagination.
    """
    records = []
    for year in range(start_year, end_year + 1):
        date_range = f"{year}-01-01..{year}-12-31"
        repo_qual = f"repo:{owner}/{repo}"

        commits = github_search_total(
            "commits",
            f"{repo_qual} committer-date:{date_range}",
            extra_headers={"Accept": "application/vnd.github.cloak-preview+json"},
        )
        issues_opened = github_search_total("issues", f"{repo_qual} is:issue created:{date_range}")
        prs_opened = github_search_total("issues", f"{repo_qual} is:pr created:{date_range}")
        prs_merged = github_search_total("issues", f"{repo_qual} is:pr is:merged merged:{date_range}")

        records.append(
            {
                "year": year,
                "commits": commits,
                "issues_opened": issues_opened,
                "prs_opened": prs_opened,
                "prs_merged": prs_merged,
            }
        )

    df = pd.DataFrame(records).set_index("year").sort_index()
    return df

def get_monthly_downloads(owner: str, repo: str, start_year: int, end_year: int) -> pd.DataFrame:
    """Get monthly PyPI download statistics for a package.

    Note: PyPI Stats API only retains data for 180 days. For historical data beyond this period,
    consider using the pypistats package or Google BigQuery as mentioned in the API docs.

    Uses the PyPI Stats API to retrieve daily download data and aggregates it to monthly totals.
    """
    # Create package name from owner/repo (e.g., "scipy/scipy" -> "scipy")
    package_name = owner if owner == repo else f"{owner}-{repo}"

    # Create a client for PyPI Stats API
    pypi_client = httpx.Client(base_url="https://pypistats.org/api")

    try:
        # Get overall download statistics (daily data for last 180 days only)
        r = pypi_client.get(f"/packages/{package_name}/overall")
        r.raise_for_status()
        data = r.json()

        # Convert to DataFrame
        df = pd.DataFrame(data["data"])

        # Filter for data without mirrors (more accurate)
        df = df[df["category"] == "without_mirrors"]

        # Convert date column to datetime
        df["date"] = pd.to_datetime(df["date"])

        # Group by month and sum downloads
        df["month"] = df["date"].dt.to_period("M")
        monthly_df = df.groupby("month")["downloads"].sum().reset_index()

        # Convert period back to datetime for easier plotting
        monthly_df["month"] = monthly_df["month"].dt.to_timestamp()

        # Format month as YYYY-MM for the desired output format
        monthly_df["month_str"] = monthly_df["month"].dt.strftime("%Y-%m")

        return monthly_df

    finally:
        pypi_client.close()

def load_scipy_downloads_data() -> pd.DataFrame:
    """Load SciPy download data from the text file."""
    data_file = os.path.join(os.path.dirname(__file__), "data", "scipy_downloads_pypi.txt")

    # Read tab-separated data
    df = pd.read_csv(data_file, sep='\t', header=0)

    # Convert month to datetime and num_downloads to numeric
    # The month column contains dates like "2018-03-01", parse as first day of month
    df['date'] = pd.to_datetime(df['month'], format='%Y-%m-%d')
    df['num_downloads'] = pd.to_numeric(df['num_downloads'])

    # Sort by date (oldest first) and reset index
    df = df.sort_values('date').reset_index(drop=True)

    return df

def plot_monthly_downloads(downloads: pd.DataFrame = None):
    """Plot monthly PyPI downloads as a bar chart with year ticks.

    If no data is provided, loads data from scipy_downloads_pypi.txt
    """
    if downloads is None:
        downloads = load_scipy_downloads_data()

    if downloads.empty:
        print("No download data available to plot.")
        return

    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 8))

    # Get the data - convert dates to matplotlib format
    x = mdates.date2num(downloads['date'])
    y = downloads['num_downloads']

    # Create bars
    ax.bar(x, y, color="tab:blue", alpha=0.75, width=20)  # width in days

    # Set title and labels
    plt.title("SciPy Monthly PyPI Downloads", fontsize=14, fontweight='bold')
    plt.ylabel("Downloads", fontsize=12)
    plt.xlabel("Month", fontsize=12)

    # Configure x-axis ticks to show every year
    ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))

    # Format y-axis with commas for thousands
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))

    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Rotate x-axis labels and adjust spacing
    plt.xticks(rotation=0, ha='center', fontsize=10)

    # Ensure the x-axis shows the full date range
    ax.set_xlim(x.min(), x.max())

    # Tight layout
    plt.tight_layout()
    plt.show()

    return ax

def print_monthly_downloads(downloads: pd.DataFrame):
    """Print monthly downloads in the format YYYY-MM, count"""
    if downloads.empty:
        print("No download data available.")
        return

    print("Monthly downloads:")
    for _, row in downloads.iterrows():
        print(f"{row['month_str']}, {int(row['downloads'])}")

def plot_yearly_trends(years_df: pd.DataFrame):
    # Plot yearly trends
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    axes = axes.ravel()

    plot_cols = [
        ("commits", "Commits"),
        ("issues_opened", "Issues opened"),
        ("prs_opened", "PRs opened"),
        ("prs_merged", "PRs merged"),
    ]

    for ax, (col, title) in zip(axes, plot_cols):
        years_df[col].plot(kind="bar", ax=ax, color="tab:blue", alpha=0.75)
        ax.set_title(title)
        ax.set_ylabel("Count")
        # Annotate last value
        if not years_df[col].dropna().empty:
            last_year = years_df[col].dropna().index.max()
            last_val = years_df.loc[last_year, col]
            ax.text(len(years_df) - 0.6, last_val, f"{int(last_val) if pd.notna(last_val) else 'NA'}",
                    va="bottom", ha="left", fontsize=9)

    for ax in axes:
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("SciPy year-by-year activity")
    plt.tight_layout()
    plt.show()
