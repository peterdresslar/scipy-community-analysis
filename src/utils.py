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
        month_prs = list(_iter_search_issues_prs(owner, repo, "pr", search_delay, date_range))
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
    # Core physical sciences and subfields
    "physics": ["physics", "quantum", "relativ", "particle", "optics", "thermo"],
    "astrophysics": ["astrophys", "star", "galax", "supernova", "exoplanet", "cosmic"],
    "cosmology": ["cosmolog", "big bang", "cmb", "dark matter", "dark energy"],
    "electromagnetism": ["electromagnet", "maxwell", "electrostat", "magnetostat", "em field"],
    "mechanics": ["mechanic", "kinematic", "dynamic", "rigid", "continuum"],
    "fluid-dynamics": ["fluid", "navier", "stokes", "turbulen", "cfd", "viscous"],
    "materials-science": ["material", "crystal", "alloy", "polymer", "composite", "microstruct"],
    "statistical-physics": ["statistical phys", "ising", "boltzmann", "gibbs", "lattice"],

    # Chemistry and related
    "chemistry": ["chem", "molecule", "reaction", "spectroscop", "chromatograph", "nmr"],
    "computational-chemistry": ["quantum chem", "dft", "hartree", "gaussian", "mopac"],
    "cheminformatics": ["cheminform", "smiles", "rdkit", "qsar", "ligand", "dock"],

    # Earth and environmental sciences
    "geoscience": ["geo", "earth", "seism", "climat", "atmos", "ocean", "hydro"],
    "geology": ["geolog", "stratigraph", "tecton", "sediment", "petrolog"],
    "meteorology": ["meteorolog", "weather", "precip", "wind", "storm", "forecast"],
    "oceanography": ["oceanograph", "curren", "salinit", "thermocline", "sea surface"],
    "hydrology": ["hydrolog", "watershed", "runoff", "streamflow", "aquifer"],
    "remote-sensing": ["remote sensing", "landsat", "sentinel", "sar", "hyperspectral"],
    "geospatial": ["gis", "geospatial", "raster", "vector", "geodes", "proj"],

    # Life sciences
    "biology": ["bio", "genom", "protein", "cell", "dna", "rna", "enzyme"],
    "bioinformatics": ["bioinform", "fasta", "blast", "alignment", "transcriptom", "proteom"],
    "ecology": ["ecolog", "ecosystem", "biodivers", "population", "species", "niche"],
    "evolutionary-biology": ["evolution", "phylogen", "selection", "drift", "mutation"],
    "population-genetics": ["population genetic", "hardy", "allele", "haplotyp", "coalescent"],
    "neuroscience": ["neuro", "brain", "cortex", "synap", "eeg", "fmri"],
    "computational-neuroscience": ["spike", "neuron model", "hodgkin", "izhikevich", "connectom"],
    "medical": ["medic", "clinic", "diagnos", "imaging", "ct", "mri", "health"],
    "epidemiology": ["epidemiolog", "incidence", "prevalence", "r0", "sero", "contact"],

    # Mathematics and applied math
    "mathematics": ["mathemat", "algebra", "calculus", "topolog", "analysis"],
    "linear-algebra": ["linear algebra", "matrix", "eigen", "svd", "lu", "qr"],
    "numerical-linear-algebra": ["iterative solver", "cg ", "gmres", "bicg", "precondition"],
    "differential-equations": ["ode", "pde", "bvp", "ivp", "stiff", "finite element"],
    "time-series": ["time series", "arima", "sarima", "seasonal", "autocorrel", "spectral"],
    "optimization": ["optimiz", "gradient", "solver", "minimiz", "l-bfgs", "newton"],
    "convex-optimization": ["convex", "cvx", "kkt", "interior-point", "proximal"],
    "combinatorial-optimization": ["combinator", "integer", "mixed-integer", "tsp", "knapsack"],
    "control-systems": ["control", "pid", "state-space", "lqr", "kalman", "observer"],
    "simulation": ["simulation", "agent-based", "discrete event", "simulator", "modeling"],
    "monte-carlo": ["monte carlo", "mcmc", "metropolis", "gibbs", "importance sampling"],

    # Statistics and data analysis
    "statistics": ["stat", "regression", "bayes", "anova", "likelihood", "bootstrap"],
    "bayesian-statistics": ["bayes", "posterior", "prior", "mcmc", "variational"],
    "probability": ["probabilit", "stochastic", "markov", "random variable", "distribution"],
    "hypothesis-testing": ["hypothesis", "p-value", "t-test", "chi-square", "confidence"],

    # ML/AI and subfields
    "machine-learning": ["machine learning", "neural", "svm", "random forest", "xgboost", "deep"],
    "deep-learning": ["deep learning", "cnn", "rnn", "transformer", "pytorch", "tensorflow"],
    "reinforcement-learning": ["reinforcement", "policy", "q-learning", "td", "actor-critic"],
    "natural-language-processing": ["nlp", "token", "bert", "sentence", "embedding", "llm"],
    "computer-vision": ["image", "vision", "segmentation", "detection", "opencv"],
    "image-processing": ["image", "denois", "deconvol", "morpholog", "edge", "registration"],
    "signal-processing": ["fft", "filter", "wavelet", "signal", "frequency", "spectral"],
    "time-frequency-analysis": ["spectrogram", "stft", "cwt", "wavelet", "hilbert"],
    "audio": ["audio", "speech", "sound", "acoustic", "phoneme"],

    # Information and network sciences
    "information-theory": ["entropy", "mutual information", "kl", "fano", "shannon"],
    "graph-theory": ["graph", "node", "edge", "centrality", "clique", "graph cut"],
    "network-science": ["network", "community", "scale-free", "small-world", "degree"],

    # Social and economic sciences
    "economics": ["econom", "finance", "market", "econometric", "macro", "micro"],
    "econometrics": ["econometr", "panel", "instrumental", "gmm", "difference-in-differences"],
    "quant-finance": ["option", "derivative", "volatilit", "black-scholes", "risk"],

    # Robotics and control-adjacent
    "robotics": ["robot", "slam", "path planning", "manipulat", "kinematic", "navigation"],
}


def _find_tags(text: str) -> List[str]:
    text_lc = (text or "").lower()
    present: List[str] = []
    for tag, keywords in _SCIENCE_KEYWORDS.items():
        if any(kw in text_lc for kw in keywords):
            present.append(tag)
    return present


def _semantic_tags_if_enabled(text: str) -> List[str]:
    """Compute semantic tags if SCA_USE_SEMANTIC is truthy; otherwise empty list.

    Uses sentence-transformers via src.semantic if available. Falls back to empty on error.
    """
    use_semantic = (os.getenv("SCA_USE_SEMANTIC", "0").lower() in ("1", "true", "yes"))
    if not use_semantic:
        return []
    # Try both package and module import styles
    semantic_tags_for_text = None
    try:
        from .semantic import semantic_tags_for_text as _stft  # type: ignore
        semantic_tags_for_text = _stft
    except Exception:
        try:
            from semantic import semantic_tags_for_text as _stft  # type: ignore
            semantic_tags_for_text = _stft
        except Exception:
            return []

    try:
        return semantic_tags_for_text(text, _SCIENCE_KEYWORDS)
    except Exception:
        # Any runtime error should not break the pipeline
        return []


def get_issue_tags(text: str) -> List[str]:
    baseline = _find_tags(text)
    semantic = _semantic_tags_if_enabled(text)
    if not semantic:
        return baseline
    # Union while preserving order: prefer baseline tags first, then add unseen semantic tags
    seen = set(baseline)
    merged = list(baseline)
    for tag in semantic:
        if tag not in seen:
            merged.append(tag)
            seen.add(tag)
    return merged


def get_pr_tags(text: str) -> List[str]:
    baseline = _find_tags(text)
    semantic = _semantic_tags_if_enabled(text)
    if not semantic:
        return baseline
    seen = set(baseline)
    merged = list(baseline)
    for tag in semantic:
        if tag not in seen:
            merged.append(tag)
            seen.add(tag)
    return merged


