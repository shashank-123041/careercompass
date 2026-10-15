"""
github_client/api.py
HTTP client used only when ROADMAP_REPO_PATH is NOT set (remote mode).

Uses:
  • raw.githubusercontent.com for file content (no rate limit)
  • api.github.com for directory listings      (60/hr unauth, 5000/hr with token)
"""
import time, logging, requests
from typing import Optional, Dict, List
from careercompass.config import Config

log = logging.getLogger(__name__)

_session: Optional[requests.Session] = None
_last_api_call: float = 0.0


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({
            "User-Agent": "roadmap-explorer/3.0",
            "Accept": "application/vnd.github+json",
        })
        if Config.GITHUB_TOKEN:
            _session.headers["Authorization"] = f"Bearer {Config.GITHUB_TOKEN}"
    return _session


def _api_get(url: str) -> Optional[dict]:
    """Rate-limited GitHub API GET."""
    global _last_api_call
    elapsed = time.time() - _last_api_call
    if elapsed < Config.API_DELAY:
        time.sleep(Config.API_DELAY - elapsed)
    _last_api_call = time.time()

    try:
        r = _get_session().get(url, timeout=Config.REQUEST_TIMEOUT)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 403:
            log.warning("GitHub API rate limited — add GITHUB_TOKEN to .env")
        else:
            log.warning("GitHub API %d for %s", r.status_code, url)
        return None
    except Exception as e:
        log.error("GitHub API error: %s", e)
        return None


def _raw_get(path: str) -> Optional[str]:
    """Fetch raw file content from raw.githubusercontent.com."""
    url = Config.raw_github_url(path)
    try:
        r = _get_session().get(url, timeout=Config.REQUEST_TIMEOUT)
        return r.text if r.status_code == 200 else None
    except Exception as e:
        log.error("raw fetch error for %s: %s", path, e)
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def list_roadmap_slugs() -> List[str]:
    """Return list of roadmap slugs by listing the roadmaps directory via API."""
    url = Config.github_api_url(
        f"contents/{Config.REMOTE_ROADMAPS_SUBPATH}"
    )
    data = _api_get(url)
    if not isinstance(data, list):
        return []
    return [item["name"] for item in data
            if item.get("type") == "dir" and item.get("name")]


def fetch_roadmap_json(slug: str) -> Optional[str]:
    path = f"{Config.REMOTE_ROADMAPS_SUBPATH}/{slug}/{slug}.json"
    return _raw_get(str(path))


def fetch_roadmap_md(slug: str) -> Optional[str]:
    path = f"{Config.REMOTE_ROADMAPS_SUBPATH}/{slug}/{slug}.md"
    return _raw_get(str(path))


def list_content_files(slug: str) -> List[Dict]:
    """
    List all .md files inside content/ for a given roadmap.
    Returns list of {"name": ..., "download_url": ...}
    """
    url = Config.github_api_url(
        f"contents/{Config.REMOTE_ROADMAPS_SUBPATH}/{slug}/content"
    )
    data = _api_get(url)
    if not isinstance(data, list):
        return []
    return [{"name": f["name"], "download_url": f.get("download_url", "")}
            for f in data if f.get("name", "").endswith(".md")]


def fetch_content_file(slug: str, filename: str) -> Optional[str]:
    path = f"{Config.REMOTE_ROADMAPS_SUBPATH}/{slug}/content/{filename}"
    return _raw_get(str(path))


def check_rate_limit() -> dict:
    data = _api_get("https://api.github.com/rate_limit")
    if data:
        core = data.get("resources", {}).get("core", {})
        return {
            "limit":     core.get("limit", 0),
            "remaining": core.get("remaining", 0),
            "reset":     core.get("reset", 0),
        }
    return {"limit": 0, "remaining": 0, "reset": 0}
