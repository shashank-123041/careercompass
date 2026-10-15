"""
github_client/structure.py — roadmap STRUCTURE fallback.

WHY THIS EXISTS: the topic grouping/hierarchy the app shows (numbered
sections, sub-groups, "Common Hooks"-style clusters) is rebuilt from each
roadmap's react-flow file <slug>/<slug>.json (plus <slug>.md for the title,
description and role/skill tags). On 2026-07-28 the upstream
developer-roadmap repo deleted those files and moved everything to
roadmaps/<slug>/content/*.md only — so a fresh clone has topic CONTENT but
no hierarchy, and every topic renders as its own flat node.

The topic ids (the "@<id>" suffix in content filenames) are largely stable
across that change, so we take the hierarchy from the last commit that still
had the structure files (Config.ROADMAP_STRUCTURE_COMMIT) and the content
from whatever the user cloned. Topics that exist in the clone but were never
placed in the old diagram fall into the existing "Bonus Topics" groups
(github_client/parser.py), and roadmaps that did not exist at the pinned
commit simply stay flat.

Files are fetched from raw.githubusercontent.com (no API rate limit) once and
cached under Config.STRUCTURE_CACHE_DIR; a 404 is cached as a marker so it is
not re-requested on every sync. If the cloned repo still has its own
<slug>.json / <slug>.md (older commit), those always win and nothing here runs.
"""
import logging
import time
from typing import Tuple

import requests

from careercompass.config import Config

log = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "careercompass-roadmap-sync/1.0"}


def _raw_url(slug: str, filename: str) -> str:
    return (f"https://raw.githubusercontent.com/{Config.GITHUB_OWNER}/{Config.GITHUB_REPO}/"
            f"{Config.ROADMAP_STRUCTURE_COMMIT}/src/data/roadmaps/{slug}/{filename}")


def _get_one(slug: str, filename: str) -> str:
    """Returns file text, or '' if unavailable. Caches hits and 404s on disk."""
    cache_dir = Config.STRUCTURE_CACHE_DIR / slug
    cached = cache_dir / filename
    missing = cache_dir / (filename + ".missing")
    if cached.exists():
        return cached.read_text(encoding="utf-8", errors="replace")
    if missing.exists():
        return ""

    try:
        r = requests.get(_raw_url(slug, filename), headers=_HEADERS,
                         timeout=Config.REQUEST_TIMEOUT)
    except requests.RequestException as e:
        log.warning("structure fetch failed for %s/%s (%s) — will retry next sync", slug, filename, e)
        return ""

    cache_dir.mkdir(parents=True, exist_ok=True)
    if r.status_code == 200:
        cached.write_text(r.text, encoding="utf-8")
        time.sleep(0.05)
        return r.text
    if r.status_code == 404:
        missing.write_text("", encoding="utf-8")
        return ""
    log.warning("structure fetch %s/%s -> HTTP %s", slug, filename, r.status_code)
    return ""


def get_structure(slug: str) -> Tuple[str, str]:
    """(json_text, md_text) for a roadmap from the pinned commit; '' for any
    part that doesn't exist. Disabled entirely by ROADMAP_STRUCTURE_FALLBACK=false."""
    if not Config.ROADMAP_STRUCTURE_FALLBACK:
        return "", ""
    return _get_one(slug, f"{slug}.json"), _get_one(slug, f"{slug}.md")
