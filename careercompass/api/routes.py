"""
api/routes.py — REST API v2.

Serves data from local SQLite knowledge base.
Adds a /fetch-resources/<slug>/<topic_slug> endpoint that scrapes
roadmap.sh directly as a fallback when the DB has no resources for a topic.

STEP 1 ADDITION: per-user progress toggle + per-user roadmap selection
endpoints, and get_roadmap() now injects each topic's completion state
for the logged-in user (falls back to no annotation if not logged in).
"""
import re
import time
from functools import wraps

import requests
from bs4 import BeautifulSoup
from flask import Blueprint, jsonify, request, session

from careercompass.config import Config
from careercompass.db import get_conn
from careercompass.db import operations as ops
from careercompass.sync.engine import get_state, trigger_manual
from careercompass.auth.routes import login_required, admin_required

api = Blueprint("api", __name__, url_prefix="/api/v2")

# ── Scraper headers (same as original app.py) ─────────────────────────────────
_SCRAPE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# ── TTL cache ─────────────────────────────────────────────────────────────────
_cache: dict = {}

def cached(ttl: int = None):
    def dec(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            key = fn.__name__ + str(sorted(kwargs.items())) + str(sorted(request.args.items()))
            now = time.time()
            if key in _cache and now - _cache[key][0] < (ttl or Config.CACHE_TTL):
                return _cache[key][1]
            result = fn(*args, **kwargs)
            _cache[key] = (now, result)
            return result
        return wrapper
    return dec

def bust_cache():
    _cache.clear()

# ── Roadmaps ──────────────────────────────────────────────────────────────────

@api.get("/roadmaps")
@cached(60)
def list_roadmaps():
    t = request.args.get("type")
    roadmaps = ops.get_all_roadmaps(type_filter=t)
    return jsonify({"count": len(roadmaps),
                    "roadmaps": [r.to_dict() for r in roadmaps]})


@api.get("/roadmap/<slug>")
def get_roadmap(slug):
    """
    NOTE: no longer @cached — completion state is per-user, so caching this
    response would leak one user's progress into another's view. If this
    becomes a performance concern, cache the base tree separately from the
    completed-ids set and merge them per-request instead.
    """
    rm = ops.get_roadmap(slug)
    if not rm:
        state = get_state()
        return jsonify({
            "error":        f"Roadmap '{slug}' not found",
            "sync_running": state["running"],
            "sync_current": state.get("current"),
        }), 404

    tree = ops.get_topic_tree(rm.id)

    completed_ids = set()
    if "user_id" in session:
        completed_ids = ops.get_completed_topic_ids(session["user_id"], rm.id)

    def _to_dict_with_progress(t):
        d = t.to_dict(recurse=False)
        d["completed"] = t.id in completed_ids
        d["children"] = [_to_dict_with_progress(c) for c in t.children]
        return d

    def _count(nodes):
        n = 0
        for t in nodes:
            n += 1 + _count(t.children)
        return n

    return jsonify({
        "roadmap":     rm.to_dict(),
        "topics":      [_to_dict_with_progress(t) for t in tree],
        "topic_count": _count(tree),
        "progress_pct": ops.get_roadmap_progress_pct(session["user_id"], rm.id)
                         if "user_id" in session else None,
    })


# ── Live resource scraping fallback ──────────────────────────────────────────

@api.get("/fetch-resources/<slug>/<path:topic_slug>")
def fetch_resources(slug, topic_slug):
    """
    Scrapes roadmap.sh directly for a topic's resources.
    Called by the frontend when the DB has no resources for a node.
    URL pattern: roadmap.sh/{slug}/{topic_slug}
    topic_slug is derived from the topic title on the frontend.
    """
    url = f"https://roadmap.sh/{slug}/{topic_slug}"
    try:
        resp = requests.get(url, headers=_SCRAPE_HEADERS, timeout=15)
        if resp.status_code == 404:
            return jsonify({"resources": [], "description": "", "url": url})

        soup = BeautifulSoup(resp.text, "lxml")

        # Extract description from page
        desc = ""
        for sel in ["article p", "main p", ".content p"]:
            paras = soup.select(sel)
            if paras:
                desc = " ".join(p.get_text(strip=True) for p in paras[:2])
                break

        # Extract resource links — skip nav/branding links
        resources = []
        seen_urls: set = set()
        skip_domains = {"roadmap.sh", "twitter.com", "github.com/kamranahmedse",
                        "youtube.com/channel", "discord.gg"}

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            text = a.get_text(strip=True)

            if not href.startswith("http"):
                continue
            if href in seen_urls:
                continue
            if not text or len(text) < 4 or len(text) > 150:
                continue
            if any(d in href for d in skip_domains):
                continue
            if text.lower() in {"roadmap.sh", "github", "twitter",
                                 "discord", "youtube", "follow", "share"}:
                continue

            seen_urls.add(href)

            # Classify resource type
            href_l = href.lower()
            if "youtube.com" in href_l or "youtu.be" in href_l:
                rtype = "video"
            elif "github.com" in href_l:
                rtype = "opensource"
            elif re.search(r"(docs\.|/docs/|documentation|developer\.)", href_l):
                rtype = "official"
            elif re.search(r"(udemy|coursera|pluralsight|egghead|frontendmasters)", href_l):
                rtype = "course"
            else:
                rtype = "article"

            resources.append({"title": text, "url": href, "res_type": rtype})
            if len(resources) >= 15:
                break

        return jsonify({
            "resources":   resources,
            "description": desc,
            "source_url":  url,
        })

    except Exception as e:
        return jsonify({"resources": [], "description": "",
                        "error": str(e), "url": url})


# ── Per-topic from DB ─────────────────────────────────────────────────────────

@api.get("/roadmap/<slug>/topics")
def get_roadmap_topics(slug):
    rm = ops.get_roadmap(slug)
    if not rm:
        return jsonify({"error": "not found"}), 404
    c = get_conn()
    rows = c.execute(
        "SELECT id, title, topic_type, depth, parent_id FROM topics "
        "WHERE roadmap_id=? ORDER BY depth, order_index LIMIT 300",
        (rm.id,)
    ).fetchall()
    return jsonify({"slug": slug, "count": len(rows),
                    "topics": [dict(r) for r in rows]})


# ── Search ────────────────────────────────────────────────────────────────────

@api.get("/search")
def search():
    q = request.args.get("q", "").strip()
    if len(q) < 2:
        return jsonify({"error": "query too short"}), 400
    rm_slug = request.args.get("roadmap")
    rm_id   = None
    if rm_slug:
        rm = ops.get_roadmap(rm_slug)
        if rm: rm_id = rm.id
    results = ops.search_topics(q, roadmap_id=rm_id)
    return jsonify({"query": q, "count": len(results), "results": results})


# ── Stats / info ──────────────────────────────────────────────────────────────

@api.get("/stats")
def stats():
    return jsonify(ops.get_stats())


@api.get("/info")
def info():
    return jsonify({
        "mode":         "local" if Config.is_local_mode() else "remote",
        "repo_path":    str(Config.local_repo_path()) if Config.is_local_mode() else None,
        "github_owner": Config.GITHUB_OWNER,
        "github_repo":  Config.GITHUB_REPO,
        "has_token":    bool(Config.GITHUB_TOKEN),
    })


# ── Admin: per-roadmap breakdown ──────────────────────────────────────────────

@api.get("/admin/roadmaps")
@admin_required
def admin_roadmaps():
    c = get_conn()
    rows = c.execute("""
        SELECT r.slug, r.title, r.type, r.node_count, r.last_synced_at,
               COUNT(t.id) topic_count
        FROM roadmaps r
        LEFT JOIN topics t ON t.roadmap_id = r.id
        GROUP BY r.id
        ORDER BY r.type, r.title
    """).fetchall()
    return jsonify({"roadmaps": [dict(r) for r in rows]})


# ── Sync ─────────────────────────────────────────────────────────────────────

@api.get("/sync/status")
def sync_status():
    latest = ops.get_latest_sync()
    return jsonify({
        "live":   get_state(),
        "latest": latest.to_dict() if latest else None,
    })


@api.get("/sync/history")
def sync_history():
    limit = min(int(request.args.get("limit", 20)), 100)
    return jsonify({"history": [h.to_dict() for h in ops.get_sync_history(limit)]})


@api.post("/sync/trigger")
@admin_required
def sync_trigger():
    bust_cache()
    ok = trigger_manual()
    if ok:
        return jsonify({"status": "started"})
    return jsonify({"status": "already_running"}), 409


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 ADDITIONS — per-user progress + multi-roadmap selection
# ═══════════════════════════════════════════════════════════════════════════

@api.post("/progress/<int:topic_id>")
@login_required
def toggle_progress(topic_id):
    completed = (request.json or {}).get("completed", True)
    ops.toggle_topic_progress(session["user_id"], topic_id, completed)

    if completed:
        # Find which roadmap this topic belongs to so we can check for 100%
        row = get_conn().execute("SELECT roadmap_id FROM topics WHERE id=?", (topic_id,)).fetchone()
        if row:
            ops.maybe_advance_primary(session["user_id"], row["roadmap_id"])

    return jsonify({"status": "ok"})


@api.post("/roadmap/<slug>/reset-progress")
@login_required
def reset_roadmap_progress(slug):
    rm = ops.get_roadmap(slug)
    if not rm:
        return jsonify({"error": "not found"}), 404
    ops.reset_roadmap_progress(session["user_id"], rm.id)
    return jsonify({"status": "ok"})


@api.get("/user/roadmaps")
@login_required
def user_roadmaps():
    return jsonify({"roadmaps": ops.get_user_roadmaps(session["user_id"])})


@api.post("/user/roadmaps")
@login_required
def add_user_roadmap():
    data = request.json or {}
    rid = data.get("roadmap_id")
    if not rid:
        return jsonify({"error": "roadmap_id required"}), 400
    ops.add_user_roadmap(session["user_id"], rid, is_primary=bool(data.get("is_primary", False)))
    return jsonify({"status": "ok"})


@api.delete("/user/roadmaps/<int:roadmap_id>")
@login_required
def remove_user_roadmap(roadmap_id):
    ops.remove_user_roadmap(session["user_id"], roadmap_id)
    return jsonify({"status": "ok"})
