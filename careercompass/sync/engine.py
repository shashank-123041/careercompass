"""
sync/engine.py — Synchronisation orchestrator.

Data pipeline per roadmap:
  1. Read {slug}/content/*.md  → topic titles + resources  (always works)
  2. Read {slug}/{slug}.json   → hierarchy (optional enhancement)
  3. Merge → store in DB

The content/ directory is the ground truth. Even if the JSON is
absent or malformed, every roadmap will have its topics populated.
"""
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from careercompass.config import Config
from careercompass.db import get_conn, init_db
from careercompass.db import operations as ops
from careercompass.github_client.structure import get_structure
from careercompass.github_client.parser import (
    parse_roadmap_md,
    scan_content_dir,
    build_tree_from_json,
    node_id_from_filename,
    title_from_filename,
    parse_content_file,
    classify_roadmap_type,
)

log = logging.getLogger(__name__)

# ── Shared state ──────────────────────────────────────────────────────────────
_lock  = threading.Lock()
_state: dict = {
    "running": False, "mode": None, "trigger": None,
    "started_at": None, "current": None,
    "progress": 0, "total": 0, "done": 0,
    "last_status": None, "last_completed": None, "error": None,
}
_run_lock = threading.Lock()
_timer: Optional[threading.Timer] = None


def get_state() -> dict:
    with _lock:
        return dict(_state)


def _set(**kw):
    with _lock:
        _state.update(kw)


# ── Public API ────────────────────────────────────────────────────────────────

def start_scheduler():
    init_db()
    mode = "local" if Config.is_local_mode() else "remote"
    log.info("Sync mode: %s", mode)
    if Config.SYNC_ON_STARTUP and ops.is_stale(Config.STALE_AFTER_HOURS):
        log.info("Data is stale — scheduling startup sync in 2 s")
        _schedule(2, "startup")
    else:
        log.info("Data is fresh — next scheduled sync in %d h",
                 Config.SYNC_INTERVAL_HOURS)
        _schedule_next()


def trigger_manual() -> bool:
    if _state["running"]:
        return False
    _schedule(0, "manual")
    return True


def stop_scheduler():
    global _timer
    if _timer:
        _timer.cancel()
        _timer = None


# ── Scheduling ────────────────────────────────────────────────────────────────

def _schedule(delay: float, trigger: str):
    t = threading.Timer(delay, _run, kwargs={"trigger": trigger})
    t.daemon = True
    t.start()


def _schedule_next():
    global _timer
    if _timer:
        _timer.cancel()
    secs = Config.SYNC_INTERVAL_HOURS * 3600
    _timer = threading.Timer(secs, _run, kwargs={"trigger": "scheduled"})
    _timer.daemon = True
    _timer.start()
    log.info("Next sync in %.1f h", Config.SYNC_INTERVAL_HOURS)


# ── Core sync ─────────────────────────────────────────────────────────────────

def _run(trigger: str = "scheduled"):
    if not _run_lock.acquire(blocking=False):
        log.warning("Sync already running — skipping (%s)", trigger)
        return

    mode = "local" if Config.is_local_mode() else "remote"
    _set(running=True, mode=mode, trigger=trigger,
         started_at=_now(), current=None,
         progress=0, total=0, done=0, error=None)

    sync_id = ops.start_sync(trigger=trigger, mode=mode)
    stats = {"status": "running", "found": 0, "added": 0, "updated": 0,
             "topics": 0, "resources": 0, "error": None}
    try:
        if mode == "local":
            _sync_local(sync_id, stats)
        else:
            _sync_remote(sync_id, stats)
        stats["status"] = "success"
    except Exception as e:
        log.exception("Sync failed: %s", e)
        stats["status"] = "failed"
        stats["error"]  = str(e)
        _set(error=str(e))
    finally:
        ops.finish_sync(sync_id, stats)
        _set(running=False, current=None, progress=100,
             last_status=stats["status"], last_completed=_now())
        _run_lock.release()
        _schedule_next()

    log.info("Sync done [%s] found=%d topics=%d resources=%d",
             stats["status"], stats["found"],
             stats["topics"], stats["resources"])


# ── LOCAL mode ────────────────────────────────────────────────────────────────

def _sync_local(sync_id: int, stats: dict):
    roadmaps_dir = Config.roadmaps_dir()
    if not roadmaps_dir.is_dir():
        raise RuntimeError(
            f"roadmaps directory not found: {roadmaps_dir}\n"
            f"Check ROADMAP_REPO_PATH in your .env file.\n"
            f"Expected: <repo_root>/roadmaps/ (or src/data/roadmaps/ in older commits)"
        )

    # Filter out slugs that are not standalone roadmaps:
    # - *-beginner variants are beginner-mode views of an existing roadmap
    #   (e.g. git-github-beginner links back to /git-github), not separate entries.
    # - Directories with no .json AND no content/ are placeholders/drafts.
    _SKIP_SLUG_SUFFIXES = ("-beginner",)
    slugs = sorted(
        d.name for d in roadmaps_dir.iterdir()
        if d.is_dir()
        and not d.name.startswith(".")
        and not any(d.name.endswith(s) for s in _SKIP_SLUG_SUFFIXES)
    )
    stats["found"] = len(slugs)
    _set(total=len(slugs), progress=2)
    log.info("Local sync: %d roadmaps in %s", len(slugs), roadmaps_dir)

    existing = ops.get_all_slugs()

    for i, slug in enumerate(slugs):
        _set(current=slug, done=i,
             progress=int(5 + 90 * i / max(len(slugs), 1)))
        try:
            _process_local_roadmap(slug, roadmaps_dir, existing, stats)
        except Exception as e:
            log.error("Error processing '%s': %s", slug, e, exc_info=True)

    log.info("Local sync complete: %d added, %d updated",
             stats["added"], stats["updated"])


def _process_local_roadmap(
    slug: str,
    roadmaps_dir: Path,
    existing: Dict[str, int],
    stats: dict,
):
    slug_dir = roadmaps_dir / slug

    # ── Step 1: metadata from {slug}.md ──────────────────────────────────
    meta: dict = {}
    md_path = slug_dir / f"{slug}.md"
    json_path = slug_dir / f"{slug}.json"

    # The upstream repo no longer ships <slug>.md / <slug>.json (hierarchy +
    # metadata). When the clone lacks them, restore both from the last commit
    # that had them — see github_client/structure.py. Local files always win.
    struct_json, struct_md = "", ""
    if not md_path.exists() or not json_path.exists():
        struct_json, struct_md = get_structure(slug)

    md_text = ""
    if md_path.exists():
        md_text = md_path.read_text(encoding="utf-8", errors="replace")
    elif struct_md:
        md_text = struct_md
    if md_text:
        try:
            meta = parse_roadmap_md(md_text)
        except Exception as e:
            log.warning("'%s': could not parse .md: %s", slug, e)

    if not meta.get("title"):
        meta["title"] = slug.replace("-", " ").title()

    meta["slug"]       = slug
    meta["type"]       = classify_roadmap_type(slug, meta.get("tags", []))
    meta["icon"]       = Config.SLUG_ICONS.get(slug, "📚")
    meta["source_url"] = f"https://roadmap.sh/{slug}"

    # ── Step 2: topic content from content/ directory ─────────────────────
    content_dir = slug_dir / "content"
    content_map = scan_content_dir(content_dir)
    log.debug("'%s': %d content files found", slug, len(content_map))

    # ── Step 3: hierarchy from {slug}.json ────────────────────────────────
    json_text = ""
    if json_path.exists():
        try:
            json_text = json_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            log.warning("'%s': could not read .json: %s", slug, e)
    elif struct_json:
        json_text = struct_json
        log.debug("'%s': using structure from pinned commit", slug)

    tree = build_tree_from_json(json_text, content_map)
    log.debug("'%s': tree has %d root nodes, %d total",
              slug, len(tree), _count_nodes(tree))

    # ── Step 4: persist ───────────────────────────────────────────────────
    rm_id = ops.upsert_roadmap(meta)
    stats["added" if slug not in existing else "updated"] += 1

    if tree:
        topics_added = ops.replace_topics(rm_id, tree)
        stats["topics"]    += topics_added
        stats["resources"] += _count_resources(tree)
        ops.mark_sync_done(slug, _count_nodes(tree))
        log.info("'%s': %d topics, %d resources stored",
                 slug, topics_added, _count_resources(tree))
    else:
        log.warning("'%s': no topics extracted (no content/ dir and no JSON)", slug)


# ── REMOTE mode ───────────────────────────────────────────────────────────────

def _sync_remote(sync_id: int, stats: dict):
    from careercompass.github_client import api as gh

    slugs = gh.list_roadmap_slugs()
    if not slugs:
        raise RuntimeError(
            "No roadmap slugs returned by GitHub API. "
            "Check GITHUB_TOKEN and network access."
        )

    # Same filter as local sync: skip beginner-variant slugs
    _SKIP_SLUG_SUFFIXES = ("-beginner",)
    slugs = [s for s in slugs if not any(s.endswith(x) for x in _SKIP_SLUG_SUFFIXES)]

    stats["found"] = len(slugs)
    _set(total=len(slugs), progress=2)
    log.info("Remote sync: %d slugs discovered", len(slugs))

    existing = ops.get_all_slugs()

    for i, slug in enumerate(slugs):
        _set(current=slug, done=i,
             progress=int(5 + 90 * i / max(len(slugs), 1)))
        try:
            _process_remote_roadmap(slug, existing, stats, gh)
            time.sleep(0.15)
        except Exception as e:
            log.error("Remote error for '%s': %s", slug, e, exc_info=True)


def _process_remote_roadmap(slug, existing, stats, gh):
    # Metadata
    md_text = gh.fetch_roadmap_md(slug) or ""
    json_text = gh.fetch_roadmap_json(slug) or ""
    if not md_text or not json_text:
        struct_json, struct_md = get_structure(slug)
        md_text = md_text or struct_md
        json_text = json_text or struct_json
    meta = parse_roadmap_md(md_text) if md_text else {}
    if not meta.get("title"):
        meta["title"] = slug.replace("-", " ").title()
    meta["slug"]       = slug
    meta["type"]       = classify_roadmap_type(slug, meta.get("tags", []))
    meta["icon"]       = Config.SLUG_ICONS.get(slug, "📚")
    meta["source_url"] = f"https://roadmap.sh/{slug}"

    # Content files
    content_map: Dict[str, dict] = {}
    for cf in gh.list_content_files(slug):
        node_id = node_id_from_filename(cf["name"])
        title   = title_from_filename(cf["name"])
        if node_id and title:
            text = gh.fetch_content_file(slug, cf["name"]) or ""
            resources, desc = parse_content_file(text) if text else ([], "")
            content_map[node_id] = {
                "node_id": node_id, "title": title,
                "description": desc, "resources": resources,
            }

    # JSON hierarchy
    tree = build_tree_from_json(json_text, content_map)

    # Persist
    rm_id = ops.upsert_roadmap(meta)
    stats["added" if slug not in existing else "updated"] += 1

    if tree:
        topics_added = ops.replace_topics(rm_id, tree)
        stats["topics"]    += topics_added
        stats["resources"] += _count_resources(tree)
        ops.mark_sync_done(slug, _count_nodes(tree))


# ── Utilities ─────────────────────────────────────────────────────────────────

def _count_nodes(tree: List[dict]) -> int:
    n = 0
    for node in tree:
        n += 1 + _count_nodes(node.get("children", []))
    return n


def _count_resources(tree: List[dict]) -> int:
    n = 0
    for node in tree:
        n += len(node.get("resources", []))
        n += _count_resources(node.get("children", []))
    return n


def _now() -> str:
    return datetime.utcnow().isoformat(sep=" ", timespec="seconds")