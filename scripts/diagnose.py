#!/usr/bin/env python3
"""
diagnose.py — Run this BEFORE starting the app to verify everything is working.

Usage:
    python diagnose.py

It will:
  1. Check your .env / ROADMAP_REPO_PATH configuration
  2. Scan the repo directory structure
  3. Parse a sample roadmap (python) end-to-end
  4. Show the resulting topic tree
  5. Report any problems found
"""
import os, sys, json
from pathlib import Path

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

os.environ["SYNC_ON_STARTUP"] = "false"

# ─── helpers ──────────────────────────────────────────────────────────────────
def ok(msg):  print(f"  \033[32m✅\033[0m  {msg}")
def warn(msg):print(f"  \033[33m⚠️ \033[0m  {msg}")
def err(msg): print(f"  \033[31m❌\033[0m  {msg}")
def hdr(msg): print(f"\n\033[1m{msg}\033[0m")

errors_found = 0

# ─── 1. Config ────────────────────────────────────────────────────────────────
hdr("1 / 5  Configuration")
from careercompass.config import Config

repo_path = Config.local_repo_path()
if Config.is_local_mode():
    ok(f"LOCAL mode — repo at: {repo_path}")
    roadmaps_dir = Config.roadmaps_dir()
    ok(f"Roadmaps dir: {roadmaps_dir}")
else:
    warn("LOCAL mode NOT active — ROADMAP_REPO_PATH is not set or path does not exist.")
    warn(f"  Value in .env: '{os.environ.get('ROADMAP_REPO_PATH','<not set>')}'")
    warn("  Falling back to REMOTE (GitHub API) mode.")
    if Config.GITHUB_TOKEN:
        ok(f"GITHUB_TOKEN is set ({Config.GITHUB_TOKEN[:8]}...)")
    else:
        warn("GITHUB_TOKEN not set — rate limited to 60 req/hr (may be too slow)")
    errors_found += 1

# ─── 2. Directory scan ────────────────────────────────────────────────────────
hdr("2 / 5  Repo directory scan")
if Config.is_local_mode():
    roadmaps_dir = Config.roadmaps_dir()
    if not roadmaps_dir.is_dir():
        err(f"Roadmaps directory not found: {roadmaps_dir}")
        err("  Is ROADMAP_REPO_PATH pointing to the repo root?")
        err("  Expected: <repo>/roadmaps/ (or src/data/roadmaps/ in older commits)")
        errors_found += 1
        sys.exit(1)

    all_slugs = sorted(d.name for d in roadmaps_dir.iterdir()
                       if d.is_dir() and not d.name.startswith("."))
    ok(f"Found {len(all_slugs)} roadmap directories")

    # count files per roadmap
    has_json = [s for s in all_slugs if (roadmaps_dir/s/f"{s}.json").exists()]
    has_md   = [s for s in all_slugs if (roadmaps_dir/s/f"{s}.md").exists()]
    has_content = [s for s in all_slugs if (roadmaps_dir/s/"content").is_dir()]

    ok(f"  with <slug>.json   : {len(has_json)}")
    ok(f"  with <slug>.md     : {len(has_md)}")
    ok(f"  with content/ dir  : {len(has_content)}")

    if len(has_content) == 0:
        err("No roadmap has a content/ directory — check the repo path!")
        errors_found += 1
else:
    warn("Skipping directory scan (remote mode)")
    all_slugs = []

# ─── 3. Parse a sample roadmap ────────────────────────────────────────────────
hdr("3 / 5  End-to-end parse test  (python roadmap)")
from careercompass.github_client.parser import (
    scan_content_dir, build_tree_from_json, parse_roadmap_md
)

if Config.is_local_mode():
    test_slug = "python" if "python" in all_slugs else (all_slugs[0] if all_slugs else None)
    if not test_slug:
        err("No roadmaps found to test!")
        errors_found += 1
    else:
        slug_dir    = Config.roadmaps_dir() / test_slug
        content_dir = slug_dir / "content"
        json_path   = slug_dir / f"{test_slug}.json"
        md_path     = slug_dir / f"{test_slug}.md"

        # Metadata
        if md_path.exists():
            meta = parse_roadmap_md(md_path.read_text(encoding="utf-8", errors="replace"))
            ok(f"  Metadata: title='{meta.get('title','?')}' tags={meta.get('tags',[])}")
        else:
            warn(f"  No {test_slug}.md found")

        # Content scan
        content_map = scan_content_dir(content_dir)
        if content_map:
            ok(f"  Content scan: {len(content_map)} topics found")
            sample = list(content_map.items())[:3]
            for nid, info in sample:
                ok(f"    • '{info['title']}' ({len(info['resources'])} resources)")
        else:
            err(f"  Content dir is empty or missing: {content_dir}")
            errors_found += 1

        # JSON hierarchy
        json_text = ""
        if json_path.exists():
            json_text = json_path.read_text(encoding="utf-8", errors="replace")
            ok(f"  Found {test_slug}.json ({len(json_text)} bytes)")
        else:
            warn(f"  No {test_slug}.json — will use flat topic list")

        tree = build_tree_from_json(json_text, content_map)
        if tree:
            ok(f"  Tree built: {len(tree)} root nodes")
            # print first 2 levels
            for root in tree[:3]:
                print(f"       [{root['topic_type']}] {root['title']}"
                      f"  ({len(root['resources'])} res, {len(root['children'])} children)")
                for ch in root["children"][:3]:
                    print(f"         └─ [{ch['topic_type']}] {ch['title']}"
                          f"  ({len(ch['resources'])} res)")
        else:
            err("  Tree is empty — nothing to store in DB!")
            errors_found += 1
else:
    warn("Skipping parse test (remote mode)")

# ─── 4. DB write test ─────────────────────────────────────────────────────────
hdr("4 / 5  Database write test")
from careercompass.db import init_db
from careercompass.db import operations as ops
init_db()

if Config.is_local_mode() and test_slug and tree:
    meta = {
        "slug": "__diag_test__", "title": "Diagnostic Test",
        "brief_title": "", "type": "skill", "description": "",
        "brief_desc": "", "icon": "🧪", "renderer": "graph",
        "is_new": False, "tags": [], "source_url": "",
    }
    rm_id   = ops.upsert_roadmap(meta)
    n_added = ops.replace_topics(rm_id, tree)
    from careercompass.db import get_conn
    get_conn().execute("DELETE FROM topics WHERE roadmap_id=?", (rm_id,))
    get_conn().execute("DELETE FROM roadmaps WHERE slug='__diag_test__'")
    get_conn().commit()
    ok(f"DB write OK — {n_added} topics written and cleaned up")
else:
    ok("DB init OK")

# ─── 5. Summary ───────────────────────────────────────────────────────────────
hdr("5 / 5  Summary")
stats = ops.get_stats()
ok(f"Current DB state: {stats['total']} roadmaps, "
   f"{stats['topics']} topics, {stats['resources']} resources")

if errors_found:
    print(f"\n\033[31m  {errors_found} issue(s) found — fix them before running the app.\033[0m\n")
    sys.exit(1)
else:
    print("\n\033[32m  All checks passed — run:  python app.py\033[0m\n")
