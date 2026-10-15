"""
db/operations.py — Every database operation lives here.
No SQL is written anywhere else in the project.

FIX (v3.1): upsert_roadmap now uses SQLite's atomic
INSERT OR REPLACE pattern — eliminates the SELECT→INSERT race
condition that caused "UNIQUE constraint failed: roadmaps.slug"
when two processes synced concurrently (Flask debug reloader bug).

STEP 1 ADDITION: user accounts, per-user roadmap selection, and
per-user topic progress (replacing the localStorage approach).
"""
import json, logging
from datetime import datetime
from typing import Dict, List, Optional
from careercompass.db import get_conn
from careercompass.models import Roadmap, Topic, Resource, SyncRecord, User

log = logging.getLogger(__name__)

# ── helpers ───────────────────────────────────────────────────────────────────

def _now(): return datetime.utcnow().isoformat(sep=" ", timespec="seconds")

def _row_roadmap(r) -> Roadmap:
    tags = r["tags"]
    if isinstance(tags, str):
        try: tags = json.loads(tags)
        except: tags = []
    return Roadmap(
        id=r["id"], slug=r["slug"], title=r["title"],
        brief_title=r["brief_title"] or "", type=r["type"],
        description=r["description"] or "", brief_desc=r["brief_desc"] or "",
        icon=r["icon"] or "📚", renderer=r["renderer"] or "graph",
        is_new=bool(r["is_new"]), tags=tags or [],
        source_url=r["source_url"] or "", node_count=r["node_count"] or 0,
        last_synced_at=r["last_synced_at"],
        created_at=r["created_at"], updated_at=r["updated_at"],
    )

def _row_topic(r) -> Topic:
    return Topic(
        id=r["id"], roadmap_id=r["roadmap_id"], node_id=r["node_id"] or "",
        title=r["title"], description=r["description"] or "",
        topic_type=r["topic_type"] or "topic", parent_id=r["parent_id"],
        order_index=r["order_index"] or 0, depth=r["depth"] or 0,
        is_optional=bool(r["is_optional"]),
        pos_x=r["pos_x"] or 0.0, pos_y=r["pos_y"] or 0.0,
    )

def _row_resource(r) -> Resource:
    return Resource(
        id=r["id"], topic_id=r["topic_id"], title=r["title"],
        url=r["url"], res_type=r["res_type"] or "article",
        order_index=r["order_index"] or 0,
    )

def _row_user(r) -> User:
    skills = r["resume_skills"]
    if isinstance(skills, str):
        try: skills = json.loads(skills)
        except: skills = {}
    return User(
        id=r["id"], firebase_uid=r["firebase_uid"], email=r["email"],
        name=r["name"] or "", avatar_url=r["avatar_url"] or "",
        resume_skills=skills or {}, onboarded=bool(r["onboarded"]),
        created_at=r["created_at"], updated_at=r["updated_at"],
    )

# ── roadmap CRUD ──────────────────────────────────────────────────────────────

def upsert_roadmap(data: dict) -> int:
    """
    Atomically insert-or-update a roadmap row.

    Uses SQLite's ON CONFLICT(slug) DO UPDATE clause (available since
    SQLite 3.24 / Python 3.8) so there is no SELECT→INSERT race window.
    Safe to call from multiple processes simultaneously.
    """
    c   = get_conn()
    now = _now()
    tags_json = json.dumps(data.get("tags", []))

    c.execute("""
        INSERT INTO roadmaps
            (slug, title, brief_title, type, description, brief_desc,
             icon, renderer, is_new, tags, source_url, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(slug) DO UPDATE SET
            title       = excluded.title,
            brief_title = excluded.brief_title,
            type        = excluded.type,
            description = excluded.description,
            brief_desc  = excluded.brief_desc,
            icon        = excluded.icon,
            renderer    = excluded.renderer,
            is_new      = excluded.is_new,
            tags        = excluded.tags,
            source_url  = excluded.source_url,
            updated_at  = excluded.updated_at
    """, (
        data["slug"], data["title"], data.get("brief_title", ""),
        data.get("type", "skill"), data.get("description", ""),
        data.get("brief_desc", ""), data.get("icon", "📚"),
        data.get("renderer", "graph"), int(data.get("is_new", False)),
        tags_json, data.get("source_url", ""), now, now,
    ))
    c.commit()

    # Return the id (works for both insert and update)
    row = c.execute("SELECT id FROM roadmaps WHERE slug=?",
                    (data["slug"],)).fetchone()
    return row["id"]


def mark_sync_done(slug: str, node_count: int):
    c = get_conn()
    c.execute(
        "UPDATE roadmaps SET last_synced_at=?, node_count=?, updated_at=? WHERE slug=?",
        (_now(), node_count, _now(), slug),
    )
    c.commit()


def get_roadmap(slug: str) -> Optional[Roadmap]:
    r = get_conn().execute("SELECT * FROM roadmaps WHERE slug=?", (slug,)).fetchone()
    return _row_roadmap(r) if r else None


def get_all_roadmaps(type_filter: Optional[str] = None) -> List[Roadmap]:
    c = get_conn()
    if type_filter:
        rows = c.execute(
            "SELECT * FROM roadmaps WHERE type=? ORDER BY title", (type_filter,)
        ).fetchall()
    else:
        rows = c.execute("SELECT * FROM roadmaps ORDER BY type, title").fetchall()
    return [_row_roadmap(r) for r in rows]


def get_all_slugs() -> Dict[str, int]:
    rows = get_conn().execute("SELECT slug, id FROM roadmaps").fetchall()
    return {r["slug"]: r["id"] for r in rows}

# ── topic CRUD ────────────────────────────────────────────────────────────────

def replace_topics(roadmap_id: int, tree: List[dict]) -> int:
    """Delete all existing topics for a roadmap, then insert the fresh tree."""
    c = get_conn()
    c.execute("DELETE FROM topics WHERE roadmap_id=?", (roadmap_id,))
    count = _insert_tree(roadmap_id, tree, None, 0, c)
    c.commit()
    return count


def _insert_tree(roadmap_id, nodes, parent_id, depth, c) -> int:
    count = 0
    for idx, n in enumerate(nodes):
        cur = c.execute("""
            INSERT INTO topics
                (roadmap_id, node_id, title, description, topic_type,
                 parent_id, order_index, depth, is_optional, pos_x, pos_y, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            roadmap_id, n.get("node_id", ""), n["title"],
            n.get("description", ""), n.get("topic_type", "topic"),
            parent_id, idx, depth,
            int(n.get("is_optional", False)),
            n.get("pos_x", 0.0), n.get("pos_y", 0.0), _now(),
        ))
        tid = cur.lastrowid
        count += 1
        for ri, res in enumerate(n.get("resources", [])):
            c.execute("""
                INSERT INTO resources (topic_id, title, url, res_type, order_index, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (tid, res["title"], res["url"],
                  res.get("res_type", "article"), ri, _now()))
        count += _insert_tree(roadmap_id, n.get("children", []), tid, depth + 1, c)
    return count


def get_topic_tree(roadmap_id: int, user_id: Optional[int] = None) -> List[Topic]:
    """
    Builds the topic tree for a roadmap. If user_id is given, each Topic's
    to_dict() output can be annotated with completion state by the caller
    using get_completed_topic_ids() — kept separate here to avoid coupling
    the base tree-builder to auth.
    """
    c = get_conn()
    rows = c.execute(
        "SELECT * FROM topics WHERE roadmap_id=? ORDER BY depth, order_index",
        (roadmap_id,),
    ).fetchall()
    if not rows:
        return []
    tmap: Dict[int, Topic] = {r["id"]: _row_topic(r) for r in rows}

    # Attach resources in one query
    ids = list(tmap.keys())
    if ids:
        ph = ",".join("?" * len(ids))
        for rrow in c.execute(
            f"SELECT * FROM resources WHERE topic_id IN ({ph}) ORDER BY order_index", ids
        ).fetchall():
            res = _row_resource(rrow)
            if res.topic_id in tmap:
                tmap[res.topic_id].resources.append(res)

    roots = []
    for t in tmap.values():
        if t.parent_id is None:
            roots.append(t)
        elif t.parent_id in tmap:
            tmap[t.parent_id].children.append(t)
    return roots


def search_topics(q: str, roadmap_id: Optional[int] = None) -> list:
    c = get_conn()
    like = f"%{q}%"
    if roadmap_id:
        rows = c.execute("""
            SELECT t.*, r.slug roadmap_slug, r.title roadmap_title
            FROM topics t JOIN roadmaps r ON r.id = t.roadmap_id
            WHERE t.roadmap_id=? AND t.title LIKE ?
            ORDER BY t.depth, t.order_index LIMIT 50
        """, (roadmap_id, like)).fetchall()
    else:
        rows = c.execute("""
            SELECT t.*, r.slug roadmap_slug, r.title roadmap_title
            FROM topics t JOIN roadmaps r ON r.id = t.roadmap_id
            WHERE t.title LIKE ?
            ORDER BY t.depth, t.order_index LIMIT 50
        """, (like,)).fetchall()
    return [dict(r) for r in rows]

# ── sync history ──────────────────────────────────────────────────────────────

def start_sync(trigger="scheduled", mode="local") -> int:
    c = get_conn()
    cur = c.execute(
        "INSERT INTO sync_history (started_at, status, trigger, mode) VALUES (?,?,?,?)",
        (_now(), "running", trigger, mode),
    )
    c.commit()
    return cur.lastrowid


def finish_sync(sync_id: int, stats: dict):
    c = get_conn()
    c.execute("""
        UPDATE sync_history
        SET completed_at=?, status=?, roadmaps_found=?, roadmaps_added=?,
            roadmaps_updated=?, topics_added=?, resources_added=?, error_message=?
        WHERE id=?
    """, (
        _now(), stats.get("status", "success"),
        stats.get("found", 0), stats.get("added", 0),
        stats.get("updated", 0), stats.get("topics", 0),
        stats.get("resources", 0), stats.get("error"),
        sync_id,
    ))
    c.commit()


def get_sync_history(limit=20) -> List[SyncRecord]:
    rows = get_conn().execute(
        "SELECT * FROM sync_history ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [
        SyncRecord(
            id=r["id"], started_at=r["started_at"],
            completed_at=r["completed_at"], status=r["status"],
            trigger=r["trigger"], mode=r["mode"],
            roadmaps_found=r["roadmaps_found"],
            roadmaps_added=r["roadmaps_added"],
            roadmaps_updated=r["roadmaps_updated"],
            topics_added=r["topics_added"],
            resources_added=r["resources_added"],
            error_message=r["error_message"],
        )
        for r in rows
    ]


def get_latest_sync() -> Optional[SyncRecord]:
    h = get_sync_history(1)
    return h[0] if h else None

# ── stats ─────────────────────────────────────────────────────────────────────

def get_stats() -> dict:
    c = get_conn()
    total     = c.execute("SELECT COUNT(*) FROM roadmaps").fetchone()[0]
    roles     = c.execute("SELECT COUNT(*) FROM roadmaps WHERE type='role'").fetchone()[0]
    skills    = c.execute("SELECT COUNT(*) FROM roadmaps WHERE type='skill'").fetchone()[0]
    topics    = c.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
    resources = c.execute("SELECT COUNT(*) FROM resources").fetchone()[0]
    synced    = c.execute(
        "SELECT COUNT(*) FROM roadmaps WHERE last_synced_at IS NOT NULL"
    ).fetchone()[0]
    last = get_latest_sync()
    return {
        "total": total, "roles": roles, "skills": skills,
        "topics": topics, "resources": resources, "synced": synced,
        "last_sync": last.to_dict() if last else None,
    }


def is_stale(hours=48) -> bool:
    c = get_conn()
    if c.execute("SELECT COUNT(*) FROM roadmaps").fetchone()[0] == 0:
        return True
    latest = get_latest_sync()
    if not latest or latest.status not in ("success", "partial"):
        return True
    if not latest.completed_at:
        return True
    try:
        diff = (
            datetime.utcnow() - datetime.fromisoformat(latest.completed_at)
        ).total_seconds() / 3600
        return diff > hours
    except Exception:
        return True


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 ADDITIONS — users, user_roadmaps, user_topic_progress
# ═══════════════════════════════════════════════════════════════════════════

# ── users ─────────────────────────────────────────────────────────────────────

def upsert_user(firebase_uid: str, email: str, name: str = "", avatar_url: str = "") -> User:
    """Called after Firebase ID token verification on every login."""
    c = get_conn()
    now = _now()
    c.execute("""
        INSERT INTO users (firebase_uid, email, name, avatar_url, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(firebase_uid) DO UPDATE SET
            email=excluded.email, name=excluded.name,
            avatar_url=excluded.avatar_url, updated_at=excluded.updated_at
    """, (firebase_uid, email, name, avatar_url, now, now))
    c.commit()
    return get_user_by_uid(firebase_uid)


def get_user_by_uid(firebase_uid: str) -> Optional[User]:
    r = get_conn().execute(
        "SELECT * FROM users WHERE firebase_uid=?", (firebase_uid,)
    ).fetchone()
    return _row_user(r) if r else None


def get_user_by_id(user_id: int) -> Optional[User]:
    r = get_conn().execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return _row_user(r) if r else None


def get_all_users_with_resume_skills() -> List[User]:
    """
    Bulk listing for ml/opportunity_bootstrap.py's pseudo-label generation —
    every user whose resume_skills column has actually been populated
    (i.e. not the DEFAULT '{}' set at row creation, and not literally
    empty/NULL). Filtered at the SQL level rather than fetching every user
    and checking in Python, since a user with no resume data can't
    contribute any bootstrap rows anyway (build_row()/generate() would
    just skip them after the fact) — no reason to pull those rows at all.
    """
    rows = get_conn().execute("""
        SELECT * FROM users
        WHERE resume_skills IS NOT NULL
          AND resume_skills != ''
          AND resume_skills != '{}'
        ORDER BY id
    """).fetchall()
    return [_row_user(r) for r in rows]


def save_resume_skills(user_id: int, skills: dict):
    """Stores the (possibly user-edited) resume_parser.py output JSON."""
    c = get_conn()
    c.execute("UPDATE users SET resume_skills=?, updated_at=? WHERE id=?",
              (json.dumps(skills), _now(), user_id))
    c.commit()


def complete_onboarding(user_id: int):
    c = get_conn()
    c.execute("UPDATE users SET onboarded=1, updated_at=? WHERE id=?", (_now(), user_id))
    c.commit()


def update_user_name(user_id: int, name: str) -> None:
    c = get_conn()
    c.execute("UPDATE users SET name=?, updated_at=? WHERE id=?", (name.strip(), _now(), user_id))
    c.commit()

# ── user <-> roadmap selection ─────────────────────────────────────────────────

def add_user_roadmap(user_id: int, roadmap_id: int, is_primary: bool = False) -> None:
    c = get_conn()
    if is_primary:
        c.execute("UPDATE user_roadmaps SET is_primary=0 WHERE user_id=?", (user_id,))
    c.execute("""
        INSERT INTO user_roadmaps (user_id, roadmap_id, is_primary, selected_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(user_id, roadmap_id) DO UPDATE SET is_primary=excluded.is_primary
    """, (user_id, roadmap_id, int(is_primary), _now()))
    c.commit()


def remove_user_roadmap(user_id: int, roadmap_id: int) -> None:
    c = get_conn()
    c.execute("DELETE FROM user_roadmaps WHERE user_id=? AND roadmap_id=?", (user_id, roadmap_id))
    c.commit()


def get_user_roadmaps(user_id: int) -> list:
    rows = get_conn().execute("""
        SELECT r.*, ur.is_primary, ur.selected_at
        FROM user_roadmaps ur JOIN roadmaps r ON r.id = ur.roadmap_id
        WHERE ur.user_id=? ORDER BY ur.is_primary DESC, ur.selected_at
    """, (user_id,)).fetchall()
    out = []
    for r in rows:
        rm = _row_roadmap(r).to_dict()
        rm["is_primary"] = bool(r["is_primary"])
        rm["progress_pct"] = get_roadmap_progress_pct(user_id, r["id"])
        out.append(rm)
    return out


def is_roadmap_selected(user_id: int, roadmap_id: int) -> bool:
    row = get_conn().execute(
        "SELECT 1 FROM user_roadmaps WHERE user_id=? AND roadmap_id=?",
        (user_id, roadmap_id),
    ).fetchone()
    return row is not None

# ── per-user progress ──────────────────────────────────────────────────────────

def get_incomplete_topic_titles(user_id: int, roadmap_id: int) -> List[str]:
    """
    Titles of non-section topics in this roadmap the user has NOT yet
    completed — i.e. what they're actively learning right now, as opposed
    to get_completed_topic_ids() which returns the opposite set. Returns
    raw titles (e.g. "Docker", "Learn Kubernetes Basics"), not skill slugs —
    matching those to canonical slugs is a taxonomy concern, not a DB one,
    and stays out of this module on purpose (see get_topic_tree()'s
    docstring on the same separation principle).
 
    Excludes 'section' topics for the same reason get_roadmap_progress_pct
    does — they're headers, not individually completable/learnable items.
    """
    rows = get_conn().execute("""
        SELECT t.title FROM topics t
        WHERE t.roadmap_id=? AND t.topic_type != 'section'
          AND t.id NOT IN (
              SELECT topic_id FROM user_topic_progress WHERE user_id=?
          )
        ORDER BY t.depth, t.order_index
    """, (roadmap_id, user_id)).fetchall()
    return [r["title"] for r in rows]

def toggle_topic_progress(user_id: int, topic_id: int, completed: bool) -> None:
    c = get_conn()
    if completed:
        c.execute("""
            INSERT INTO user_topic_progress (user_id, topic_id, completed_at)
            VALUES (?, ?, ?) ON CONFLICT(user_id, topic_id) DO NOTHING
        """, (user_id, topic_id, _now()))
    else:
        c.execute("DELETE FROM user_topic_progress WHERE user_id=? AND topic_id=?",
                  (user_id, topic_id))
    c.commit()


def get_completed_topic_ids(user_id: int, roadmap_id: int) -> set:
    rows = get_conn().execute("""
        SELECT p.topic_id FROM user_topic_progress p
        JOIN topics t ON t.id = p.topic_id
        WHERE p.user_id=? AND t.roadmap_id=?
    """, (user_id, roadmap_id)).fetchall()
    return {r["topic_id"] for r in rows}


def get_roadmap_progress_pct(user_id: int, roadmap_id: int) -> float:
    """
    Section-type topics are headers, not individually completable items —
    excluded from both sides of the ratio so this matches what the UI
    actually lets a user check off.
    """
    c = get_conn()
    total = c.execute(
        "SELECT COUNT(*) FROM topics WHERE roadmap_id=? AND topic_type != 'section'",
        (roadmap_id,),
    ).fetchone()[0]
    if total == 0:
        return 0.0
    done = c.execute("""
        SELECT COUNT(*) FROM user_topic_progress p
        JOIN topics t ON t.id = p.topic_id
        WHERE p.user_id=? AND t.roadmap_id=? AND t.topic_type != 'section'
    """, (user_id, roadmap_id)).fetchone()[0]
    return round(done / total * 100, 1)


def reset_roadmap_progress(user_id: int, roadmap_id: int) -> None:
    """Clears all completed-topic records a user has for one roadmap."""
    c = get_conn()
    c.execute("""
        DELETE FROM user_topic_progress
        WHERE user_id=? AND topic_id IN (
            SELECT id FROM topics WHERE roadmap_id=?
        )
    """, (user_id, roadmap_id))
    c.commit()


def maybe_advance_primary(user_id: int, roadmap_id: int) -> None:
    """
    If the roadmap that just hit 100% is the user's current primary,
    promote the next-selected roadmap (chronologically) to primary.
    No-op if this roadmap isn't primary, isn't at 100%, or there's no
    other roadmap to hand primary status to.
    """
    c = get_conn()
    row = c.execute(
        "SELECT is_primary FROM user_roadmaps WHERE user_id=? AND roadmap_id=?",
        (user_id, roadmap_id),
    ).fetchone()
    if not row or not row["is_primary"]:
        return
    if get_roadmap_progress_pct(user_id, roadmap_id) < 100:
        return

    nxt = c.execute("""
        SELECT roadmap_id FROM user_roadmaps
        WHERE user_id=? AND roadmap_id != ?
        ORDER BY selected_at ASC LIMIT 1
    """, (user_id, roadmap_id)).fetchone()
    if not nxt:
        return  # nothing else to promote — leave current as primary

    now = _now()
    c.execute("UPDATE user_roadmaps SET is_primary=0 WHERE user_id=? AND roadmap_id=?",
              (user_id, roadmap_id))
    c.execute("UPDATE user_roadmaps SET is_primary=1 WHERE user_id=? AND roadmap_id=?",
              (user_id, nxt["roadmap_id"]))
    c.commit()