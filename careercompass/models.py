"""
models.py — Data models and SQLite schema.

STEP 1 ADDITION: users, user_roadmaps, user_topic_progress tables added
to support Firebase-authenticated accounts, multi-roadmap selection,
and server-side (per-user) progress tracking — replacing the localStorage
approach originally proposed for roadmap_detail.html.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List

# ── DDL ───────────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS roadmaps (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    slug           TEXT    NOT NULL UNIQUE,
    title          TEXT    NOT NULL,
    brief_title    TEXT    DEFAULT '',
    type           TEXT    NOT NULL DEFAULT 'skill',   -- 'role' | 'skill'
    description    TEXT    DEFAULT '',
    brief_desc     TEXT    DEFAULT '',
    icon           TEXT    DEFAULT '📚',
    renderer       TEXT    DEFAULT 'graph',            -- 'graph' | 'editor'
    is_new         INTEGER DEFAULT 0,
    tags           TEXT    DEFAULT '[]',               -- JSON array of strings
    source_url     TEXT    DEFAULT '',
    node_count     INTEGER DEFAULT 0,
    last_synced_at TEXT,
    created_at     TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS topics (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    roadmap_id   INTEGER NOT NULL REFERENCES roadmaps(id) ON DELETE CASCADE,
    node_id      TEXT,                              -- original node id from JSON
    title        TEXT    NOT NULL,
    description  TEXT    DEFAULT '',
    topic_type   TEXT    DEFAULT 'topic',           -- 'section' | 'topic' | 'subtopic'
    parent_id    INTEGER REFERENCES topics(id) ON DELETE CASCADE,
    order_index  INTEGER DEFAULT 0,
    depth        INTEGER DEFAULT 0,
    is_optional  INTEGER DEFAULT 0,
    pos_x        REAL    DEFAULT 0,
    pos_y        REAL    DEFAULT 0,
    created_at   TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_topics_roadmap ON topics(roadmap_id);
CREATE INDEX IF NOT EXISTS idx_topics_parent  ON topics(parent_id);
CREATE INDEX IF NOT EXISTS idx_topics_node    ON topics(node_id);

CREATE TABLE IF NOT EXISTS resources (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id     INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    title        TEXT    NOT NULL,
    url          TEXT    NOT NULL,
    res_type     TEXT    DEFAULT 'article',   -- official|article|video|course|opensource|feed
    order_index  INTEGER DEFAULT 0,
    created_at   TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_resources_topic ON resources(topic_id);

CREATE TABLE IF NOT EXISTS opportunities (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT    NOT NULL,
    external_id     TEXT    NOT NULL,
    title           TEXT    NOT NULL,
    org             TEXT    DEFAULT '',
    opp_type        TEXT    NOT NULL,
    url             TEXT    NOT NULL,
    location        TEXT    DEFAULT '',
    work_mode       TEXT    DEFAULT '',
    is_remote       INTEGER DEFAULT 0,
    date_label      TEXT    DEFAULT '',
    deadline        TEXT,
    compensation    TEXT    DEFAULT '',
    duration_info   TEXT    DEFAULT '',
    description     TEXT    DEFAULT '',
    thumbnail_url   TEXT    DEFAULT '',
    is_fallback     INTEGER DEFAULT 0,
    scraped_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE(source, external_id)
);
CREATE INDEX IF NOT EXISTS idx_opp_type ON opportunities(opp_type);
CREATE INDEX IF NOT EXISTS idx_opp_deadline ON opportunities(deadline);

CREATE TABLE IF NOT EXISTS opportunity_skills (
    opportunity_id INTEGER NOT NULL REFERENCES opportunities(id) ON DELETE CASCADE,
    skill_slug     TEXT    NOT NULL,
    PRIMARY KEY (opportunity_id, skill_slug)
);

CREATE TABLE IF NOT EXISTS opportunity_sync_history (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at          TEXT    NOT NULL DEFAULT (datetime('now')),
    completed_at        TEXT,
    status              TEXT    DEFAULT 'running',   -- running|success|partial|failed
    trigger             TEXT    DEFAULT 'scheduled',  -- scheduled|manual|startup
    sources_total       INTEGER DEFAULT 0,
    total_fetched       INTEGER DEFAULT 0,
    total_upserted      INTEGER DEFAULT 0,
    pruned_expired      INTEGER DEFAULT 0,
    pruned_stale        INTEGER DEFAULT 0,
    fallback_sources    TEXT    DEFAULT '[]',   -- JSON array of domain names that used fallback data
    error_message       TEXT
);

CREATE TABLE IF NOT EXISTS sync_history (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at          TEXT    NOT NULL DEFAULT (datetime('now')),
    completed_at        TEXT,
    status              TEXT    DEFAULT 'running',
    trigger             TEXT    DEFAULT 'scheduled',
    mode                TEXT    DEFAULT 'local',   -- 'local' | 'remote'
    roadmaps_found      INTEGER DEFAULT 0,
    roadmaps_added      INTEGER DEFAULT 0,
    roadmaps_updated    INTEGER DEFAULT 0,
    topics_added        INTEGER DEFAULT 0,
    resources_added     INTEGER DEFAULT 0,
    error_message       TEXT
);

-- ═══════════════════════════════════════════════════════════════════════════
-- STEP 1 ADDITIONS — Auth, multi-roadmap selection, per-user progress
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS users (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    firebase_uid   TEXT    NOT NULL UNIQUE,
    email          TEXT    NOT NULL UNIQUE,
    name           TEXT    DEFAULT '',
    avatar_url     TEXT    DEFAULT '',
    resume_skills  TEXT    DEFAULT '{}',    -- raw+edited resume_parser.py output JSON
    onboarded      INTEGER DEFAULT 0,       -- has completed signup role-selection step
    created_at     TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS user_roadmaps (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    roadmap_id    INTEGER NOT NULL REFERENCES roadmaps(id) ON DELETE CASCADE,
    is_primary    INTEGER DEFAULT 0,        -- the one chosen at signup
    selected_at   TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE(user_id, roadmap_id)
);
CREATE INDEX IF NOT EXISTS idx_user_roadmaps_user ON user_roadmaps(user_id);

CREATE TABLE IF NOT EXISTS user_topic_progress (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    topic_id      INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    completed_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE(user_id, topic_id)
);
CREATE INDEX IF NOT EXISTS idx_progress_user  ON user_topic_progress(user_id);
CREATE INDEX IF NOT EXISTS idx_progress_topic ON user_topic_progress(topic_id);
"""

# ── Python dataclasses ────────────────────────────────────────────────────────

@dataclass
class Roadmap:
    id: int = 0
    slug: str = ""
    title: str = ""
    brief_title: str = ""
    type: str = "skill"
    description: str = ""
    brief_desc: str = ""
    icon: str = "📚"
    renderer: str = "graph"
    is_new: bool = False
    tags: list = field(default_factory=list)
    source_url: str = ""
    node_count: int = 0
    last_synced_at: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id, "slug": self.slug,
            "title": self.title, "brief_title": self.brief_title,
            "type": self.type, "description": self.description,
            "brief_desc": self.brief_desc, "icon": self.icon,
            "renderer": self.renderer, "is_new": self.is_new,
            "tags": self.tags, "source_url": self.source_url,
            "node_count": self.node_count,
            "last_synced_at": self.last_synced_at,
            "created_at": self.created_at, "updated_at": self.updated_at,
        }


@dataclass
class Topic:
    id: int = 0
    roadmap_id: int = 0
    node_id: str = ""
    title: str = ""
    description: str = ""
    topic_type: str = "topic"
    parent_id: Optional[int] = None
    order_index: int = 0
    depth: int = 0
    is_optional: bool = False
    pos_x: float = 0.0
    pos_y: float = 0.0
    children: List["Topic"] = field(default_factory=list)
    resources: List["Resource"] = field(default_factory=list)

    def to_dict(self, recurse: bool = True) -> dict:
        d = {
            "id": self.id, "roadmap_id": self.roadmap_id,
            "node_id": self.node_id, "title": self.title,
            "description": self.description, "topic_type": self.topic_type,
            "parent_id": self.parent_id, "order_index": self.order_index,
            "depth": self.depth, "is_optional": self.is_optional,
            "resources": [r.to_dict() for r in self.resources],
        }
        if recurse:
            d["children"] = [c.to_dict() for c in self.children]
        return d


@dataclass
class Resource:
    id: int = 0
    topic_id: int = 0
    title: str = ""
    url: str = ""
    res_type: str = "article"
    order_index: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id, "title": self.title, "url": self.url,
            "res_type": self.res_type, "order_index": self.order_index,
        }


@dataclass
class SyncRecord:
    id: int = 0
    started_at: str = ""
    completed_at: Optional[str] = None
    status: str = "running"
    trigger: str = "scheduled"
    mode: str = "local"
    roadmaps_found: int = 0
    roadmaps_added: int = 0
    roadmaps_updated: int = 0
    topics_added: int = 0
    resources_added: int = 0
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id, "started_at": self.started_at,
            "completed_at": self.completed_at, "status": self.status,
            "trigger": self.trigger, "mode": self.mode,
            "roadmaps_found": self.roadmaps_found,
            "roadmaps_added": self.roadmaps_added,
            "roadmaps_updated": self.roadmaps_updated,
            "topics_added": self.topics_added,
            "resources_added": self.resources_added,
            "error_message": self.error_message,
        }


@dataclass
class User:
    id: int = 0
    firebase_uid: str = ""
    email: str = ""
    name: str = ""
    avatar_url: str = ""
    resume_skills: dict = field(default_factory=dict)
    onboarded: bool = False
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id, "email": self.email, "name": self.name,
            "avatar_url": self.avatar_url, "resume_skills": self.resume_skills,
            "onboarded": self.onboarded,
        }
