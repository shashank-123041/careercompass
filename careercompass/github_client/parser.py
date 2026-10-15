"""
github_client/parser.py
=======================
Parses all three file types from the developer-roadmap repo.

REWRITE NOTES (see chat history for full diagnosis):

1. Resource links in content/*.md are plain Markdown, NOT YAML frontmatter:
       [@official@aiohttp docs](https://docs.aiohttp.org/en/stable/)
   The `@type@Title` convention is embedded in the link text itself.

2. The react-flow JSON encodes hierarchy through TWO independent,
   coexisting mechanisms, not one:
     a) Edges (source -> target) between 'topic'/'subtopic'/'section' nodes.
        Many edges in the file are stale leftovers referencing node ids
        that no longer exist -- these are dropped automatically because
        both endpoints must be present in node_map.
     b) Spatial containment: a 'section' rectangle + a 'label' node
        (the cluster's caption) + one or more 'topic'/'subtopic'/
        'checklist' nodes positioned inside that rectangle's bounds,
        with ZERO edges connecting any of them. This is how things like
        "Common Hooks" or "Environments" sub-groups are represented.

   A node is real and kept if EITHER it touches a live edge OR it falls
   inside a labeled section's bounds. Everything else (decorative chrome,
   or genuinely abandoned duplicate nodes from editor regenerations) is
   discarded. NOTE: the `oldId` field is NOT a reliable duplicate marker
   for topic/subtopic nodes in the edge-tree (verified against real data --
   there it's just internal edge-redirect bookkeeping from the editor).
   It IS reliable for 'section' rectangles specifically, though: verified
   against real backend.json data that a section's own oldId genuinely
   points at a stale duplicate copy of itself left behind after a re-save
   (see _build_spatial_clusters). Don't assume this generalizes to other
   node types without re-verifying against real data first.
"""

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

log = logging.getLogger(__name__)

# ─── Decorative-label filter ─────────────────────────────────────────────────

_DECORATIVE_RE = re.compile(
    r"(roadmap\.sh"
    r"|view\s+the\s+detailed"
    r"|visit\s+the\s+detailed"
    r"|visit\s+the\s+beginner"
    r"|related\s+roadmaps?"
    r"|have\s+a\s+look"
    r"|find\s+the\s+detailed"
    r"|other\s+similar"
    r"|visit\s+roadmap"
    r"|continue\s+learning"
    r"|if\s+you\s+are\s+a\s+beginner"
    r"|this\s+version\s+is\s+designed"
    r"|backend\s+roadmap"
    r"|devops\s+roadmap"
    r"|ai\s*[&/]\s*data\s+scien"
    r"|check\s+out"
    r"|click\s+here"
    r"|learn\s+more\s+at"
    r"|^\s*https?://"
    r")",
    re.IGNORECASE,
)

# Node types that are pure decorative chrome -- never structural, never content.
# NOTE: "button" was removed from this set (see _build_spatial_clusters) --
# verified against real backend.json data that roadmap.sh uses type=="button"
# nodes as genuine section captions (e.g. "API Security Best Practices"),
# not just decorative nav links. Truly decorative buttons (e.g. "roadmap.sh")
# still get caught by is_decorative_label() below via _DECORATIVE_RE; the
# rest (e.g. "Visit Beginner Friendly Version", "DevOps", "Docker" -- nav
# links to other roadmaps) simply never fall inside any section's bounds
# and never qualify as edge-tree roots either, so they're harmlessly ignored
# either way.
_SKIP_TYPES = {
    "link", "text", "annotation", "external",
    "frame", "group", "comment", "sticky",
    "title", "paragraph", "linksgroup", "vertical", "horizontal",
}

# Node types that participate in the edge-tree / spatial-cluster hierarchy.
_HIERARCHY_TYPES = {"topic", "subtopic", "section"}
# Node types that can be spatial-cluster *content* (children) but never edge roots.
_CLUSTER_CONTENT_TYPES = {"topic", "subtopic", "checklist"}


def is_decorative_label(label: str) -> bool:
    label = label.strip()
    if not label:
        return True
    if len(label) > 160:
        return True
    if _DECORATIVE_RE.search(label):
        return True
    return False


def is_decorative_node(node: dict) -> bool:
    ntype = str(node.get("type") or "").lower().strip()
    if ntype in _SKIP_TYPES:
        return True
    data = node.get("data") or {}
    if data.get("href") or data.get("url") or data.get("link"):
        return True
    if ntype in ("label", "checklist", "section"):
        # these are structural helpers, never filtered by label text
        return False
    label = _node_label(node)
    return is_decorative_label(label)


# ─── Public API ───────────────────────────────────────────────────────────────

def parse_roadmap_md(text: str) -> dict:
    """Parse YAML frontmatter from {slug}/{slug}.md -> roadmap metadata dict."""
    try:
        fm, body = _split_fm(text)
        if not fm:
            return {"description": text.strip()[:400]}
        data = yaml.safe_load(fm) or {}
        return {
            "title":       data.get("title") or data.get("briefTitle", ""),
            "brief_title": data.get("briefTitle", ""),
            "description": data.get("description", ""),
            "brief_desc":  data.get("briefDescription", ""),
            "renderer":    data.get("renderer", "graph"),
            "is_new":      bool(data.get("isNew", False)),
            "tags":        _coerce_list(data.get("tags", [])),
        }
    except Exception as e:
        log.debug("parse_roadmap_md error: %s", e)
        return {}


# Fallback only for roadmaps with no .md file / no tags at all (brand new,
# not yet documented upstream) -- e.g. "forward-deployed-engineer" as
# observed in the live repo. Matches a job-title-shaped slug suffix.
_ROLE_SLUG_SUFFIX_RE = re.compile(
    r"(engineer|developer|manager|analyst|architect|scientist|designer"
    r"|writer|advocate|admin|administrator)s?$"
)

# Rare exceptions: a .md file exists but its frontmatter is missing the
# role-roadmap/skill-roadmap tag entirely (verified against live repo data:
# 'devsecops' has tags=[] despite being a real role roadmap, and doesn't
# match the slug-suffix heuristic either). Add slugs here only if both the
# tag check and the suffix heuristic fail for a roadmap you've confirmed
# really is a role on roadmap.sh.
_KNOWN_ROLE_OVERRIDES = {"devsecops"}


def classify_roadmap_type(slug: str, tags: list) -> str:
    """
    Determine 'role' vs 'skill' for a roadmap.

    Primary signal: the roadmap.sh tags convention itself --
    'role-roadmap' / 'skill-roadmap' -- read directly from each
    {slug}.md's frontmatter. This replaces a hardcoded slug allowlist,
    which goes stale every time roadmap.sh adds a new role (verified:
    'bi-analyst' is tagged 'role-roadmap' upstream but was missing from
    the old hardcoded list, silently misclassifying it as a skill).

    Fallback 1 (no .md / no tags at all, e.g. a brand new undocumented
    roadmap): a job-title-shaped slug suffix heuristic.
    Fallback 2: a tiny manual override list for the rare case where a
    .md file exists but its tags omit the role/skill marker entirely.
    """
    tags = [str(t).lower() for t in (tags or [])]
    if "role-roadmap" in tags:
        return "role"
    if "skill-roadmap" in tags:
        return "skill"
    if slug.lower() in _KNOWN_ROLE_OVERRIDES:
        return "role"
    return "role" if _ROLE_SLUG_SUFFIX_RE.search(slug.lower()) else "skill"


# Matches roadmap.sh's real content-file link convention:
#   [@official@aiohttp docs](https://docs.aiohttp.org/en/stable/)
_LINK_RE = re.compile(r"\[@(\w+)@([^\]]+)\]\((https?://[^\s)]+)\)")


def parse_content_file(text: str) -> Tuple[List[dict], str]:
    """
    Parse one topic content file -> (resources, description).

    Real format is plain Markdown:
        # Title
        Some description text...
        Visit the following resources to learn more:
        - [@official@Docs](https://...)
        - [@article@Some article](https://...)
    No YAML frontmatter is involved -- the old frontmatter-based parser
    silently returned zero resources for every single file.
    """
    try:
        resources = []
        for rtype, title, url in _LINK_RE.findall(text):
            title = title.strip()
            url = url.strip()
            if not title or not url:
                continue
            resources.append({
                "title": title,
                "url": url,
                "res_type": _norm_type(rtype),
            })

        # Description = everything before the resource list, with the
        # leading '# Title' heading and the "visit resources" lead-in
        # stripped out.
        body = _LINK_RE.sub("", text)
        body = re.sub(r"^\s*#{1,6}\s+.*$", "", body, count=1, flags=re.MULTILINE)
        body = re.sub(r"visit the following resources.*", "", body,
                       flags=re.IGNORECASE | re.DOTALL)
        desc = _strip_md(body).strip()[:600]
        return resources, desc
    except Exception as e:
        log.debug("parse_content_file: %s", e)
        return [], ""


def node_id_from_filename(filename: str) -> str:
    """'variables-and-types@AbCdEf123.md'  ->  'AbCdEf123'"""
    stem = filename[:-3] if filename.endswith(".md") else filename
    return stem.rsplit("@", 1)[1].strip() if "@" in stem else ""


def title_from_filename(filename: str) -> str:
    """'variables-and-types@AbCdEf123.md'  ->  'Variables And Types'"""
    stem = filename[:-3] if filename.endswith(".md") else filename
    label = stem.split("@")[0] if "@" in stem else stem
    return label.replace("-", " ").replace("_", " ").strip().title()


# ─── Content directory scan ───────────────────────────────────────────────────

def scan_content_dir(content_dir: Path) -> Dict[str, dict]:
    """
    Read every *.md in content/ and return:
        { node_id: {node_id, title, description, resources} }
    """
    result: Dict[str, dict] = {}
    if not content_dir.is_dir():
        return result

    for md_file in sorted(content_dir.glob("*.md")):
        if md_file.stem.lower() in {"index", "mapping", "readme"}:
            continue
        node_id = node_id_from_filename(md_file.name)
        title = title_from_filename(md_file.name)
        if not node_id or not title:
            continue
        try:
            text = md_file.read_text(encoding="utf-8", errors="replace")
            resources, description = parse_content_file(text)
        except Exception as e:
            log.warning("content read error %s: %s", md_file.name, e)
            resources, description = [], ""

        result[node_id] = {
            "node_id": node_id,
            "title": title,
            "description": description,
            "resources": resources,
        }
    log.debug("content_dir %s: %d topics", content_dir.parent.name, len(result))
    return result


# ─── JSON -> topic tree ────────────────────────────────────────────────────────

def build_tree_from_json(
    json_text: str,
    content_map: Dict[str, dict],
) -> List[dict]:
    """
    Parse react-flow JSON and reconstruct the topic hierarchy using both
    mechanisms roadmap.sh actually uses (edges + spatial containment),
    attach content-map data, and return a nested list ready for
    ops.replace_topics(). Falls back to a flat list from content_map if
    the JSON is unusable.
    """
    nodes: List[dict] = []
    edges: List[dict] = []

    if json_text:
        try:
            raw = json.loads(json_text)
            nodes = raw.get("nodes") or []
            edges = raw.get("edges") or []
        except Exception as e:
            log.warning("JSON parse error: %s", e)

    if not nodes:
        log.debug("no nodes in JSON -- flat content list")
        return _flat_from_content(content_map)

    # ── Filter: drop pure decorative chrome ────────────────────────────────
    valid_nodes = [n for n in nodes if not is_decorative_node(n)]
    node_map: Dict[str, dict] = {
        str(n.get("id", "")).strip(): n
        for n in valid_nodes
        if str(n.get("id", "")).strip()
    }

    # ── Pass 1: edge-tree among topic/subtopic/section nodes only ─────────
    children_of: Dict[str, List[str]] = defaultdict(list)
    in_degree: Dict[str, int] = defaultdict(int)
    hierarchy_ids = {
        nid for nid, n in node_map.items()
        if str(n.get("type", "")).lower() in _HIERARCHY_TYPES
    }

    for e in edges:
        src = str(e.get("source") or e.get("from") or "").strip()
        tgt = str(e.get("target") or e.get("to") or "").strip()
        if (src and tgt and src in hierarchy_ids and tgt in hierarchy_ids
                and src != tgt):
            children_of[src].append(tgt)
            in_degree[tgt] += 1

    # Only 'topic'/'section' nodes may become top-level roots -- 'subtopic'
    # nodes with zero edges are spatial-cluster content (Pass 2's job), not
    # standalone roots. Without this, a label-less floating subtopic like
    # "aiohttp" or "Fast API" would wrongly become its own top-level root
    # and get marked used before Pass 2 ever sees it.
    _root_types = {"topic", "section"}
    root_candidates = [
        nid for nid in hierarchy_ids
        if in_degree.get(nid, 0) == 0
        and str(node_map[nid].get("type", "")).lower() in _root_types
        and _node_label(node_map[nid])
    ]

    # Dedup by cleaned label: editor regenerations sometimes leave a
    # zero-edge duplicate of a node alongside the live, edge-connected
    # version (same label, different id). Keep whichever copy has more
    # total edge connections; drop the rest. (The `oldId` field is NOT a
    # reliable duplicate marker -- verified against real repo data.)
    _degree: Dict[str, int] = defaultdict(int)
    for e in edges:
        s = str(e.get("source") or e.get("from") or "").strip()
        t = str(e.get("target") or e.get("to") or "").strip()
        if s in node_map:
            _degree[s] += 1
        if t in node_map:
            _degree[t] += 1

    by_label: Dict[str, List[str]] = defaultdict(list)
    for nid in root_candidates:
        key = _clean(_node_label(node_map[nid])).lower()
        by_label[key].append(nid)

    edge_roots = []
    for key, ids in by_label.items():
        if len(ids) == 1:
            edge_roots.append(ids[0])
        else:
            ids.sort(key=lambda i: _degree.get(i, 0), reverse=True)
            edge_roots.append(ids[0])

    def _pos(nid: str) -> Tuple[float, float]:
        pos = node_map.get(nid, {}).get("position") or {}
        return (float(pos.get("y", 0)), float(pos.get("x", 0)))

    edge_roots.sort(key=_pos)
    for nid in children_of:
        children_of[nid].sort(key=_pos)

    used_ids: set = set()  # tracks every node consumed by either pass

    def _build_edge_node(nid: str, depth: int) -> Optional[dict]:
        if nid in used_ids:
            return None
        used_ids.add(nid)

        raw_node = node_map[nid]
        label = _clean(_node_label(raw_node))
        if not label or is_decorative_label(label):
            return None

        ntype = str(raw_node.get("type") or "").lower()
        pos = raw_node.get("position") or {}

        content = content_map.get(nid, {})
        children = []
        for child_id in children_of.get(nid, []):
            if child_id in node_map:
                child = _build_edge_node(child_id, depth + 1)
                if child is not None:
                    children.append(child)

        # A depth-0 root only acts as a section HEADER (excluded from
        # progress-percentage math -- see db/operations.get_roadmap_progress_pct)
        # if it actually groups children. A depth-0 root with NO children is a
        # genuine standalone leaf topic -- the frontend already renders it as
        # a checkable standalone button (wp-standalone), not a header, so it
        # must be typed the same way here. Verified: without this check, a
        # standalone root got marked "section" purely for being at depth 0,
        # silently dropping it from both sides of the progress ratio -- the
        # roadmap's own detail page correctly showed 100% (frontend counts
        # every rendered checkbox regardless of stored topic_type) while the
        # dashboard card and profile's "completed roadmaps" check both used
        # the backend percentage, which could never reach 100 as a result.
        if ntype == "section" or (depth == 0 and children):
            topic_type = "section"
        elif ntype == "subtopic":
            topic_type = "subtopic"
        else:
            topic_type = "topic"

        return {
            "node_id": nid,
            "title": label,
            "description": content.get("description", ""),
            "topic_type": topic_type,
            "pos_x": float(pos.get("x", 0)),
            "pos_y": float(pos.get("y", 0)),
            "is_optional": False,
            "resources": content.get("resources", []),
            "children": children,
        }

    tree = []
    for root_id in edge_roots:
        node = _build_edge_node(root_id, 0)
        if node is not None:
            tree.append(node)

    # ── Pass 2: spatial clusters for everything edges never touched ───────
    cluster_nodes = _build_spatial_clusters(node_map, used_ids, content_map)
    tree.extend(cluster_nodes)

    # ── Prune empty sections, attach leftover orphan content topics ───────
    tree = _prune_empty_sections(tree, content_map)

    covered = _collect_node_ids(tree)
    orphans = [
        v for k, v in content_map.items()
        if k not in covered and v.get("title")
    ]

    if orphans:
        # Dedup by title: content_map is keyed by node_id, but roadmap.sh's
        # content library can have multiple files (different ids, e.g. from
        # older reorganizations) that render to the identical title -- keep
        # whichever copy has more resources attached, drop the rest.
        by_title: Dict[str, dict] = {}
        for o in orphans:
            key = o["title"].strip().lower()
            existing = by_title.get(key)
            if not existing or len(o.get("resources", [])) > len(existing.get("resources", [])):
                by_title[key] = o
        orphans = sorted(by_title.values(), key=lambda x: x["title"].lower())

    if orphans:
        if tree:
            # These are real content-library topics that never got placed
            # in the visual diagram (verified against real data -- roughly
            # 80 for backend.json, spanning things like AI-tooling bonus
            # content and superseded database entries). Rather than one
            # flat wall of dozens of chips under a repeated generic label
            # (confirmed to look broken/unscannable to a real user), chunk
            # into readable groups with an alphabetical-range title so each
            # card is scannable and individually collapsible.
            CHUNK_SIZE = 14
            chunks = [orphans[i:i + CHUNK_SIZE] for i in range(0, len(orphans), CHUNK_SIZE)]
            for i, chunk in enumerate(chunks):
                first, last = chunk[0]["title"], chunk[-1]["title"]
                label = f"Bonus Topics: {first} – {last}" if first != last else f"Bonus Topics: {first}"
                tree.append({
                    "node_id": f"__extra_{i}__",
                    "title": label,
                    "description": "",
                    "topic_type": "section",
                    "pos_x": 0, "pos_y": 999999 + i,
                    "is_optional": False, "resources": [],
                    "children": [
                        {**o, "topic_type": "topic",
                         "pos_x": 0, "pos_y": 0,
                         "is_optional": False, "children": []}
                        for o in chunk
                    ],
                })
        else:
            return _flat_from_content(content_map)

    if not tree:
        log.debug("no roots after edge+spatial pass -- flat content list")
        return _flat_from_content(content_map)

    log.debug("tree: %d roots, %d total after pruning", len(tree), _count(tree))
    return tree


def _rect_of(node: dict) -> Optional[Tuple[float, float, float, float]]:
    """Returns (x1, y1, x2, y2) bounding box for a node, or None."""
    pos = node.get("position") or {}
    try:
        x = float(pos.get("x", 0))
        y = float(pos.get("y", 0))
    except (TypeError, ValueError):
        return None
    w = float(node.get("width") or (node.get("style") or {}).get("width") or 0)
    h = float(node.get("height") or (node.get("style") or {}).get("height") or 0)
    if w <= 0 or h <= 0:
        return None
    return (x, y, x + w, y + h)


def _point_in_rect(x: float, y: float, rect: Tuple[float, float, float, float],
                    margin: float = 5.0) -> bool:
    x1, y1, x2, y2 = rect
    return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)


def _build_spatial_clusters(
    node_map: Dict[str, dict],
    used_ids: set,
    content_map: Dict[str, dict],
) -> List[dict]:
    """
    For every 'section' rectangle, find topic/subtopic/checklist nodes
    positioned inside its bounds that no edge ever touched, and the
    caption node (if any) inside the same bounds to use as the cluster
    title. Returns a list of synthetic section dicts ready to append
    to the tree.
    """
    all_sections: List[Tuple[str, dict, Tuple[float, float, float, float]]] = [
        (nid, n, rect)
        for nid, n in node_map.items()
        if str(n.get("type", "")).lower() == "section"
        and (rect := _rect_of(n)) is not None
    ]
    if not all_sections:
        return []

    # Dedup duplicate section rectangles left behind by roadmap.sh's editor
    # after a re-save. Verified against real backend.json data: two nearly
    # identical rects (same style width/height, positioned close together),
    # where the CURRENT copy's own data.oldId field points at the id of the
    # STALE copy that's still sitting in the file. Without this, both are
    # processed independently and each generates its own separate fallback
    # section for what should be one.
    stale_ids = {
        n.get("data", {}).get("oldId")
        for _, n, _ in all_sections
        if n.get("data", {}).get("oldId")
    }
    section_entries = [(nid, n, rect) for nid, n, rect in all_sections if nid not in stale_ids]
    section_rects: List[Tuple[str, Tuple[float, float, float, float]]] = [
        (nid, rect) for nid, n, rect in section_entries
    ]
    if not section_rects:
        return []

    # Map each unused content-bearing node to the smallest containing rect.
    def _smallest_containing(x: float, y: float) -> Optional[str]:
        best_id, best_area = None, None
        for sid, rect in section_rects:
            if _point_in_rect(x, y, rect):
                area = (rect[2] - rect[0]) * (rect[3] - rect[1])
                if best_area is None or area < best_area:
                    best_id, best_area = sid, area
        return best_id

    clusters: Dict[str, List[dict]] = defaultdict(list)
    labels_by_section: Dict[str, str] = {}

    for nid, n in node_map.items():
        if nid in used_ids:
            continue
        ntype = str(n.get("type", "")).lower()
        pos = n.get("position") or {}
        try:
            x, y = float(pos.get("x", 0)), float(pos.get("y", 0))
        except (TypeError, ValueError):
            continue

        # "button" included alongside "label" -- verified against real data
        # that roadmap.sh uses type=="button" nodes as genuine section
        # captions (see _SKIP_TYPES note above).
        if ntype in ("label", "button"):
            label_text = _clean(_node_label(n))
            if label_text and not is_decorative_label(label_text):
                sid = _smallest_containing(x, y)
                if sid:
                    labels_by_section[sid] = label_text
            used_ids.add(nid)
            continue

        if ntype in _CLUSTER_CONTENT_TYPES:
            sid = _smallest_containing(x, y)
            if sid:
                clusters[sid].append(n)

    result = []
    for sid, members in clusters.items():
        children = []
        for n in members:
            nid = str(n.get("id", "")).strip()
            ntype = str(n.get("type", "")).lower()
            pos = n.get("position") or {}

            if ntype == "checklist":
                used_ids.add(nid)
                content = content_map.get(nid, {})
                for item in (n.get("data") or {}).get("checklists", []):
                    item_label = _clean(str(item.get("label", "")))
                    if not item_label:
                        continue
                    children.append({
                        "node_id": f"{nid}:{item.get('id', '')}",
                        "title": item_label,
                        "description": "",
                        "topic_type": "subtopic",
                        "pos_x": float(pos.get("x", 0)),
                        "pos_y": float(pos.get("y", 0)),
                        "is_optional": False,
                        "resources": content.get("resources", []),
                        "children": [],
                    })
                continue

            label = _clean(_node_label(n))
            if not label or is_decorative_label(label):
                continue
            used_ids.add(nid)
            content = content_map.get(nid, {})
            children.append({
                "node_id": nid,
                "title": label,
                "description": content.get("description", ""),
                "topic_type": "subtopic" if ntype == "subtopic" else "topic",
                "pos_x": float(pos.get("x", 0)),
                "pos_y": float(pos.get("y", 0)),
                "is_optional": False,
                "resources": content.get("resources", []),
                "children": [],
            })

        if not children:
            continue

        title = labels_by_section.get(sid)
        if not title:
            # Verified against real data: some clusters genuinely have no
            # caption node anywhere in the source JSON -- not a parsing
            # bug, roadmap.sh's own diagram doesn't give them a header
            # either (just a dashed grouping box). Repeating the identical
            # literal "Additional Topics" string across several such
            # clusters reads as broken even though the underlying fact is
            # real, so derive something readable from the cluster's own
            # contents instead.
            names = [c["title"] for c in children[:2]]
            title = ", ".join(names)
            if len(children) > 2:
                title += f" +{len(children) - 2} more"

        sect_pos = node_map.get(sid, {}).get("position") or {}
        result.append({
            "node_id": sid,
            "title": title,
            "description": "",
            "topic_type": "section",
            "pos_x": float(sect_pos.get("x", 0)),
            "pos_y": float(sect_pos.get("y", 0)),
            "is_optional": False,
            "resources": [],
            "children": children,
        })

    result.sort(key=lambda n: (n["pos_y"], n["pos_x"]))
    return result


def _prune_empty_sections(
    nodes: List[dict],
    content_map: Dict[str, dict],
) -> List[dict]:
    """
    By the time a node reaches here it has already passed label validation
    in _build_edge_node / _build_spatial_clusters, so it's a legitimate
    roadmap item regardless of content-file coverage -- including a
    childless depth-0 root (a real standalone topic; the frontend already
    renders these as a wide standalone button rather than a header). This
    pass just recurses to keep nested child lists consistent; it is
    intentionally NOT a content-map-based filter (the original version's
    "drop if not in content_map and no children" rule was wrong -- it
    cascade-deleted entire branches whenever a leaf topic simply had no
    matching content file yet, which is common and not an error).
    """
    for node in nodes:
        node["children"] = _prune_empty_sections(node["children"], content_map)
    return nodes


def _flat_from_content(content_map: Dict[str, dict]) -> List[dict]:
    """
    Fallback when a roadmap has no usable hierarchy (no <slug>.json, e.g. a
    roadmap newer than the pinned structure commit — see
    github_client/structure.py). Rather than one flat wall of dozens of
    separate nodes, group the topics alphabetically into readable chunks,
    the same way leftover "Bonus Topics" are grouped.
    """
    topics = [
        {**info, "topic_type": "topic",
         "pos_x": 0, "pos_y": 0,
         "is_optional": False, "children": []}
        for info in sorted(content_map.values(),
                           key=lambda x: x.get("title", "").lower())
    ]
    CHUNK_SIZE = 14
    if len(topics) <= CHUNK_SIZE:
        return topics

    sections = []
    for i in range(0, len(topics), CHUNK_SIZE):
        chunk = topics[i:i + CHUNK_SIZE]
        first, last = chunk[0]["title"], chunk[-1]["title"]
        label = f"Topics: {first} – {last}" if first != last else f"Topics: {first}"
        sections.append({
            "node_id": f"__flat_{i // CHUNK_SIZE}__",
            "title": label,
            "description": "",
            "topic_type": "section",
            "pos_x": 0, "pos_y": i,
            "is_optional": False, "resources": [],
            "children": chunk,
        })
    return sections


def _collect_node_ids(tree: List[dict]) -> set:
    ids = set()
    for n in tree:
        ids.add(n["node_id"])
        ids |= _collect_node_ids(n.get("children", []))
    return ids


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _split_fm(text: str) -> Tuple[Optional[str], str]:
    m = re.match(r"^---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n?(.*)",
                 text, re.DOTALL)
    return (m.group(1), m.group(2)) if m else (None, text)


def _norm_type(raw: str) -> str:
    return {
        "official": "official", "article": "article", "video": "video",
        "course": "course", "opensource": "opensource", "feed": "feed",
        "podcast": "article", "website": "article", "youtube": "video",
        "book": "article", "blog": "article",
    }.get(str(raw).lower().strip(), "article")


def _node_label(node: dict) -> str:
    data = node.get("data") or {}
    return (
        data.get("label") or data.get("title") or data.get("name") or
        node.get("label") or node.get("title") or node.get("name") or ""
    )


def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", str(text))
    return re.sub(r"\s+", " ", text).strip()


def _strip_md(text: str) -> str:
    text = re.sub(r"#{1,6}\s+", "", text)
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
    text = re.sub(r"`[^`]*`", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _coerce_list(v) -> list:
    if isinstance(v, list): return v
    if isinstance(v, str): return [v] if v else []
    return []


def _count(tree: List[dict]) -> int:
    return sum(1 + _count(n.get("children", [])) for n in tree)