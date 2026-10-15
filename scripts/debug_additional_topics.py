"""
debug_additional_topics_v3.py — run from your project root, same as before.

Usage:
    python -m scripts.debug_additional_topics backend

1. Dumps each section node's OWN full `data` dict -- checking whether the
   section rectangle itself carries its title directly (the same way Pass 1
   edge-connected sections do), which the current spatial-cluster code
   never checks before hunting for a separate caption node.
2. Dumps every 'vertical' / 'horizontal' / 'legend' typed node -- the only
   node types in this file not yet inspected.
"""
import json
import sys
from pathlib import Path

from careercompass.config import Config

slug = sys.argv[1] if len(sys.argv) > 1 else "backend"
json_path = Config.roadmaps_dir() / slug / f"{slug}.json"

raw = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
nodes = raw.get("nodes") or []

def rect_of(node):
    pos = node.get("position") or {}
    try:
        x, y = float(pos.get("x", 0)), float(pos.get("y", 0))
    except (TypeError, ValueError):
        return None
    w = float(node.get("width") or (node.get("style") or {}).get("width") or 0)
    h = float(node.get("height") or (node.get("style") or {}).get("height") or 0)
    return (x, y, w, h)

print(f"=== Full 'data' dict for each SECTION node in {slug}.json ===\n")
sections = [n for n in nodes if str(n.get("type", "")).lower() == "section"]
for s in sections:
    print(f"id={s.get('id')!r}")
    print(f"  full node keys: {list(s.keys())}")
    print(f"  data: {json.dumps(s.get('data'), indent=4, default=str)}")
    print()

print(f"\n=== Every 'vertical' / 'horizontal' / 'legend' node ===\n")
for n in nodes:
    t = str(n.get("type", "")).lower()
    if t in ("vertical", "horizontal", "legend"):
        r = rect_of(n)
        data = n.get("data")
        print(f"type={t}  id={n.get('id')!r}  pos={r}")
        print(f"  data: {json.dumps(data, indent=4, default=str)}")
        print()