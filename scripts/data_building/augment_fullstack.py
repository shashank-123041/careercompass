"""
augment_fullstack.py — grows the 'full-stack' class specifically.

WHY THIS EXISTS: the Tier 3 expansion (tier3_expand.py) tripled 9 role
classes' row counts, which — via auto_class_weights='Balanced' — appears
to have pulled decision-boundary territory away from classes sharing
generic surface-area skills (python, docker, aws, system-design) with
those newly-boosted roles. full-stack (30 rows, F1 dropped to 0.27 in
the retrain after the Tier 3 boost) is now relatively the weakest class
in the dataset, despite not being touched by that intervention at all.
This targets it the same way Tier 3 was targeted, NOT because it's a
Tier 3 role (it isn't — no confidence-threshold special-casing needed),
just because it's now the thinnest class relative to its neighbors.

GROUNDING: paraphrased from current (2026) full-stack developer job-
description/skill-requirement research — MERN/MEAN stack dominance,
React/Vue frontend + Node.js/Python/Java backend, PostgreSQL/MongoDB,
Docker/cloud deployment — not invented, not copied from any single
source verbatim.

BUDGET: one Gemini call for the whole batch (n=40), same shared tracker
as tier3_expand.py. Check gu.budget_remaining() before running — if
today's budget is exhausted (it was down to 1 call after the Tier 3
run), this will simply wait for tomorrow's reset rather than burn a
call guaranteed to 429.
"""
import csv
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from careercompass.config import Config
from careercompass.resume import gemini_utils as gu

OUTPUT_CSV = str(Config.INTERIM_DATA_DIR / "fullstack_expanded_v1.csv")
SAMPLES_TO_GENERATE = 40  # brings full-stack from 30 -> ~70 rows, back in line with mid-size classes
SOURCE_TAG = "fullstack_websearch_v1"
ROLE_SLUG = "full-stack"

# Paraphrased from current full-stack developer job-description research —
# see module docstring.
GROUNDING = (
    "Full-Stack Developer. Works across BOTH frontend and backend of web "
    "applications. Frontend: JavaScript/TypeScript with a modern framework "
    "(React, Vue, or Angular), HTML5, CSS3. Backend: Node.js, Python "
    "(Django/Flask/FastAPI), or Java (Spring Boot) for APIs and server "
    "logic. Database: SQL (PostgreSQL, MySQL) or NoSQL (MongoDB) for data "
    "modeling and access. Deployment/DevOps awareness: Docker, basic CI/CD, "
    "cloud platforms (AWS/Azure/GCP) — not deep DevOps expertise, but "
    "working familiarity with deploying and monitoring what they build. "
    "Git for version control is universal. Increasingly, MERN "
    "(MongoDB/Express/React/Node) and MEAN stacks dominate job postings; "
    "some roles specifically want a Java-based full stack (Spring Boot + "
    "React/Angular) instead. Strong problem-solving and cross-functional "
    "communication (working with designers, PMs, backend/frontend "
    "specialists) are explicitly called out as core expectations, not "
    "just technical breadth."
)

AUGMENTATION_PROMPT = """You are generating realistic, varied synthetic candidate profiles for a machine learning training set — for a role of "Full Stack Developer".

Here is a grounded summary of what this role actually involves, based on current job-description and skill-requirement research (use this as your factual anchor — do not invent responsibilities or tools that contradict it):
{grounding}

Generate {n} DIFFERENT, realistic candidate profiles for this role. Requirements for genuine diversity:
- Vary which STACK COMBINATION each profile represents — some MERN, some MEAN, some Java+React, some Python+Vue, etc. Not everyone should have the identical frontend+backend pairing.
- Vary skill subset size and composition beyond just the core stack (e.g. some with testing tools, some with specific cloud platforms, some with GraphQL vs REST).
- Vary experience_level across entry/mid/senior realistically.
- Vary degree_level and degree_field realistically (most will be computer-science-adjacent, but include some non-traditional backgrounds — bootcamp-style entrants are common and realistic for this specific role).

FORMATTING RULES (strict):
- skill_tags: lowercase, hyphenated, e.g. "react", "node.js", "postgresql", "docker" — no spaces, no mixed case.
- degree_field: ALSO lowercase and hyphenated, e.g. "computer-science", "information-technology" — NOT "Computer Science" or "computer science". This is a hard requirement — every degree_field value must follow this exact format.

Return ONLY a valid JSON array, no markdown fences, no preamble. Each element:
{{
  "skill_tags": ["lowercase", "hyphenated", "skill", "slugs"],
  "degree_level": "bachelors" | "masters" | "phd" | "unknown",
  "degree_field": "lowercase-hyphenated-field-name",
  "years_experience": integer
}}
"""


def parse_json_array(raw: str) -> list:
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array found in response: {raw[:200]}")
    return json.loads(raw[start:end + 1])


def main():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not found in environment. Set it in .env before running.")

    remaining = gu.budget_remaining()
    print(f"Daily Gemini budget remaining: {remaining} calls. This run needs 1.")
    if remaining < 1:
        raise SystemExit("Budget exhausted for today — rerun after the daily reset "
                          "(gemini_utils.py tracks this by calendar date).")

    from google import genai
    client = genai.Client(api_key=api_key)

    prompt = AUGMENTATION_PROMPT.format(grounding=GROUNDING, n=SAMPLES_TO_GENERATE)
    print(f"Generating {SAMPLES_TO_GENERATE} web-grounded samples for '{ROLE_SLUG}'...")

    raw = gu.call_gemini_with_backoff(prompt, client)
    samples = parse_json_array(raw)

    results = []
    for s in samples:
        results.append({
            "source": SOURCE_TAG,
            "original_category": "Full Stack Developer",
            "role_slug": ROLE_SLUG,
            "original_text_preview": "",
            "skill_tags": s.get("skill_tags", []),
            "degree_level": s.get("degree_level", "unknown"),
            "degree_field": s.get("degree_field", "unknown"),
            "years_experience": s.get("years_experience", "unknown"),
            "extraction_error": "",
        })

    print(f"  got {len(results)} samples (budget remaining: {gu.budget_remaining()})")

    fieldnames = ["source", "original_category", "role_slug", "original_text_preview",
                  "skill_tags", "degree_level", "degree_field", "years_experience",
                  "extraction_error"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone. Wrote {len(results)} rows to {OUTPUT_CSV}")

    import pandas as pd
    df = pd.DataFrame(results)
    print(f"\nUnique skill_tags combos: {df['skill_tags'].astype(str).nunique()}/{len(df)}")
    print(f"degree_field values seen: {sorted(df['degree_field'].unique())}")
    print("(check these are all already lowercase-hyphenated — if not, the prompt's "
          "formatting rule didn't fully hold and normalize_degree_field.py should still be run)")


if __name__ == "__main__":
    main()