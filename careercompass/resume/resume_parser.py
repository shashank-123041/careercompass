"""
resume_parser.py — PATCHED to fix the schema mismatch and quota-exhaustion
failure mode found when checking this against role_classifier.cbm.

WHAT CHANGED AND WHY (see full review in conversation — summarizing here
so the reasoning travels with the code, not just the chat log)

1. SCHEMA MISMATCH (blocking, independent of rate limits):
   The original prompt only extracted a nested {technical: {...}, domain_
   knowledge: {...}} skill structure — no degree_level, degree_field, or
   years_experience at all, and skill casing ("React.js") didn't match the
   flat, lowercase-hyphenated tag vocabulary role_classifier.cbm was
   trained on ("reactjs"). The nested structure is PRESERVED below (in case
   other parts of the app / UI depend on it) — a new "ml_features" block is
   ADDED alongside it with the exact 4-field contract the model needs,
   using the same extraction rules as the offline training pipeline
   (pilot_extract_skills.py's EXTRACTION_PROMPT) for consistency between
   what the model was trained on and what it'll see at inference time.

2. QUOTA EXHAUSTION WAS SILENT AND UNRECOVERABLE (the actual root cause of
   "limit hit after only a few tries"): confirmed empirically, this API key
   has 20 requests/day on the only Gemini model that still works. The old
   code tried all 3 models and then just raised ConnectionError — meaning
   onboarding hard-fails for every user past roughly the 20th resume of the
   day. Now:
     a. Reuses gemini_utils.py's daily budget tracker (the SAME one the
        offline batch pipeline uses) instead of a second, weaker, duplicated
        retry loop — one shared source of truth for "how much quota is left
        today," not two.
     b. When budget is exhausted OR all models fail, FALLS BACK to
        auto_extract.py's dictionary-based extractor (validated at ~88%
        agreement against manual extraction) instead of hard-failing.
        Lower fidelity than Gemini, but free and always available — this
        turns "onboarding is broken for the rest of the day" into
        "onboarding still works, just slightly less precise after quota
        runs out."

3. CACHING: identical resume file (by content hash) is never sent to Gemini
   twice — every duplicate parse was previously wasting quota you can't
   afford to waste at 20/day. Cache is a simple local JSON keyed by SHA256
   of the file bytes; fine for now, worth moving into the DB (db/operations.py)
   if this needs to survive across deployments/restarts more robustly.

4. print() -> logging, for consistency with app.py's existing logging setup.

STILL OPEN, NOT DONE HERE (flagging rather than silently deciding):
  - The dictionary-fallback path needs resume TEXT, but Gemini currently
    does its own OCR/text-extraction from the raw PDF/image bytes — there's
    no local text-extraction step. For now, the fallback only activates
    cleanly for cases where SOME text can be pulled trivially (e.g. a
    text-based PDF via a lightweight extraction). A scanned/image-only
    resume with Gemini unavailable will still fail — closing that gap
    fully needs a local OCR/text-extraction library (e.g. pdfplumber +
    pytesseract), which is a new dependency worth deciding on deliberately
    rather than adding silently.
  - taxonomy/matcher.py wasn't available to check — if it already has a
    canonical skill-normalization function, ml_features.skill_tags below
    should probably route through THAT instead of / in addition to
    skill_dictionary.txt, to avoid two different "canonical vocabulary"
    sources in the same app. Worth checking before this goes further.
"""
import base64
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

import requests

from careercompass.config import Config

log = logging.getLogger(__name__)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
GEMINI_API_KEY = Config.GEMINI_API_KEY
GEMINI_MODELS_TO_TRY = Config.GEMINI_FALLBACK_MODELS or [Config.GEMINI_MODEL]

CACHE_FILE = Path(Config.RESUME_PARSE_CACHE_PATH)  # data/cache/ (gitignored — contains user resume data)

# Reuses the SAME daily-budget tracker the offline training pipeline uses —
# one shared quota picture instead of two independent, inconsistent ones.
try:
    from careercompass.resume import gemini_utils as gu
    _HAS_BUDGET_TRACKER = True
except ImportError:
    _HAS_BUDGET_TRACKER = False
    log.warning("gemini_utils not found — falling back to no budget tracking "
                "(quota exhaustion will surface as an API error instead of "
                "being predicted in advance)")

# The free, API-independent fallback extractor built and validated earlier —
# used when Gemini is unavailable rather than hard-failing onboarding.
try:
    from careercompass.resume import auto_extract as ae
    _SKILL_DICTIONARY = ae.load_dictionary(str(Config.SKILL_DICTIONARY_PATH))
    _HAS_FALLBACK_EXTRACTOR = True
except (ImportError, FileNotFoundError, AttributeError):
    _HAS_FALLBACK_EXTRACTOR = False
    log.warning("auto_extract / skill_dictionary not found — no fallback "
                "extraction available if Gemini fails")

# Canonical roadmap.sh-anchored vocabulary (~65 terms) — DIFFERENT from and
# complementary to skill_dictionary.txt (568 granular terms), not a
# replacement for it. skill_tags (rich/granular) feeds role_classifier.cbm's
# text features; canonical_skill_slugs (below) feeds anything that needs to
# intersect with opportunity tags or roadmap skill-gap tracking, since BOTH
# of those are already normalized to this same ~65-term space via
# taxonomy/matcher.py. Mixing these up would silently break skill-overlap
# matching — a granular tag like "sap-hana" will never match an opportunity
# tag, since roadmap.sh has no roadmap for it.
try:
    from careercompass.taxonomy.matcher import match_skills as _canonicalize_skills
    _HAS_CANONICAL_MATCHER = True
except ImportError:
    _HAS_CANONICAL_MATCHER = False
    log.warning("taxonomy.matcher not found — canonical_skill_slugs will be "
                "empty, opportunity-matching against this resume won't work")


def _gemini_url(model: str) -> str:
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# ──────────────────────────────────────────────────────────────────────────────


PROMPT = """You are a resume parser. I have provided my resume as a file.

Your job is to extract TWO things:

PART 1 — Skills (same structure as before):
- Technical skills: programming languages, frameworks, libraries, databases, dev tools/platforms.
- Domain knowledge: broad industry knowledge areas only (e.g. Backend Development, Machine Learning,
  Database Management, Cloud Computing, Cybersecurity). High-level domains, not sub-topics.
- Infer parent-child relationships for technical skills (framework/library -> parent language),
  nesting subskills under their parent even if the resume lists them separately.
- Ignore soft skills, personality traits, certifications, and standalone concepts (CRUD, REST API
  Design, OOP) unless explicitly tied to a parent skill in the resume.
- Deduplicate: a skill in both categories stays only in domain_knowledge.

PART 2 — ML-ready features (NEW — this feeds a trained classifier, follow this exactly):
- skill_tags: EVERY skill and subskill from Part 1, flattened into ONE list, each tag normalized
  to lowercase, hyphenated where multi-word (e.g. "Machine Learning" -> "machine-learning",
  "Node.js" -> "nodejs", "React.js" -> "react").
- degree_level: one of "bachelors" | "masters" | "phd" | "unknown" — use "unknown" if not clearly stated.
- degree_field: short field name (e.g. "computer-science") or "unknown" if not clearly stated.
- years_experience: integer (total professional experience, estimate from date ranges if not
  explicitly stated) or "unknown" if genuinely not inferable.
- NEVER leave a field blank or null in Part 2 — use "unknown" instead.

Return ONLY this exact JSON structure, no markdown fences, no explanation outside the JSON:
{
  "skills": {
    "technical": {
      "Python": ["Flask", "OpenCV", "MediaPipe"],
      "Java": null,
      "MySQL": null,
      "Git": null
    },
    "domain_knowledge": {
      "Backend Development": null,
      "Machine Learning": null
    }
  },
  "all_skills_flat": ["Python", "Flask", "OpenCV", "MediaPipe", "Java", "MySQL", "Git", "Backend Development", "Machine Learning"],
  "ml_features": {
    "skill_tags": ["python", "flask", "opencv", "mediapipe", "java", "mysql", "git", "backend-development", "machine-learning"],
    "degree_level": "bachelors",
    "degree_field": "computer-science",
    "years_experience": 2
  }
}

Rules:
- Only extract skills/info actually present in the resume. Do not invent or assume beyond
  reasonable inference (e.g. computing years_experience from explicit date ranges is fine;
  guessing a degree field never mentioned is not).
- Do not include any text outside the JSON object."""


def encode_file(file_path: str) -> tuple[str, str]:
    """Encode file to base64 and determine MIME type."""
    extension = Path(file_path).suffix.lower()

    mime_map = {
        ".pdf":  "application/pdf",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
    }

    if extension not in mime_map:
        raise ValueError(f"Unsupported file type: {extension}. Supported: PDF, JPG, PNG")

    with open(file_path, "rb") as f:
        raw_bytes = f.read()

    return base64.b64encode(raw_bytes).decode("utf-8"), mime_map[extension], raw_bytes


def _file_hash(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()


def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except json.JSONDecodeError:
            log.warning("resume_parse_cache.json is corrupt — starting fresh")
    return {}


def _save_cache(cache: dict):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache))


def _call_gemini(model: str, payload: dict) -> requests.Response:
    return requests.post(
        _gemini_url(model),
        headers={"Content-Type": "application/json"},
        params={"key": GEMINI_API_KEY},
        json=payload,
        timeout=120,
    )


def _fallback_extract(raw_bytes: bytes, extension: str) -> dict:
    """
    Used when Gemini is unavailable (quota exhausted or all models failing).
    NOTE (see module docstring): this currently only works when text can be
    trivially pulled from the file — a scanned/image-only PDF with no text
    layer will still fail here, since there's no local OCR step yet.
    """
    if not _HAS_FALLBACK_EXTRACTOR:
        raise ConnectionError(
            "Gemini is unavailable and no fallback extractor is configured. "
            "Set up auto_extract.py + skill_dictionary.txt to enable graceful "
            "degradation instead of hard failure."
        )

    text = ""
    if extension == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(__import__("io").BytesIO(raw_bytes))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            raise ConnectionError(
                "Gemini is unavailable and pypdf isn't installed for the "
                "fallback text-extraction path. pip install pypdf, or the "
                "onboarding flow will hard-fail for PDFs until Gemini's "
                "quota resets."
            )

    if not text.strip():
        raise ConnectionError(
            "Gemini is unavailable, and no extractable text was found for "
            "fallback processing (likely a scanned/image resume with no "
            "text layer — the fallback path doesn't do OCR yet). "
            "Ask the user to retry once Gemini's daily quota resets."
        )

    extracted = ae.extract_row(text, _SKILL_DICTIONARY)
    log.info("Used fallback dictionary extractor (Gemini unavailable) — "
             "%d tags found", len(extracted["skill_tags"]))

    return {
        "skills": {"technical": {}, "domain_knowledge": {}},  # not populated by the fallback path
        "all_skills_flat": extracted["skill_tags"],
        "ml_features": extracted,
        "canonical_skill_slugs": _canonicalize_skills(extracted["skill_tags"]) if _HAS_CANONICAL_MATCHER else [],
        "_extraction_method": "fallback_dictionary",
    }


def parse_resume(file_path: str) -> dict:
    """
    Send resume to Gemini and extract skills + ml_features.

    Order of operations, and why:
      1. Check cache by content hash — never re-spend quota on an identical file.
      2. Check the shared daily budget BEFORE attempting any call — if it's
         already known to be exhausted, skip straight to the fallback
         extractor instead of burning time on calls guaranteed to fail.
      3. Try each model in GEMINI_MODELS_TO_TRY, same fallback-on-404/429
         logic as before.
      4. If everything fails, use the fallback extractor rather than raising
         and breaking onboarding outright.
    """
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set.\n"
            "Add it to your .env file as GEMINI_API_KEY=your-key-here, "
            "then restart the server (env vars are only read at process "
            "startup, so a running server won't pick up a .env edit)."
        )

    log.info("Reading file: %s", file_path)
    extension = Path(file_path).suffix.lower()
    encoded_file, mime_type, raw_bytes = encode_file(file_path)

    file_hash = _file_hash(raw_bytes)
    cache = _load_cache()
    if file_hash in cache:
        log.info("Cache hit for %s — skipping Gemini call entirely", file_path)
        return cache[file_hash]

    if _HAS_BUDGET_TRACKER and gu.budget_remaining() <= 0:
        log.warning("Daily Gemini budget already exhausted — using fallback "
                    "extractor directly instead of attempting calls known to fail")
        result = _fallback_extract(raw_bytes, extension)
        cache[file_hash] = result
        _save_cache(cache)
        return result

    payload = {
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": mime_type, "data": encoded_file}},
                {"text": PROMPT},
            ]
        }],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 4096},
    }

    last_error = None
    for model in GEMINI_MODELS_TO_TRY:
        if _HAS_BUDGET_TRACKER and gu.budget_remaining() <= 0:
            log.warning("Budget exhausted mid-loop — stopping model attempts, using fallback")
            break

        log.info("Sending to Gemini (%s)...", model)
        try:
            response = _call_gemini(model, payload)
        except requests.exceptions.RequestException as e:
            log.warning("%s: connection error (%s) — trying next model", model, e)
            last_error = e
            continue

        if _HAS_BUDGET_TRACKER:
            gu.record_call()

        if response.status_code == 404:
            log.warning("%s: 404 (not available on this key/tier) — trying next model", model)
            last_error = ConnectionError(f"Gemini API 404 for model '{model}'")
            continue
        if response.status_code == 429:
            log.warning("%s: 429 (rate limited) — trying next model", model)
            last_error = ConnectionError(f"Gemini API 429 for model '{model}'")
            continue
        if response.status_code != 200:
            raise ConnectionError(f"Gemini API error {response.status_code} ({model}): {response.text}")

        log.info("%s: success", model)
        try:
            result = _extract_json(response)
            result["_extraction_method"] = "gemini"
            if _HAS_CANONICAL_MATCHER:
                result["canonical_skill_slugs"] = _canonicalize_skills(
                    result.get("ml_features", {}).get("skill_tags", [])
                )
            cache[file_hash] = result
            _save_cache(cache)
            return result
        except (ValueError, json.JSONDecodeError, KeyError, IndexError) as e:
            log.warning("%s: response wasn't valid JSON (%s) — trying next model", model, e)
            last_error = e
            continue

    log.error("All Gemini models failed (%s) — falling back to dictionary extractor. "
              "Last error: %s", ", ".join(GEMINI_MODELS_TO_TRY), last_error)
    result = _fallback_extract(raw_bytes, extension)
    cache[file_hash] = result
    _save_cache(cache)
    return result


def _extract_json(response: requests.Response) -> dict:
    """Extracts the JSON object from Gemini's response text — robust to
    preamble/markdown-fence wrapping by locating the outermost {...} rather
    than assuming the response starts with the fence."""
    raw_text = (
        response.json()
        ["candidates"][0]
        ["content"]
        ["parts"][0]
        ["text"]
        .strip()
    )

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in model response: {raw_text[:200]!r}")

    json_str = raw_text[start:end + 1]
    return json.loads(json_str)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage:   python resume_parser.py <path_to_resume>")
        print("Example: python resume_parser.py resume.pdf")
        print("         python resume_parser.py resume.png")
        sys.exit(1)

    file_path = sys.argv[1]

    if not Path(file_path).exists():
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

    skills = parse_resume(file_path)

    print("\n========== EXTRACTED SKILLS ==========")
    print(json.dumps(skills, indent=2))

    output_file = Path(file_path).stem + "_skills.json"
    with open(output_file, "w") as f:
        json.dump(skills, f, indent=2)
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
