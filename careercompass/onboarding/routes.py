"""
onboarding/routes.py

Signup flow after first Firebase login:
  1. Upload resume (PDF/JPG/PNG) -> resume_parser.py (Gemini) extracts skills
  2. User reviews/edits the extracted skill chips (never trust LLM output blindly)
  3. Suggest role roadmaps based on skill overlap (naive word-overlap heuristic
     for now -- will be replaced by the shared skill-taxonomy matcher once the
     opportunity-finder integration needs the same logic)
  4. User picks one primary role roadmap -> onboarding complete
"""
import os
import tempfile
import json as jsonlib
import logging

from flask import Blueprint, request, jsonify, session, render_template
from werkzeug.utils import secure_filename

"""
from auth.routes import login_required
from db import get_conn
from db import operations as ops
import ml.role_inference as role_clf  # SVM, production
"""
from careercompass.auth.routes import login_required
from careercompass.db import get_conn
from careercompass.db import operations as ops
import careercompass.ml.role_inference as role_clf  # SVM, production

log = logging.getLogger(__name__)

onboarding_bp = Blueprint("onboarding", __name__, url_prefix="/onboarding")

ALLOWED_RESUME_EXT = {".pdf", ".jpg", ".jpeg", ".png"}


@onboarding_bp.get("/")
@login_required
def onboarding_page():
    return render_template("onboarding.html")


@onboarding_bp.post("/parse-resume")
@login_required
def parse_resume_endpoint():
    """Accepts an uploaded resume file, runs the Gemini-based parser, and
    returns the extracted (not-yet-saved-as-final) skill JSON for review."""
    if "resume" not in request.files:
        return jsonify({"error": "no file uploaded"}), 400

    f = request.files["resume"]
    ext = os.path.splitext(secure_filename(f.filename or ""))[1].lower()
    if ext not in ALLOWED_RESUME_EXT:
        return jsonify({
            "error": f"Unsupported file type '{ext}'. Upload a PDF, JPG, or PNG. "
                     f"(DOCX is not supported yet — convert to PDF first.)"
        }), 400

    fd, tmp_path = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    try:
        f.save(tmp_path)
    except Exception as e:
        os.remove(tmp_path)
        log.exception("Could not save uploaded resume to disk")
        return jsonify({"error": f"Could not save the uploaded file: {e}"}), 500

    try:
        # resume_parser.py lives at the project root and is imported as a module.
        from careercompass.resume.resume_parser import parse_resume
        result = parse_resume(tmp_path)
    except EnvironmentError as e:
        # Raised by resume_parser.py itself when GEMINI_API_KEY is unset —
        # by far the most common setup mistake, so it gets its own message.
        log.error("GEMINI_API_KEY not configured: %s", e)
        return jsonify({
            "error": "Resume parsing isn't configured yet — GEMINI_API_KEY is "
                     "missing from .env. Add it, restart the server, and try again."
        }), 500
    except ConnectionError as e:
        # Raised by resume_parser.py when the Gemini API itself returns a
        # non-200 response (bad key, rate limit, quota, etc.)
        log.error("Gemini API error: %s", e)
        return jsonify({"error": f"Gemini API rejected the request: {e}"}), 502
    except ValueError as e:
        # json.loads failing on Gemini's response — usually means the model
        # returned something that wasn't clean JSON (rare with temperature=0
        # but not impossible).
        log.error("Could not parse Gemini's response as JSON: %s", e)
        return jsonify({
            "error": "The resume parser got an unexpected response — try again, "
                     "or upload a clearer scan/PDF."
        }), 502
    except Exception as e:
        log.exception("Unexpected error while parsing resume")
        return jsonify({"error": f"Could not parse resume: {e}"}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Save immediately so a refresh doesn't lose the extraction, but this is
    # still considered provisional until the user confirms via /confirm-skills.
    ops.save_resume_skills(session["user_id"], result)
    return jsonify({"skills": result})


@onboarding_bp.post("/confirm-skills")
@login_required
def confirm_skills():
    """User has edited the extracted skill chips in the UI; persist final version."""
    edited = (request.json or {}).get("skills", {})
    if not edited:
        return jsonify({"error": "skills required"}), 400
    ops.save_resume_skills(session["user_id"], edited)
    return jsonify({"status": "ok"})


@onboarding_bp.get("/suggested-roles")
@login_required
def suggested_roles():
    """
    Suggests role roadmaps for the logged-in user's onboarding step.

    Returns a FULL ranked list with a match percentage per role (see
    role_clf.predict_role_rankings), not just a single top-1 highlighted
    pick — this is what the onboarding UI's "Recommended For You" ranked
    list renders. Roles the classifier has no training data for at all
    (e.g. devops-engineer, server-side-game-developer) get match_score:
    null and match_rank: null rather than a fabricated 0% — "no data" and
    "confirmed low fit" are different things and are not conflated here.

    match_tier is computed from the SAME per-role confidence thresholds
    the model's own top-1 confidence gating uses (see
    role_clf.get_confidence_threshold) — not a separately invented UI
    cutoff — so a role crossing "Strong Match" in the UI means the same
    thing it means everywhere else in this system.

    FALLS BACK entirely to the original word-overlap heuristic for
    match_hint (unchanged behavior) whenever the model is unavailable or
    not confident for this profile — same fallback contract as before.
    match_score/match_rank/match_tier are independent of that fallback:
    they come from predict_role_rankings() directly and are simply absent
    (null) for every role when ranking_available is false.
    """
    row = get_conn().execute(
        "SELECT resume_skills FROM users WHERE id=?", (session["user_id"],)
    ).fetchone()
    skills = jsonlib.loads(row["resume_skills"] or "{}") if row else {}
    flat = set(s.lower() for s in skills.get("all_skills_flat", []))
    ml_features = skills.get("ml_features", {}) or {}

    model_prediction = role_clf.predict_role_for_user(ml_features)
    rankings = role_clf.predict_role_rankings(ml_features)
    score_by_slug = {r["role_slug"]: r["score"] for r in rankings}

    use_model = model_prediction.get("available") and model_prediction.get("is_confident")
    model_pick = model_prediction.get("resolved_role") if use_model else None

    roles = ops.get_all_roadmaps("role")
    scored = []
    for r in roles:
        d = r.to_dict()
        score = score_by_slug.get(r.slug)

        if score is not None:
            threshold = role_clf.get_confidence_threshold(r.slug)
            d["match_score"] = round(score, 4)
            d["match_percent"] = round(score * 100)
            if score >= threshold:
                d["match_tier"] = "strong"
            elif score >= threshold * 0.5:
                d["match_tier"] = "possible"
            else:
                d["match_tier"] = "low"
        else:
            d["match_score"] = None
            d["match_percent"] = None
            d["match_tier"] = None

        if use_model:
            d["match_hint"] = (r.slug == model_pick)
        else:
            title_words = set(r.title.lower().replace("-", " ").split())
            d["match_hint"] = bool(flat & title_words)

        scored.append(d)

    # Scored roles first (highest match_score first), unscored roles last
    # (alphabetical among themselves) — unscored roles are still fully
    # browsable/selectable, just not ranked.
    scored.sort(key=lambda x: (
        x["match_score"] is None,
        -(x["match_score"] or 0.0),
        x["title"],
    ))

    rank = 0
    for d in scored:
        if d["match_score"] is not None:
            rank += 1
            d["match_rank"] = rank
            if rank == 1 and d["match_tier"] in ("strong", "possible"):
                d["match_tier"] = "best"
        else:
            d["match_rank"] = None

    return jsonify({
        "roles": scored,
        "detected_skills": sorted(flat),
        "model_prediction": model_prediction,
        "ranking_available": bool(rankings),
    })


@onboarding_bp.post("/select-role")
@login_required
def select_role():
    roadmap_id = (request.json or {}).get("roadmap_id")
    if not roadmap_id:
        return jsonify({"error": "roadmap_id required"}), 400

    ops.add_user_roadmap(session["user_id"], roadmap_id, is_primary=True)
    ops.complete_onboarding(session["user_id"])
    session["onboarded"] = True
    return jsonify({"status": "ok"})