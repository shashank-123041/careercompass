"""
user_profile/routes.py

The profile page itself is mostly a read-heavy view. It deliberately reuses
existing endpoints rather than duplicating logic:
  - Skills pool edits    -> POST /onboarding/confirm-skills   (already exists)
  - Resume re-upload     -> POST /onboarding/parse-resume     (already exists)
  - Password change      -> handled entirely client-side via the Firebase JS
                             SDK (reauthenticateWithCredential + updatePassword)
                             -- we never see or store the password either way,
                             so there's no backend endpoint for it at all.
Only genuinely new server-side logic lives here: rendering the page itself,
and updating the free-text display name we store ourselves.
"""
from flask import Blueprint, request, jsonify, session, render_template

from careercompass.auth.routes import login_required
from careercompass.db import operations as ops

profile_bp = Blueprint("profile", __name__, url_prefix="/profile")


@profile_bp.get("/")
@login_required
def profile_page():
    user = ops.get_user_by_id(session["user_id"])
    if not user:
        session.clear()
        return jsonify({"error": "session invalid"}), 401

    roadmaps = ops.get_user_roadmaps(user.id)
    completed = [r for r in roadmaps if r.get("progress_pct", 0) >= 100]
    in_progress = [r for r in roadmaps if r.get("progress_pct", 0) < 100]

    return render_template(
        "profile.html",
        user=user.to_dict(),
        all_roadmaps=roadmaps,
        completed_roadmaps=completed,
        in_progress_roadmaps=in_progress,
    )


@profile_bp.post("/update-name")
@login_required
def update_name():
    name = (request.json or {}).get("name", "").strip()
    if not name:
        return jsonify({"error": "name required"}), 400
    if len(name) > 100:
        return jsonify({"error": "name too long"}), 400
    ops.update_user_name(session["user_id"], name)
    return jsonify({"status": "ok", "name": name})
