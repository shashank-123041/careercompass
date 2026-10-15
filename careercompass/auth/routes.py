"""
auth/routes.py

The Google sign-in popup itself is handled entirely client-side by the
Firebase JS SDK (see templates/login.html). This backend never talks to
Google directly — it only verifies the ID token Firebase already vetted,
then starts a normal Flask cookie session keyed to our own users table.
"""
from functools import wraps
from flask import Blueprint, request, jsonify, session
from firebase_admin import auth as fb_auth
from careercompass.db import operations as ops
from careercompass.config import Config


auth_bp = Blueprint("auth", __name__, url_prefix="/auth")


@auth_bp.post("/session-login")
def session_login():
    id_token = (request.json or {}).get("idToken", "")
    if not id_token:
        return jsonify({"error": "idToken required"}), 400

    try:
        decoded = fb_auth.verify_id_token(id_token, clock_skew_seconds=60)
    except Exception as e:
        return jsonify({"error": f"Invalid token: {e}"}), 401

    user = ops.upsert_user(
        firebase_uid=decoded["uid"],
        email=decoded.get("email", ""),
        name=decoded.get("name", ""),
        avatar_url=decoded.get("picture", ""),
    )
    session["user_id"] = user.id
    session["onboarded"] = user.onboarded

    return jsonify({
        "user": user.to_dict(),
        "next": "/onboarding" if not user.onboarded else "/dashboard",
    })


@auth_bp.post("/logout")
def logout():
    session.clear()
    return jsonify({"status": "ok"})


@auth_bp.get("/me")
def me():
    if "user_id" not in session:
        return jsonify({"authenticated": False}), 200
    user = ops.get_user_by_id(session["user_id"])
    if not user:
        session.clear()
        return jsonify({"authenticated": False}), 200
    return jsonify({"authenticated": True, "user": user.to_dict()})


def login_required(fn):
    """Decorator for any route/endpoint that requires an active session."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "not authenticated"}), 401
        return fn(*args, **kwargs)
    return wrapper


def admin_required(fn):
    """Only emails listed in ADMIN_EMAILS may pass. In local development
    (not production) with no ADMIN_EMAILS set, any signed-in user may pass."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "not authenticated"}), 401
        user = ops.get_user_by_id(session["user_id"])
        email = (user.email if user else "").lower()
        if Config.ADMIN_EMAILS:
            allowed = email in Config.ADMIN_EMAILS
        else:
            allowed = not Config.IS_PRODUCTION
        if not user or not allowed:
            return jsonify({"error": "forbidden"}), 403
        return fn(*args, **kwargs)
    return wrapper
