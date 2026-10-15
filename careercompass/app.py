"""
app.py — Flask application factory (careercompass/app.py).

Entry point is wsgi.py (`flask --app wsgi run` / `gunicorn wsgi:app`).
Importing THIS module has no side effects: every import that boots something
(Firebase, schedulers, blueprints) happens inside create_app(), so scripts
and tests can import the package without credentials present.

Debug-mode note: Flask's reloader spawns two processes. The parent has
WERKZEUG_RUN_MAIN unset, the child has it "true". Background schedulers
start only in the child (or when there is no reloader at all), otherwise
two sync threads race on the SQLite upsert.

Production note: schedulers run IN-PROCESS on timers. Run gunicorn with a
single worker (threads are fine), or set RUN_SCHEDULERS=false on every
extra worker/replica so only one process scrapes and syncs.
"""
import logging
import os
from datetime import timedelta

from flask import Flask, jsonify, redirect, render_template, session, url_for

_INSECURE_SECRETS = {"", "dev-change-in-production", "change-this-to-a-random-secret"}


def create_app() -> Flask:
    # Imports live here on purpose — see module docstring.
    from careercompass.config import Config

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )

    if Config.IS_PRODUCTION and Config.SECRET_KEY in _INSECURE_SECRETS:
        raise RuntimeError(
            "SECRET_KEY is unset or still a placeholder. Set a long random "
            "SECRET_KEY in the environment before running in production."
        )

    from careercompass.db import init_db
    from careercompass.db import operations as ops
    from careercompass.db import interaction_operations as interaction_ops
    from careercompass.sync.engine import start_scheduler
    from careercompass.sync.opportunity_engine import start_scheduler as start_opportunity_scheduler
    from careercompass.ml.bootstrap_scheduler import start_scheduler as start_bootstrap_scheduler
    from careercompass.api.routes import api
    from careercompass.api.opportunity_routes import opportunities_api
    import careercompass.auth  # noqa: F401  (side-effect import: boots Firebase Admin SDK)
    from careercompass.auth.routes import auth_bp, admin_required
    from careercompass.onboarding.routes import onboarding_bp
    from careercompass.user_profile.routes import profile_bp

    app = Flask(__name__, template_folder="templates")
    app.secret_key = Config.SECRET_KEY
    app.permanent_session_lifetime = timedelta(days=Config.SESSION_LIFETIME_DAYS)
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=Config.IS_PRODUCTION,   # HTTPS-only cookie in production
        MAX_CONTENT_LENGTH=10 * 1024 * 1024,           # cap resume uploads at 10 MB
    )

    if Config.IS_PRODUCTION:
        # Behind a reverse proxy (Render/Railway/Nginx) the app sees http and the
        # proxy's IP; trust one hop of X-Forwarded-* so redirects/cookies are right.
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

    app.register_blueprint(api)
    app.register_blueprint(auth_bp)
    app.register_blueprint(onboarding_bp)
    app.register_blueprint(profile_bp)
    app.register_blueprint(opportunities_api)

    @app.context_processor
    def inject_current_user():
        """Makes `current_user` available in every template so base.html's nav
        can show the profile menu on every page a logged-in visitor sees."""
        if "user_id" in session:
            u = ops.get_user_by_id(session["user_id"])
            if u:
                return {"current_user": u.to_dict()}
        return {"current_user": None}

    # ── Health check (hosting platforms poll this) ─────────────────────────────

    @app.get("/healthz")
    def healthz():
        return jsonify({"status": "ok"})

    # ── Auth pages ─────────────────────────────────────────────────────────────

    @app.get("/login")
    def login():
        if "user_id" in session:
            return redirect(url_for("dashboard") if session.get("onboarded") else "/onboarding")
        return render_template("login.html")

    @app.get("/dashboard")
    def dashboard():
        if "user_id" not in session:
            return redirect("/login")
        user = ops.get_user_by_id(session["user_id"])
        if not user:
            session.clear()
            return redirect("/login")
        if not user.onboarded:
            return redirect("/onboarding")
        return render_template(
            "index.html",
            stats=ops.get_stats(),
            mode="local" if Config.is_local_mode() else "remote",
            user=user.to_dict(),
            user_roadmaps=ops.get_user_roadmaps(user.id),
        )

    # ── Page routes ────────────────────────────────────────────────────────────

    @app.get("/")
    def index():
        """Routes a visitor by session state: signed in -> dashboard/onboarding,
        otherwise -> login."""
        if "user_id" in session:
            user = ops.get_user_by_id(session["user_id"])
            if user:
                return redirect(url_for("dashboard") if user.onboarded else "/onboarding")
            session.clear()  # stale session (user row deleted)
        return redirect("/login")

    @app.get("/roles")
    def roles():
        return render_template(
            "roles.html",
            roadmaps=[r.to_dict() for r in ops.get_all_roadmaps("role")],
        )

    @app.get("/skills")
    def skills():
        return render_template(
            "skills.html",
            roadmaps=[r.to_dict() for r in ops.get_all_roadmaps("skill")],
        )

    @app.get("/opportunities")
    def opportunities():
        return render_template("opportunities.html")

    @app.get("/roadmap/<rtype>/<slug>")
    def view_roadmap(rtype, slug):
        rm = ops.get_roadmap(slug)
        info = rm.to_dict() if rm else {
            "slug": slug,
            "title": slug.replace("-", " ").title(),
            "icon": Config.SLUG_ICONS.get(slug, "📚"),
            "description": "",
            "type": rtype,
        }
        return render_template("roadmap_detail.html", roadmap=info, roadmap_type=rtype)

    @app.get("/admin")
    @admin_required
    def admin():
        from careercompass.sync.engine import get_state
        return render_template(
            "admin.html",
            stats=ops.get_stats(),
            sync=get_state(),
            history=[h.to_dict() for h in ops.get_sync_history(15)],
            mode="local" if Config.is_local_mode() else "remote",
            repo_path=str(Config.local_repo_path()) if Config.is_local_mode() else "",
        )

    @app.errorhandler(404)
    def e404(e): return render_template("404.html"), 404

    @app.errorhandler(500)
    def e500(e): return render_template("500.html"), 500

    # ── Initialise DB (idempotent) ─────────────────────────────────────────────
    init_db()
    interaction_ops.init_interaction_schema()

    # ── Background schedulers ──────────────────────────────────────────────────
    werkzeug_main = os.environ.get("WERKZEUG_RUN_MAIN")
    running_in_debug_parent = app.debug and werkzeug_main is None

    if Config.RUN_SCHEDULERS and not running_in_debug_parent:
        start_scheduler()
        start_opportunity_scheduler()
        start_bootstrap_scheduler()
    else:
        logging.getLogger(__name__).info(
            "Schedulers NOT started (RUN_SCHEDULERS=%s, debug_parent=%s)",
            Config.RUN_SCHEDULERS, running_in_debug_parent,
        )

    return app