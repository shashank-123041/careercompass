"""
config.py — All application configuration.

IMPORTANT — two data source modes:
  1. LOCAL  (recommended): set ROADMAP_REPO_PATH to your local clone of
     https://github.com/nilbuild/developer-roadmap
     The app reads files directly from disk — fast, no API limits.

  2. REMOTE (fallback): leave ROADMAP_REPO_PATH empty and the app will
     fetch from the GitHub API.  Set GITHUB_TOKEN for higher rate limits
     (5 000 req/hr authenticated vs 60 unauthenticated).
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ── Project layout ───────────────────────────────────────────────────────────
# PACKAGE_DIR  = .../careercompass/careercompass   (application code, templates)
# PROJECT_ROOT = .../careercompass                 (repo root: data/, models/, scripts/ ...)
PACKAGE_DIR  = Path(__file__).parent.absolute()
PROJECT_ROOT = PACKAGE_DIR.parent
BASE_DIR     = PROJECT_ROOT          # kept for backward compatibility with older callers

load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
GENERATED_DIR = DATA_DIR / "generated"
for _d in (DATA_DIR, CACHE_DIR, GENERATED_DIR,
           DATA_DIR / "raw", DATA_DIR / "interim", DATA_DIR / "final",
           PROJECT_ROOT / "models", PROJECT_ROOT / "results"):
    _d.mkdir(parents=True, exist_ok=True)


class Config:
    # ── Paths (single source of truth — no module should hardcode a path) ──────
    PACKAGE_DIR  = PACKAGE_DIR
    PROJECT_ROOT = PROJECT_ROOT
    BASE_DIR     = BASE_DIR
    DATA_DIR     = DATA_DIR
    CACHE_DIR    = CACHE_DIR                       # runtime caches (gitignored)
    SCRAPER_CACHE_DIR = CACHE_DIR / "scrapers"     # 1h scraper file cache
    GENERATED_DIR = GENERATED_DIR                  # regenerated outputs, e.g. pseudo-labels (gitignored)
    RAW_DATA_DIR     = DATA_DIR / "raw"            # original public datasets (gitignored)
    INTERIM_DATA_DIR = DATA_DIR / "interim"        # intermediate data-building outputs (gitignored)
    FINAL_DATA_DIR   = DATA_DIR / "final"          # committed training datasets
    MODELS_DIR  = PROJECT_ROOT / "models"
    RESULTS_DIR = PROJECT_ROOT / "results"

    # Training dataset used by training/, scripts/ and the opportunity bootstrap
    FINAL_DATASET_CSV = Path(os.environ.get(
        "FINAL_DATASET_CSV", str(FINAL_DATA_DIR / "final_dataset_v7.csv")))
    TIER1_RAW_CSV = RAW_DATA_DIR / "UpdatedResumeDataSet.csv"
    TIER2_RAW_CSV = RAW_DATA_DIR / "candidate_job_role_dataset.csv"

    # Resume-parser support files
    SKILL_DICTIONARY_PATH   = PACKAGE_DIR / "resume" / "skill_dictionary.txt"
    RESUME_PARSE_CACHE_PATH = CACHE_DIR / "resume_parse_cache.json"
    GEMINI_BUDGET_STATE_PATH = CACHE_DIR / "gemini_budget_state.json"

    # Opportunity-ranker pseudo-label outputs
    OPP_PSEUDO_LABELS_PATH           = GENERATED_DIR / "opportunity_pseudo_labels.csv"
    OPP_PSEUDO_LABELS_SYNTHETIC_PATH = GENERATED_DIR / "opportunity_pseudo_labels_synthetic.csv"
    OPP_PSEUDO_LABELS_COMBINED_PATH  = GENERATED_DIR / "opportunity_pseudo_labels_combined.csv"

    # ── Flask ──────────────────────────────────────────────────────────────────
    SECRET_KEY  = os.environ.get("SECRET_KEY", "dev-change-in-production")
    DEBUG       = os.environ.get("DEBUG", "true").lower() == "true"
    HOST        = os.environ.get("HOST", "127.0.0.1")
    PORT        = int(os.environ.get("PORT", 5050))
    # How long a login session stays valid without the user having to sign in
    # again — applies only when session.permanent = True is set at login.
    SESSION_LIFETIME_DAYS = int(os.environ.get("SESSION_LIFETIME_DAYS", 30))
    # ── Deployment ─────────────────────────────────────────────────────────────
    IS_PRODUCTION = os.environ.get("APP_ENV", "development").lower() == "production"
    RUN_SCHEDULERS = os.environ.get("RUN_SCHEDULERS", "true").lower() == "true"
    ADMIN_EMAILS = frozenset(
        e.strip().lower() for e in os.environ.get("ADMIN_EMAILS", "").split(",") if e.strip()
    )

    # ── Database ───────────────────────────────────────────────────────────────
    DB_PATH = DATA_DIR / "roadmaps.db"

    # ── Data source ────────────────────────────────────────────────────────────
    # Path to local clone of nilbuild/developer-roadmap (recommended)
    ROADMAP_REPO_PATH: Path = Path(os.environ.get("ROADMAP_REPO_PATH", ""))

    # GitHub remote fallback settings
    GITHUB_OWNER  = os.environ.get("GITHUB_OWNER",  "nilbuild")
    GITHUB_REPO   = os.environ.get("GITHUB_REPO",   "developer-roadmap")
    GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "master")
    GITHUB_TOKEN  = os.environ.get("GITHUB_TOKEN",  "")

    # Roadmap STRUCTURE fallback (see github_client/structure.py): the upstream repo
    # deleted <slug>.json / <slug>.md on 2026-07-28. This commit (the last one that
    # still had them) is used to restore the topic grouping.
    ROADMAP_STRUCTURE_COMMIT = os.environ.get(
        "ROADMAP_STRUCTURE_COMMIT", "6f745dc1eaa9d6a9841fe964be33835263f0fd92")
    ROADMAP_STRUCTURE_FALLBACK = os.environ.get(
        "ROADMAP_STRUCTURE_FALLBACK", "true").lower() == "true"
    STRUCTURE_CACHE_DIR = CACHE_DIR / "roadmap_structure"

    # Sub-paths inside the repo
    ROADMAPS_SUBPATH      = Path("src/data/roadmaps")
    # Remote (GitHub API) mode reads the CURRENT repo layout
    REMOTE_ROADMAPS_SUBPATH = Path(os.environ.get("REMOTE_ROADMAPS_SUBPATH", "roadmaps"))
    BEST_PRACTICES_SUBPATH = Path("src/data/best-practices")

    # ── HTTP ───────────────────────────────────────────────────────────────────
    REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", 20))
    # Polite delay between remote GitHub API calls (seconds)
    API_DELAY       = float(os.environ.get("API_DELAY", 0.3))

    # ── Sync ───────────────────────────────────────────────────────────────────
    SYNC_ON_STARTUP     = os.environ.get("SYNC_ON_STARTUP", "true").lower() == "true"
    SYNC_INTERVAL_HOURS = int(os.environ.get("SYNC_INTERVAL_HOURS", 24))
    STALE_AFTER_HOURS   = int(os.environ.get("STALE_AFTER_HOURS", 48))

    # ── Cache ──────────────────────────────────────────────────────────────────
    CACHE_TTL = int(os.environ.get("CACHE_TTL", 300))
    # ── Add this block to config.py, inside class Config, right after the
    # ── existing "Cache" section (after the CACHE_TTL line) ──────────────────

    # ── Opportunity finder (Step 2) ─────────────────────────────────────────────
    OPPORTUNITY_SYNC_INTERVAL_HOURS = int(os.environ.get("OPP_SYNC_INTERVAL_HOURS", 6))
    OPPORTUNITY_RETENTION_DAYS      = int(os.environ.get("OPP_RETENTION_DAYS", 3))
    OPPORTUNITY_STALE_SCRAPE_DAYS   = int(os.environ.get("OPP_STALE_SCRAPE_DAYS", 14))
    SKILL_MATCH_THRESHOLD = float(os.environ.get("SKILL_MATCH_THRESHOLD", 82))

    # Hard ceiling for /api/v2/opportunities list requests. Was hardcoded to
    # 300 — too low once real scrape volume kicked in.
    OPPORTUNITY_API_MAX_LIMIT = int(os.environ.get("OPP_API_MAX_LIMIT", 2000))

    # ── ML opportunity ranker bootstrap (Phase B) ──────────────────────────────
    # How often ml/bootstrap_scheduler.py regenerates the pseudo-label CSVs.
    # Cheap batch job (CSV write, not live-serving), so daily is reasonable.
    # Safe to run even with zero real users or a missing grounding dataset —
    # see ml/bootstrap_scheduler.py's module docstring.
    BOOTSTRAP_INTERVAL_HOURS = int(os.environ.get("BOOTSTRAP_INTERVAL_HOURS", 24))
 
    # ── Resume parsing (Gemini) ─────────────────────────────────────────────────
    # Routed through Config, not read directly via os.environ in
    # resume_parser.py — that module had no load_dotenv() of its own and
    # relied on being imported AFTER config.py. Reproduced concretely:
    # importing resume_parser first left GEMINI_API_KEY as '' even with a
    # valid .env present.
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
    GEMINI_FALLBACK_MODELS = [
        m.strip() for m in os.environ.get(
            "GEMINI_FALLBACK_MODELS", "gemini-3.5-flash,gemini-2.5-flash,gemini-2.0-flash,gemini-flash-latest"
        ).split(",") if m.strip()
    ]
    
    # ── Firebase (auth) ──────────────────────────────────────────────────────────
    # Path to the service-account JSON downloaded from Firebase Console →
    # Project Settings → Service Accounts → Generate new private key.
    # NEVER commit this file — keep it under secrets/ and gitignore it.
    FIREBASE_SERVICE_ACCOUNT_PATH = os.environ.get(
        "FIREBASE_SERVICE_ACCOUNT_PATH", "./secrets/firebase-service-account.json"
    )

    # ── Gemini (resume parser) ───────────────────────────────────────────────────
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

    # ── Role classifier artifacts (all under models/, see Config.MODELS_DIR) ────
    # SVM = PRODUCTION (training/train_svm.py). CatBoost = rollback only
    # (training/train_catboost.py). Binary artifacts (.joblib/.cbm) are NOT
    # committed to git — retrain, or download them from a GitHub Release.
    ROLE_CLASSIFIER_MODEL_PATH = os.environ.get(
        "ROLE_CLASSIFIER_MODEL_PATH",
        str(BASE_DIR / "role_classifier" / "role_classifier_svm.joblib"),
    )
    ROLE_CLASSIFIER_META_PATH = os.environ.get(
        "ROLE_CLASSIFIER_META_PATH",
        str(BASE_DIR / "role_classifier" / "role_classifier_svm_meta.json"),
    )
    SVM_MODEL_PATH = os.environ.get(
        "SVM_MODEL_PATH", str(MODELS_DIR / "role_classifier_svm.joblib")
    )
    SVM_META_PATH = os.environ.get(
        "SVM_META_PATH", str(MODELS_DIR / "role_classifier_svm_meta.json")
    )

    # ── Type classification ────────────────────────────────────────────────────
    # Slugs that are role-based career roadmaps (everything else = skill)
    ROLE_SLUGS: frozenset = frozenset({
        "frontend", "backend", "full-stack", "devops", "devsecops",
        "android", "ios", "data-analyst", "ai-engineer", "ai-data-scientist",
        "data-engineer", "machine-learning", "postgresql-dba", "blockchain",
        "qa", "software-architect", "cyber-security", "ux-design",
        "game-developer", "mlops", "product-manager", "devrel",
        "engineering-manager", "technical-writer", "devops-engineer",
        "network-engineer", "server-side-game-developer",
    })

    # ── Icons ──────────────────────────────────────────────────────────────────
    SLUG_ICONS: dict = {
        "frontend": "🖥️", "backend": "⚙️", "full-stack": "🔗",
        "devops": "🚀", "devsecops": "🔒", "android": "📱", "ios": "🍎",
        "data-analyst": "📊", "ai-engineer": "🤖", "ai-data-scientist": "🧠",
        "data-engineer": "🗄️", "machine-learning": "⚡", "postgresql-dba": "🐘",
        "blockchain": "⛓️", "qa": "🧪", "software-architect": "🏗️",
        "cyber-security": "🛡️", "ux-design": "🎨", "game-developer": "🎮",
        "mlops": "🔄", "product-manager": "📋", "devrel": "🤝",
        "engineering-manager": "👔", "technical-writer": "✍️",
        "network-engineer": "🌐",
        # Skills
        "python": "🐍", "javascript": "🟨", "typescript": "🔷",
        "react": "⚛️", "nodejs": "🟩", "sql": "🗃️",
        "computer-science": "💻", "vue": "💚", "angular": "🔴",
        "java": "☕", "golang": "🐹", "rust": "🦀", "cpp": "⚙️",
        "docker": "🐳", "kubernetes": "☸️", "linux": "🐧", "aws": "☁️",
        "mongodb": "🍃", "react-native": "📱", "flutter": "🎯",
        "git-github": "📦", "system-design": "🏛️",
        "datastructures-and-algorithms": "🧮", "terraform": "🌍",
        "nextjs": "▲", "spring-boot": "🌱", "redis": "🔴",
        "graphql": "🔮", "php": "🐘", "html": "🌐", "css": "🎨",
        "django": "🎸", "ruby": "💎", "prompt-engineering": "✨",
        "swift": "🍊", "kotlin": "🟣", "scala": "🔺", "r": "📉",
        "c": "🔵", "c-sharp": "💜", "aspnet-core": "💜",
        "wordpress": "🔵", "laravel": "🔴",
    }

    @classmethod
    def local_repo_path(cls) -> Path:
        """Returns the resolved local repo path, or empty Path if not set."""
        if cls.ROADMAP_REPO_PATH and cls.ROADMAP_REPO_PATH.exists():
            return cls.ROADMAP_REPO_PATH
        return Path("")

    @classmethod
    def _roadmaps_subpath_for(cls, repo: Path) -> Path:
        """
        The developer-roadmap repo moved its data. Older commits keep each
        roadmap under src/data/roadmaps/<slug>/ (with <slug>.json hierarchy +
        <slug>.md metadata + content/); the current repo has only
        roadmaps/<slug>/content/*.md. Use whichever layout exists locally.
        """
        for cand in (cls.ROADMAPS_SUBPATH, Path("roadmaps")):
            if (repo / cand).is_dir():
                return cand
        return cls.ROADMAPS_SUBPATH

    @classmethod
    def is_local_mode(cls) -> bool:
        p = cls.local_repo_path()
        if str(p) in ("", "."):
            return False
        return (p / cls._roadmaps_subpath_for(p)).is_dir()

    @classmethod
    def roadmaps_dir(cls) -> Path:
        """Local path to the roadmaps/ folder inside the repo clone
        (src/data/roadmaps or roadmaps, auto-detected)."""
        p = cls.local_repo_path()
        return p / cls._roadmaps_subpath_for(p)

    @classmethod
    def raw_github_url(cls, path: str) -> str:
        return (f"https://raw.githubusercontent.com/{cls.GITHUB_OWNER}/"
                f"{cls.GITHUB_REPO}/{cls.GITHUB_BRANCH}/{path}")

    @classmethod
    def github_api_url(cls, path: str) -> str:
        return f"https://api.github.com/repos/{cls.GITHUB_OWNER}/{cls.GITHUB_REPO}/{path}"