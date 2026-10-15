"""db package — thread-safe SQLite connection + schema init."""
import sqlite3, threading, logging
from careercompass.config import Config
from careercompass.models import SCHEMA_SQL

log = logging.getLogger(__name__)
_local = threading.local()

def get_conn() -> sqlite3.Connection:
    if not getattr(_local, "conn", None):
        _local.conn = sqlite3.connect(str(Config.DB_PATH),
                                      check_same_thread=False, timeout=30)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
    return _local.conn

def init_db():
    conn = get_conn()
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    log.info("DB ready at %s", Config.DB_PATH)
