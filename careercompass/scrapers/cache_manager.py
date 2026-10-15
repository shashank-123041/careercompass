import json
import os
import time
import logging
import threading

from careercompass.config import Config

logger = logging.getLogger(__name__)

CACHE_DIR = str(Config.SCRAPER_CACHE_DIR)  # data/cache/scrapers (gitignored)
CACHE_TTL = 3600  # 1 hour

_lock = threading.Lock()

def _cache_path(key):
    return os.path.join(CACHE_DIR, f"{key}.json")

def get(key):
    path = _cache_path(key)
    try:
        with _lock:
            if not os.path.exists(path):
                return None
            with open(path, "r") as f:
                data = json.load(f)
            if time.time() - data.get("timestamp", 0) > CACHE_TTL:
                return None
            return data.get("value")
    except Exception as e:
        logger.debug(f"Cache get error for {key}: {e}")
        return None

def set(key, value):
    path = _cache_path(key)
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with _lock:
            with open(path, "w") as f:
                json.dump({"timestamp": time.time(), "value": value}, f)
    except Exception as e:
        logger.debug(f"Cache set error for {key}: {e}")

def get_age(key):
    path = _cache_path(key)
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            data = json.load(f)
        age_secs = int(time.time() - data.get("timestamp", 0))
        if age_secs < 60:
            return "Just now"
        elif age_secs < 3600:
            return f"{age_secs // 60}m ago"
        else:
            return f"{age_secs // 3600}h ago"
    except:
        return None

def invalidate(key):
    path = _cache_path(key)
    try:
        if os.path.exists(path):
            os.remove(path)
    except:
        pass