"""
auth package — Firebase Admin SDK initialization.

Credentials are read, in order, from:
  1. FIREBASE_SERVICE_ACCOUNT_JSON — the key's JSON, raw or base64-encoded
     (use this on hosting platforms that provide env vars, not secret files)
  2. FIREBASE_SERVICE_ACCOUNT_PATH — path to the key file (local development)

Importing this module (done once from create_app()) boots the firebase_admin
app used by auth/routes.py to verify ID tokens sent from the frontend.
"""
import base64
import json
import logging
import os

import firebase_admin
from firebase_admin import credentials

from careercompass.config import Config

log = logging.getLogger(__name__)


def _load_credential():
    raw = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
    if raw:
        if not raw.startswith("{"):          # assume base64
            try:
                raw = base64.b64decode(raw).decode("utf-8")
            except Exception as e:
                raise ValueError(
                    "FIREBASE_SERVICE_ACCOUNT_JSON is neither valid JSON nor "
                    "valid base64-encoded JSON."
                ) from e
        try:
            info = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError("FIREBASE_SERVICE_ACCOUNT_JSON is not valid JSON.") from e
        log.info("Firebase credentials loaded from FIREBASE_SERVICE_ACCOUNT_JSON")
        return credentials.Certificate(info)

    path = Config.FIREBASE_SERVICE_ACCOUNT_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Firebase service account not found at '{path}'.\n"
            f"Either set FIREBASE_SERVICE_ACCOUNT_JSON, or download the key from "
            f"Firebase Console → Project Settings → Service Accounts → Generate "
            f"new private key, place it under secrets/, and set "
            f"FIREBASE_SERVICE_ACCOUNT_PATH in .env."
        )
    log.info("Firebase credentials loaded from %s", path)
    return credentials.Certificate(path)


if not firebase_admin._apps:
    firebase_admin.initialize_app(_load_credential())
    log.info("Firebase Admin SDK initialized")