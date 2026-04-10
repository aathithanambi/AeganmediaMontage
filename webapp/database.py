from __future__ import annotations

import certifi
from pymongo import MongoClient
from pymongo.database import Database

from webapp.config import settings

_client: MongoClient | None = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        kwargs = {"serverSelectionTimeoutMS": 5000}
        if settings.mongodb_uri.startswith("mongodb+srv://") or "tls=true" in settings.mongodb_uri.lower():
            kwargs["tlsCAFile"] = certifi.where()
        _client = MongoClient(settings.mongodb_uri, **kwargs)
    return _client


def get_db() -> Database:
    return get_client()[settings.mongodb_db]

