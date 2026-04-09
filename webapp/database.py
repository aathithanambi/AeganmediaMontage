from __future__ import annotations

from pymongo import MongoClient
from pymongo.database import Database

from webapp.config import settings

_client: MongoClient | None = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=5000)
    return _client


def get_db() -> Database:
    return get_client()[settings.mongodb_db]

