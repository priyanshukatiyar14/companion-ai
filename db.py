import sqlite3
import json
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent / "companion_memory.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    fact_key TEXT NOT NULL,          -- e.g. "relationship_status", "job", "pet_name"
    fact_value TEXT NOT NULL,        -- e.g. "single, recently broke up with Alex"
    category TEXT NOT NULL,          -- relationship | work | plan | opinion | preference | event | other
    embedding TEXT NOT NULL,         -- JSON list[float]
    status TEXT NOT NULL DEFAULT 'active',   -- active | superseded
    superseded_by INTEGER,           -- id of the fact that replaced this one
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    source_turn INTEGER NOT NULL     -- turn number this was extracted from, for debugging/eval
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn INTEGER NOT NULL,
    role TEXT NOT NULL,              -- user | assistant
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_facts_session_key ON facts(session_id, fact_key, status);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, turn);
"""


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def encode_embedding(vec: list[float]) -> str:
    return json.dumps(vec)


def decode_embedding(s: str) -> list[float]:
    return json.loads(s)
