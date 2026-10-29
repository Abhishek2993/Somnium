"""
db.py — SQLite database layer for Somnium
"""
import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'somnium.db')


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dreams (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                date        TEXT NOT NULL,
                title       TEXT,
                text        TEXT NOT NULL,
                genre       TEXT,
                confidence  REAL DEFAULT 0.0,
                mood        TEXT,
                type        TEXT DEFAULT 'Normal',
                keywords    TEXT,  -- JSON array stored as string
                entities    TEXT,  -- JSON array of NER results
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Reddit-collected dream posts (see data/collect_reddit.py). Never
        # stores usernames -- author is used only to filter deleted posts.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reddit_dreams (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id       TEXT NOT NULL UNIQUE,
                subreddit     TEXT NOT NULL,
                created_utc   REAL NOT NULL,
                flair         TEXT,
                title         TEXT NOT NULL,
                body          TEXT NOT NULL,
                score         INTEGER DEFAULT 0,
                num_comments  INTEGER DEFAULT 0,
                word_count    INTEGER NOT NULL,
                content_hash  TEXT NOT NULL UNIQUE,
                collected_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Optional local DreamBank corpus (see data/dreambank_loader.py).
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dreambank_dreams (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                source_label  TEXT,
                text          TEXT NOT NULL,
                word_count    INTEGER NOT NULL,
                content_hash  TEXT NOT NULL UNIQUE,
                collected_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


def insert_dream(dream: dict) -> int:
    with get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO dreams (date, title, text, genre, confidence, mood, type, keywords, entities)
            VALUES (:date, :title, :text, :genre, :confidence, :mood, :type, :keywords, :entities)
        """, {
            'date':       dream.get('date', ''),
            'title':      dream.get('title', ''),
            'text':       dream.get('text', ''),
            'genre':      dream.get('genre', 'Surreal'),
            'confidence': dream.get('confidence', 0.0),
            'mood':       dream.get('mood', ''),
            'type':       dream.get('type', 'Normal'),
            'keywords':   json.dumps(dream.get('keywords', [])),
            'entities':   json.dumps(dream.get('entities', [])),
        })
        conn.commit()
        return cur.lastrowid


def get_all_dreams() -> list:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM dreams ORDER BY created_at DESC"
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d['keywords'] = json.loads(d.get('keywords') or '[]')
        d['entities'] = json.loads(d.get('entities') or '[]')
        result.append(d)
    return result


def delete_dream(dream_id: int) -> bool:
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM dreams WHERE id = ?", (dream_id,))
        conn.commit()
        return cur.rowcount > 0


def get_dream_by_id(dream_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM dreams WHERE id = ?", (dream_id,)
        ).fetchone()
    if row:
        d = dict(row)
        d['keywords'] = json.loads(d.get('keywords') or '[]')
        d['entities'] = json.loads(d.get('entities') or '[]')
        return d
    return None


# ── reddit_dreams ────────────────────────────────────────────────────────────

def insert_reddit_dream(record: dict) -> int | None:
    """
    Insert a collected Reddit post. Returns the new row id, or None if a
    row with the same post_id or content_hash already exists (resumable:
    safe to call repeatedly with overlapping data).
    """
    with get_conn() as conn:
        cur = conn.execute("""
            INSERT OR IGNORE INTO reddit_dreams
                (post_id, subreddit, created_utc, flair, title, body,
                 score, num_comments, word_count, content_hash)
            VALUES (:post_id, :subreddit, :created_utc, :flair, :title, :body,
                    :score, :num_comments, :word_count, :content_hash)
        """, record)
        conn.commit()
        return cur.lastrowid if cur.rowcount > 0 else None


def get_reddit_dreams(subreddit: str | None = None) -> list:
    query = "SELECT * FROM reddit_dreams"
    params: tuple = ()
    if subreddit:
        query += " WHERE subreddit = ?"
        params = (subreddit,)
    query += " ORDER BY created_utc DESC"
    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def get_reddit_post_ids(subreddit: str | None = None) -> set:
    query = "SELECT post_id FROM reddit_dreams"
    params: tuple = ()
    if subreddit:
        query += " WHERE subreddit = ?"
        params = (subreddit,)
    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
    return {r['post_id'] for r in rows}


# ── dreambank_dreams ─────────────────────────────────────────────────────────

def insert_dreambank_dream(record: dict) -> int | None:
    """Insert a dream parsed from a local DreamBank file. See insert_reddit_dream."""
    with get_conn() as conn:
        cur = conn.execute("""
            INSERT OR IGNORE INTO dreambank_dreams (source_label, text, word_count, content_hash)
            VALUES (:source_label, :text, :word_count, :content_hash)
        """, record)
        conn.commit()
        return cur.lastrowid if cur.rowcount > 0 else None


def get_dreambank_dreams() -> list:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM dreambank_dreams ORDER BY id").fetchall()
    return [dict(r) for r in rows]
