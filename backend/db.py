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
