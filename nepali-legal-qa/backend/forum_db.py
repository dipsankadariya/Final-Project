import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

DB_PATH = os.getenv("FORUM_DB_PATH", "forum.db")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                tags TEXT NOT NULL DEFAULT '',
                author_sub TEXT NOT NULL,
                author_name TEXT NOT NULL,
                author_picture TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                upvotes INTEGER NOT NULL DEFAULT 0,
                downvotes INTEGER NOT NULL DEFAULT 0,
                answer_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id INTEGER NOT NULL,
                body TEXT NOT NULL,
                author_sub TEXT NOT NULL,
                author_name TEXT NOT NULL,
                author_picture TEXT,
                created_at TEXT NOT NULL,
                upvotes INTEGER NOT NULL DEFAULT 0,
                downvotes INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY(question_id) REFERENCES questions(id) ON DELETE CASCADE
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_questions_created ON questions(created_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_questions_updated ON questions(updated_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_answers_question_id ON answers(question_id)")


def normalize_tags(raw_tags: list[str]) -> list[str]:
    tags = []
    for tag in raw_tags:
        cleaned = tag.strip().lower()
        if not cleaned:
            continue
        if cleaned not in tags:
            tags.append(cleaned)
    return tags[:6]
