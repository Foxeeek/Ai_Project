"""SQLite persistence layer for approved IdeaForge outputs."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


DB_FILENAME = "approved_ideas.db"


@dataclass
class IdeaRecord:
    idea_name: str
    target_audience: str
    tech_stack: str
    markdown_spec: str


class IdeaDatabase:
    """Simple SQLite helper for storing approved ideas and tech specs."""

    def __init__(self, db_path: str | Path = DB_FILENAME) -> None:
        self.db_path = Path(db_path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS approved_ideas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    idea_name TEXT NOT NULL,
                    target_audience TEXT NOT NULL,
                    tech_stack TEXT NOT NULL,
                    markdown_spec TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.commit()

    def save_approved_idea(self, record: IdeaRecord) -> int:
        """Persist one approved idea and return inserted row id."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO approved_ideas (idea_name, target_audience, tech_stack, markdown_spec)
                VALUES (?, ?, ?, ?)
                """,
                (record.idea_name, record.target_audience, record.tech_stack, record.markdown_spec),
            )
            conn.commit()
            return int(cursor.lastrowid)
