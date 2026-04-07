from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ConversationStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_conversation_id
                ON messages (conversation_id, id);
                """
            )

    def list_conversations(self, limit: int = 50, search_query: str = "") -> list[dict[str, Any]]:
        like_query = f"%{search_query.strip()}%"
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    conversations.id,
                    conversations.title,
                    conversations.created_at,
                    conversations.updated_at,
                    COUNT(messages.id) AS message_count
                FROM conversations
                LEFT JOIN messages ON messages.conversation_id = conversations.id
                WHERE
                    ? = ''
                    OR conversations.title LIKE ? COLLATE NOCASE
                    OR EXISTS (
                        SELECT 1
                        FROM messages AS matched_messages
                        WHERE matched_messages.conversation_id = conversations.id
                        AND matched_messages.content LIKE ? COLLATE NOCASE
                    )
                GROUP BY conversations.id
                ORDER BY conversations.updated_at DESC, conversations.id DESC
                LIMIT ?
                """,
                (search_query.strip(), like_query, like_query, limit),
            ).fetchall()

        return [dict(row) for row in rows]

    def create_conversation(self, title: str = "New chat") -> int:
        now = utc_now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO conversations (title, created_at, updated_at)
                VALUES (?, ?, ?)
                """,
                (title, now, now),
            )
            return int(cursor.lastrowid)

    def rename_conversation(self, conversation_id: int, title: str) -> None:
        clean_title = " ".join(title.split()).strip() or "New chat"
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE conversations
                SET title = ?, updated_at = ?
                WHERE id = ?
                """,
                (clean_title, utc_now(), conversation_id),
            )

    def get_messages(self, conversation_id: int) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, role, content, sources_json, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id ASC
                """,
                (conversation_id,),
            ).fetchall()

        messages: list[dict[str, Any]] = []
        for row in rows:
            sources = []
            raw_sources = row["sources_json"]
            if raw_sources:
                try:
                    sources = json.loads(raw_sources)
                except json.JSONDecodeError:
                    sources = []

            messages.append(
                {
                    "id": int(row["id"]),
                    "role": row["role"],
                    "content": row["content"],
                    "sources": sources,
                    "created_at": row["created_at"],
                }
            )
        return messages

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        sources: list[dict[str, Any]] | None = None,
    ) -> None:
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO messages (conversation_id, role, content, sources_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conversation_id, role, content, json.dumps(sources or []), now),
            )
            connection.execute(
                """
                UPDATE conversations
                SET updated_at = ?
                WHERE id = ?
                """,
                (now, conversation_id),
            )

    def delete_conversation(self, conversation_id: int) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                DELETE FROM messages
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            )
            connection.execute(
                """
                DELETE FROM conversations
                WHERE id = ?
                """,
                (conversation_id,),
            )
