
import sqlite3
from pathlib import Path
from datetime import datetime, timezone


DB_PATH = Path(__file__).resolve().parent / "nexusai_memory.db"


def get_connection():
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL DEFAULT 'New Chat',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(id)
                    ON DELETE CASCADE
            )
        """)


def create_conversation(title="New Chat"):
    now = datetime.now(timezone.utc).isoformat()

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO conversations (title, created_at, updated_at)
            VALUES (?, ?, ?)
            """,
            (title, now, now),
        )
        return cursor.lastrowid


def list_conversations():
    with get_connection() as connection:
        rows = connection.execute("""
            SELECT
                c.id,
                c.title,
                c.created_at,
                c.updated_at,
                COUNT(m.id) AS message_count
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            GROUP BY c.id
            ORDER BY c.updated_at DESC
        """).fetchall()

        return [dict(row) for row in rows]


def add_message(conversation_id, role, content):
    if role not in ("user", "assistant"):
        raise ValueError("Role must be user or assistant.")

    now = datetime.now(timezone.utc).isoformat()

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO messages
                (conversation_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, role, content, now),
        )

        connection.execute(
            """
            UPDATE conversations
            SET updated_at = ?
            WHERE id = ?
            """,
            (now, conversation_id),
        )


def get_messages(conversation_id):
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT role, content, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()

        return [dict(row) for row in rows]


def rename_conversation(conversation_id, title):
    title = title.strip()

    if not title:
        raise ValueError("Conversation title cannot be empty.")

    with get_connection() as connection:
        connection.execute(
            """
            UPDATE conversations
            SET title = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                title[:100],
                datetime.now(timezone.utc).isoformat(),
                conversation_id,
            ),
        )


def delete_conversation(conversation_id):
    with get_connection() as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            "DELETE FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        )
        connection.execute(
            "DELETE FROM conversations WHERE id = ?",
            (conversation_id,),
        )


initialize_database()
