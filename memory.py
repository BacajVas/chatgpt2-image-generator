"""Persistent, single-user memory for conversations and generated images."""

from __future__ import annotations

import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


PROFILE_ID = "local"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MemoryStore:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.images_dir = self.data_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.database = self.data_dir / "memory.sqlite3"
        if os.name == "posix":
            os.chmod(self.data_dir, 0o700)
            os.chmod(self.images_dir, 0o700)
        descriptor = os.open(self.database, os.O_CREAT | os.O_WRONLY, 0o600)
        os.close(descriptor)
        if os.name == "posix":
            os.chmod(self.database, 0o600)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS profiles (
                    id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL DEFAULT 'Friend',
                    preferred_style TEXT NOT NULL DEFAULT '',
                    notes TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS generations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_id TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    effective_prompt TEXT NOT NULL,
                    style TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    steps INTEGER NOT NULL,
                    seed INTEGER NOT NULL,
                    image_file TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(profiles)")
            }
            if "notes" not in columns:
                connection.execute(
                    "ALTER TABLE profiles ADD COLUMN notes TEXT NOT NULL DEFAULT ''"
                )
            connection.execute(
                "INSERT OR IGNORE INTO profiles (id) VALUES (?)", (PROFILE_ID,)
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def profile(self) -> dict:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT display_name, preferred_style, notes FROM profiles WHERE id = ?",
                (PROFILE_ID,),
            ).fetchone()
        return dict(row)

    def set_name(self, name: str) -> None:
        name = name.strip()
        if not name or len(name) > 80:
            raise ValueError("Enter a name of up to 80 characters.")
        with self._connect() as connection:
            connection.execute(
                "UPDATE profiles SET display_name = ? WHERE id = ?", (name, PROFILE_ID)
            )

    def set_style(self, style: str) -> None:
        style = style.strip()
        if len(style) > 400:
            raise ValueError("The preferred style must be at most 400 characters.")
        with self._connect() as connection:
            connection.execute(
                "UPDATE profiles SET preferred_style = ? WHERE id = ?",
                (style, PROFILE_ID),
            )

    def set_notes(self, notes: str) -> None:
        notes = notes.strip()
        if len(notes) > 1500:
            raise ValueError("Personal notes must be at most 1500 characters.")
        with self._connect() as connection:
            connection.execute(
                "UPDATE profiles SET notes = ? WHERE id = ?", (notes, PROFILE_ID)
            )

    def messages(self, limit: int = 20) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT role, content, created_at FROM messages
                   WHERE profile_id = ? ORDER BY id DESC LIMIT ?""",
                (PROFILE_ID, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def relevant_older_messages(self, query: str, limit: int = 4) -> list[dict]:
        """Recall older turns with matching word stems, beyond the recent context."""
        terms = {
            word[:4] for word in re.findall(r"[^\W\d_]{4,}", query.casefold())
        }
        if not terms:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT id, role, content FROM messages WHERE profile_id = ?
                   ORDER BY id DESC LIMIT 500""",
                (PROFILE_ID,),
            ).fetchall()
        ranked = []
        for row in rows[12:]:
            words = {
                word[:4] for word in re.findall(
                    r"[^\W\d_]{4,}", row["content"].casefold()
                )
            }
            score = len(terms & words)
            if score:
                ranked.append((score, row["id"], dict(row)))
        ranked.sort(key=lambda match: (match[0], match[1]), reverse=True)
        return [item for _, _, item in sorted(ranked[:limit], key=lambda match: match[1])]

    def add_exchange(self, user_text: str, assistant_text: str) -> None:
        with self._connect() as connection:
            connection.executemany(
                """INSERT INTO messages (profile_id, role, content, created_at)
                   VALUES (?, ?, ?, ?)""",
                [
                    (PROFILE_ID, "user", user_text, _now()),
                    (PROFILE_ID, "assistant", assistant_text, _now()),
                ],
            )

    def clear_messages(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM messages WHERE profile_id = ?", (PROFILE_ID,))

    def save_generation(
        self,
        image: Image.Image,
        *,
        prompt: str,
        effective_prompt: str,
        style: str,
        model_id: str,
        steps: int,
        seed: int,
    ) -> dict:
        image_file = f"{uuid.uuid4().hex}.png"
        destination = self.images_dir / image_file
        temporary = self.images_dir / f".{image_file}.tmp.png"
        try:
            image.save(temporary, format="PNG")
            if os.name == "posix":
                os.chmod(temporary, 0o600)
            os.replace(temporary, destination)
            with self._connect() as connection:
                cursor = connection.execute(
                    """INSERT INTO generations
                       (profile_id, prompt, effective_prompt, style, model_id,
                        steps, seed, image_file, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        PROFILE_ID,
                        prompt,
                        effective_prompt,
                        style,
                        model_id,
                        steps,
                        seed,
                        image_file,
                        _now(),
                    ),
                )
                record_id = cursor.lastrowid
        except Exception:
            temporary.unlink(missing_ok=True)
            destination.unlink(missing_ok=True)
            raise
        return self.generation(record_id)

    def generation(self, record_id: int) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM generations WHERE id = ? AND profile_id = ?",
                (record_id, PROFILE_ID),
            ).fetchone()
        return self._generation_dict(row) if row else None

    def generations(self, limit: int = 24) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM generations WHERE profile_id = ?
                   ORDER BY id DESC LIMIT ?""",
                (PROFILE_ID, limit),
            ).fetchall()
        return [self._generation_dict(row) for row in rows]

    def _generation_dict(self, row: sqlite3.Row) -> dict:
        record = dict(row)
        record["image_path"] = str(self.images_dir / record["image_file"])
        return record
