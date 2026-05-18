"""SQLite-backed shared task queue for the agent team."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import aiosqlite


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskQueue:
    def __init__(self, db_path: str):
        self.db_path = db_path

    async def init(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(
                """
                CREATE TABLE IF NOT EXISTS goals (
                    id          TEXT PRIMARY KEY,
                    chat_id     INTEGER NOT NULL,
                    description TEXT NOT NULL,
                    status      TEXT NOT NULL DEFAULT 'pending',
                    result      TEXT,
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tasks (
                    id            TEXT PRIMARY KEY,
                    goal_id       TEXT NOT NULL,
                    assigned_role TEXT NOT NULL,
                    description   TEXT NOT NULL,
                    inputs        TEXT,
                    output        TEXT,
                    status        TEXT NOT NULL DEFAULT 'pending',
                    sequence      INTEGER NOT NULL,
                    created_at    TEXT NOT NULL,
                    updated_at    TEXT NOT NULL,
                    FOREIGN KEY (goal_id) REFERENCES goals(id)
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_goal ON tasks(goal_id, sequence);

                CREATE TABLE IF NOT EXISTS knowledge (
                    id         TEXT PRIMARY KEY,
                    chat_id    INTEGER NOT NULL,
                    source     TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_knowledge_chat ON knowledge(chat_id, created_at);
                """
            )
            await db.commit()

    # ----- goals -----
    async def create_goal(self, chat_id: int, description: str) -> str:
        gid = str(uuid.uuid4())
        now = _now()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO goals (id, chat_id, description, status, created_at, updated_at) "
                "VALUES (?, ?, ?, 'pending', ?, ?)",
                (gid, chat_id, description, now, now),
            )
            await db.commit()
        return gid

    async def update_goal(
        self, goal_id: str, status: TaskStatus, result: Optional[str] = None
    ) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE goals SET status = ?, result = COALESCE(?, result), updated_at = ? WHERE id = ?",
                (status.value, result, _now(), goal_id),
            )
            await db.commit()

    async def latest_goal(self, chat_id: int) -> Optional[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT * FROM goals WHERE chat_id = ? ORDER BY created_at DESC LIMIT 1",
                (chat_id,),
            )
            row = await cur.fetchone()
            return dict(row) if row else None

    async def latest_completed_goal(self, chat_id: int) -> Optional[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT * FROM goals "
                "WHERE chat_id = ? AND status = 'done' AND result IS NOT NULL AND result != '' "
                "ORDER BY created_at DESC LIMIT 1",
                (chat_id,),
            )
            row = await cur.fetchone()
            return dict(row) if row else None

    async def goal_status(self, goal_id: str) -> Optional[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            gcur = await db.execute("SELECT * FROM goals WHERE id = ?", (goal_id,))
            goal = await gcur.fetchone()
            if not goal:
                return None
            tcur = await db.execute(
                "SELECT * FROM tasks WHERE goal_id = ? ORDER BY sequence", (goal_id,)
            )
            tasks = await tcur.fetchall()
            return {"goal": dict(goal), "tasks": [dict(t) for t in tasks]}

    # ----- tasks -----
    async def add_task(
        self,
        goal_id: str,
        role: str,
        description: str,
        sequence: int,
        inputs: Optional[dict] = None,
    ) -> str:
        tid = str(uuid.uuid4())
        now = _now()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO tasks (id, goal_id, assigned_role, description, inputs, "
                "status, sequence, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?)",
                (tid, goal_id, role, description, json.dumps(inputs or {}), sequence, now, now),
            )
            await db.commit()
        return tid

    async def next_pending_task(self, goal_id: str) -> Optional[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT * FROM tasks WHERE goal_id = ? AND status = 'pending' "
                "ORDER BY sequence ASC LIMIT 1",
                (goal_id,),
            )
            row = await cur.fetchone()
            return dict(row) if row else None

    async def update_task(
        self, task_id: str, status: TaskStatus, output: Optional[str] = None
    ) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE tasks SET status = ?, output = COALESCE(?, output), updated_at = ? WHERE id = ?",
                (status.value, output, _now(), task_id),
            )
            await db.commit()

    async def find_last_done_task(self, goal_id: str, role: str) -> Optional[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT * FROM tasks WHERE goal_id = ? AND assigned_role = ? AND status = 'done' "
                "ORDER BY sequence DESC LIMIT 1",
                (goal_id, role),
            )
            row = await cur.fetchone()
            return dict(row) if row else None

    async def next_sequence(self, goal_id: str) -> int:
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "SELECT COALESCE(MAX(sequence), -1) + 1 FROM tasks WHERE goal_id = ?",
                (goal_id,),
            )
            row = await cur.fetchone()
            return int(row[0]) if row else 0

    # ----- knowledge / brain pack -----
    async def add_knowledge(self, chat_id: int, source: str, content: str) -> str:
        kid = str(uuid.uuid4())
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO knowledge (id, chat_id, source, content, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (kid, chat_id, source, content, _now()),
            )
            await db.commit()
        return kid

    async def list_knowledge(self, chat_id: int) -> list[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT * FROM knowledge WHERE chat_id = ? ORDER BY created_at DESC",
                (chat_id,),
            )
            return [dict(r) for r in await cur.fetchall()]

    # ----- reset -----
    async def reset_chat(self, chat_id: int) -> int:
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "UPDATE goals SET status = 'failed', updated_at = ? "
                "WHERE chat_id = ? AND status IN ('pending', 'in_progress')",
                (_now(), chat_id),
            )
            await db.commit()
            return cur.rowcount or 0
