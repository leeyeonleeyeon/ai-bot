from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def save_local(out_dir: str, goal_id: str, content: str) -> str:
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = base / f"{ts}_{goal_id[:8]}.md"
    path.write_text(content, encoding="utf-8")
    return str(path)
