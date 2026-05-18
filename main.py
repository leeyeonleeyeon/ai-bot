from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from src.bot.telegram_bot import TelegramBot
from src.llm.gemma_client import GemmaClient
from src.orchestrator import Orchestrator
from src.storage.github import from_env as github_from_env
from src.task_queue.queue import TaskQueue

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("main")


def _required(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"missing env var: {name}")
    return v


async def _bootstrap() -> tuple[TaskQueue, Orchestrator, str]:
    db_path = os.environ.get("DB_PATH", "data/tasks.db")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    queue = TaskQueue(db_path)
    await queue.init()

    llm = GemmaClient(
        base_url=_required("GEMMA_BASE_URL"),
        model=os.environ.get("GEMMA_MODEL", "gemma-4"),
        api_key=os.environ.get("GEMMA_API_KEY") or None,
    )

    e2b_key = os.environ.get("E2B_API_KEY") or None
    if e2b_key:
        log.info("e2b sandbox enabled for executor")
    else:
        log.info("e2b not configured — executor will run in plain text mode")

    gh = github_from_env()
    if gh:
        log.info("github auto-commit enabled: %s @ %s", gh.repo, gh.branch)

    orchestrator = Orchestrator(
        llm=llm,
        queue=queue,
        e2b_api_key=e2b_key,
        github=gh,
    )
    return queue, orchestrator, _required("TELEGRAM_BOT_TOKEN")


def main() -> None:
    queue, orchestrator, token = asyncio.run(_bootstrap())
    bot = TelegramBot(token, orchestrator, queue)
    log.info("bot starting")
    bot.run()


if __name__ == "__main__":
    main()
