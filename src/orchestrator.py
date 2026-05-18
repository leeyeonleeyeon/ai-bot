from __future__ import annotations

import logging
import os
import re
from typing import Awaitable, Callable, Optional

from src.agents.base import AgentContext
from src.agents.executor import ExecutorAgent
from src.agents.leader import LeaderAgent
from src.agents.planner import PlannerAgent
from src.agents.reviewer import ReviewerAgent
from src.llm.gemma_client import GemmaClient
from src.storage.github import GitHubCommitter
from src.storage.output import save_local
from src.task_queue.queue import TaskQueue, TaskStatus

log = logging.getLogger(__name__)

ProgressCb = Callable[[str], Awaitable[None]]

_VERDICT_RE = re.compile(r"\[VERDICT:\s*(ACCEPT|REVISE)\s*\]", re.IGNORECASE)
_MAX_REVISE_PER_GOAL = 1

_TRIVIAL_MAX_LEN = 30
_FAST_SYSTEM_PROMPT = (
    "사용자가 한 줄짜리 짧은 메시지를 보냈습니다. 분석이나 계획을 길게 늘어놓지 말고, "
    "한 번에 자연스럽고 친근하게 답하세요. 사용자 메시지와 같은 언어로 응답하세요."
)


def _extract_verdict(text: str) -> Optional[str]:
    m = _VERDICT_RE.search(text or "")
    return m.group(1).lower() if m else None


def _is_trivial_input(text: str) -> bool:
    s = (text or "").strip()
    if not s or len(s) >= _TRIVIAL_MAX_LEN:
        return False
    if "\n" in s or "```" in s:
        return False
    if "http://" in s or "https://" in s:
        return False
    return True


class Orchestrator:
    """리더 → 플래너/실행/리뷰 파이프라인을 조율한다."""

    def __init__(
        self,
        llm: GemmaClient,
        queue: TaskQueue,
        output_dir: Optional[str] = None,
        e2b_api_key: Optional[str] = None,
        github: Optional[GitHubCommitter] = None,
    ):
        self.llm = llm
        self.queue = queue
        self.output_dir = output_dir or os.environ.get("OUTPUT_DIR", "data/outputs")
        self.github = github
        self.leader = LeaderAgent(llm)
        self.agents = {
            "planner": PlannerAgent(llm),
            "executor": ExecutorAgent(llm, e2b_api_key=e2b_api_key),
            "reviewer": ReviewerAgent(llm),
        }

    async def run_goal(
        self,
        chat_id: int,
        goal_description: str,
        on_update: Optional[ProgressCb] = None,
    ) -> dict:
        async def notify(msg: str) -> None:
            if on_update:
                await on_update(msg)

        knowledge_rows = await self.queue.list_knowledge(chat_id)
        knowledge = [k["content"] for k in knowledge_rows]

        goal_id = await self.queue.create_goal(chat_id, goal_description)
        await self.queue.update_goal(goal_id, TaskStatus.IN_PROGRESS)

        if _is_trivial_input(goal_description):
            return await self._run_fast(goal_id, goal_description, knowledge, notify)

        await notify("📋 리더가 목표를 분해 중...")

        try:
            tasks = await self.leader.decompose(goal_description, knowledge)
        except Exception as e:
            log.exception("decomposition failed")
            await self.queue.update_goal(goal_id, TaskStatus.FAILED, f"decomposition failed: {e}")
            raise

        if not tasks:
            await self.queue.update_goal(goal_id, TaskStatus.FAILED, "no tasks produced")
            return {"goal_id": goal_id, "result": None, "error": "no tasks"}

        for i, t in enumerate(tasks):
            await self.queue.add_task(goal_id, t["role"], t["description"], sequence=i)

        await notify(f"🧩 {len(tasks)}개 서브태스크로 분해됨")

        outputs: list[str] = []
        history: list[dict] = []
        revise_used = 0

        while True:
            task = await self.queue.next_pending_task(goal_id)
            if not task:
                break

            role = task["assigned_role"]
            agent = self.agents.get(role)
            if not agent:
                await self.queue.update_task(
                    task["id"], TaskStatus.FAILED, f"no agent for role {role}"
                )
                continue

            await self.queue.update_task(task["id"], TaskStatus.IN_PROGRESS)
            await notify(f"🤖 [{role}] {task['description']}")

            ctx = AgentContext(
                task_id=task["id"],
                goal=task["description"],
                knowledge=knowledge,
                history=history.copy(),
            )
            try:
                result = await agent.run(ctx)
                output = (result.get("output") or "").strip()
                await self.queue.update_task(task["id"], TaskStatus.DONE, output)
                outputs.append(f"### [{role}] {task['description']}\n\n{output}")
                history.append({"role": "assistant", "content": f"[{role}]\n{output}"})
            except Exception as e:
                log.exception("task %s failed", task["id"])
                await self.queue.update_task(task["id"], TaskStatus.FAILED, str(e))
                outputs.append(f"### [{role}] FAILED\n\n{e}")
                continue

            if role == "reviewer" and revise_used < _MAX_REVISE_PER_GOAL:
                if _extract_verdict(output) == "revise":
                    prev = await self.queue.find_last_done_task(goal_id, "executor")
                    if prev:
                        seq = await self.queue.next_sequence(goal_id)
                        revise_desc = (
                            "이전 산출물을 reviewer의 비평을 반영해 수정하세요. "
                            "비평에서 지적한 문제만 고치고 새 기능을 추가하지 마세요.\n\n"
                            f"[원본 서브태스크]\n{prev['description']}\n\n"
                            f"[원본 산출물]\n{prev.get('output') or ''}\n\n"
                            f"[reviewer 비평]\n{output}"
                        )
                        await self.queue.add_task(
                            goal_id, "executor", revise_desc, sequence=seq
                        )
                        revise_used += 1
                        await notify("🔁 reviewer가 수정 요청 — executor 재실행")

        final = "\n\n---\n\n".join(outputs) if outputs else "(no output)"
        await self.queue.update_goal(goal_id, TaskStatus.DONE, final)

        try:
            saved = save_local(self.output_dir, goal_id, final)
            log.info("result saved: %s", saved)
        except Exception as e:
            log.warning("save_local failed: %s", e)

        github_url = None
        if self.github:
            from datetime import datetime, timezone

            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            path = f"results/{ts}_{goal_id[:8]}.md"
            body = (
                f"# {goal_description}\n\n"
                f"_goal_id: `{goal_id}` · chat: `{chat_id}` · ts: `{ts}`_\n\n"
                f"---\n\n{final}\n"
            )
            try:
                github_url = await self.github.commit_file(
                    path=path,
                    content=body,
                    message=f"ai-bot: {goal_description[:60]}",
                )
                log.info("github commit: %s", github_url)
                await notify(f"📤 GitHub 커밋: {github_url}")
            except Exception as e:
                log.warning("github commit failed: %s", e)
                await notify(f"⚠️ GitHub 커밋 실패: {e}")

        return {
            "goal_id": goal_id,
            "result": final,
            "error": None,
            "github_url": github_url,
        }

    async def _run_fast(
        self,
        goal_id: str,
        goal_description: str,
        knowledge: list[str],
        notify: Callable[[str], Awaitable[None]],
    ) -> dict:
        await notify("⚡ 빠른 응답 모드 (단순 입력 감지)")
        tid = await self.queue.add_task(goal_id, "fast", goal_description, sequence=0)
        await self.queue.update_task(tid, TaskStatus.IN_PROGRESS)

        messages: list[dict] = [{"role": "system", "content": _FAST_SYSTEM_PROMPT}]
        for k in knowledge:
            messages.append({"role": "system", "content": f"[KNOWLEDGE]\n{k}"})
        messages.append({"role": "user", "content": goal_description})

        try:
            answer = (await self.llm.complete(messages, temperature=0.5)).strip()
        except Exception as e:
            log.exception("fast response failed")
            await self.queue.update_task(tid, TaskStatus.FAILED, str(e))
            await self.queue.update_goal(
                goal_id, TaskStatus.FAILED, f"fast response failed: {e}"
            )
            raise

        await self.queue.update_task(tid, TaskStatus.DONE, answer)
        final = answer or "(no output)"
        await self.queue.update_goal(goal_id, TaskStatus.DONE, final)

        try:
            saved = save_local(self.output_dir, goal_id, final)
            log.info("result saved (fast): %s", saved)
        except Exception as e:
            log.warning("save_local failed: %s", e)

        return {
            "goal_id": goal_id,
            "result": final,
            "error": None,
            "github_url": None,
        }
