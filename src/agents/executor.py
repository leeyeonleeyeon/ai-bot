from __future__ import annotations

import re
from typing import Optional

from src.agents.base import AgentContext, BaseAgent
from src.llm.gemma_client import GemmaClient
from src.tools.e2b_runner import E2BRunner

_MAX_TOOL_ITER = 5
_RUN_TAG_RE = re.compile(r"<run_python>(.*?)</run_python>", re.DOTALL | re.IGNORECASE)

_BASE_PROMPT = (
    "당신은 EXECUTOR입니다. history에 담긴 planner의 메모와 제공된 지식을 활용해, "
    "이 서브태스크의 구체적 산출물(코드·문서·초안·상세 답변)을 작성하세요. "
    "서브태스크와 같은 언어로 응답하되, 코드와 식별자는 원형 그대로 유지하세요."
)

_TOOL_PROMPT = (
    "\n\n파이썬 샌드박스(e2b)를 사용할 수 있습니다. 코드를 실행하려면 다음 태그 안에 출력하세요:\n"
    "<run_python>\n# 코드는 여기에\nprint(...)\n</run_python>\n"
    "실행 후에는 <stdout>...</stdout>, <stderr>...</stderr>, "
    "<result>...</result>, 또는 <error>...</error>를 받게 됩니다. "
    "계산이 필요하거나, URL을 가져오거나, 파일을 다루거나, 코드 동작을 확인할 때 샌드박스를 쓰세요.\n"
    "최종 산출물이 준비되면 <run_python> 태그 없이 평문으로 출력하세요 — 그게 작업 완료 신호입니다."
)


def _extract_run(text: str) -> Optional[str]:
    m = _RUN_TAG_RE.search(text)
    return m.group(1).strip() if m else None


class ExecutorAgent(BaseAgent):
    role = "executor"

    def __init__(self, llm: GemmaClient, e2b_api_key: Optional[str] = None):
        super().__init__(llm)
        self.e2b_api_key = e2b_api_key
        self.system_prompt = _BASE_PROMPT + (_TOOL_PROMPT if e2b_api_key else "")

    async def run(self, ctx: AgentContext) -> dict:
        if not self.e2b_api_key:
            out = await self._think(ctx, ctx.goal, temperature=0.5)
            return {"output": out}
        return await self._tool_loop(ctx)

    async def _tool_loop(self, ctx: AgentContext) -> dict:
        history = list(ctx.history)
        user_msg = ctx.goal
        last_out = ""
        runs_log: list[str] = []

        async with E2BRunner(self.e2b_api_key) as runner:  # type: ignore[arg-type]
            for step in range(_MAX_TOOL_ITER):
                turn_ctx = AgentContext(
                    task_id=ctx.task_id,
                    goal=ctx.goal,
                    knowledge=ctx.knowledge,
                    history=history,
                )
                out = await self._think(turn_ctx, user_msg, temperature=0.4)
                last_out = out

                code = _extract_run(out)
                if not code:
                    return {"output": out, "tool_steps": step}

                history.append({"role": "assistant", "content": out})
                try:
                    result = await runner.run(code)
                    feedback = result.format_feedback()
                except Exception as e:
                    feedback = f"<error>\nrunner failure: {e}\n</error>"

                runs_log.append(f"step {step + 1}: ran {len(code)} chars")
                history.append({"role": "user", "content": feedback})
                user_msg = (
                    "Continue. When you have the final deliverable, output it as "
                    "plain text without any <run_python> tag."
                )

        return {
            "output": last_out
            + f"\n\n[note: max tool iterations ({_MAX_TOOL_ITER}) reached]",
            "tool_steps": _MAX_TOOL_ITER,
        }
