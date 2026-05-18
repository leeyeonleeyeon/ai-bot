from __future__ import annotations

from src.agents.base import AgentContext, BaseAgent


class ReviewerAgent(BaseAgent):
    role = "reviewer"
    system_prompt = (
        "당신은 REVIEWER입니다. executor의 산출물을 정확성·완성도·명료성 관점에서 비평하세요. "
        "가능하면 줄/위치를 짚어 구체적 문제를 나열하고 수정 방향을 제안하세요. "
        "수정된 산출물 자체(코드·문서·답변)는 절대 직접 작성하지 마세요 — 그것은 executor의 일입니다. "
        "본문은 서브태스크와 같은 언어로 쓰고, 응답의 가장 마지막 줄은 다음 두 형식 중 정확히 하나로 끝내세요:\n"
        "[VERDICT: ACCEPT]   ← 산출물을 그대로 채택\n"
        "[VERDICT: REVISE]   ← 비평을 반영해 executor가 다시 작성해야 함"
    )

    async def run(self, ctx: AgentContext) -> dict:
        out = await self._think(ctx, ctx.goal, temperature=0.2)
        return {"output": out}
