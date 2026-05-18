from __future__ import annotations

from src.agents.base import AgentContext, BaseAgent


class PlannerAgent(BaseAgent):
    role = "planner"
    system_prompt = (
        "당신은 PLANNER입니다. 주어진 서브태스크에 대해, executor가 작업할 수 있도록 "
        "명확한 계획·조사 메모·설계 근거를 작성하세요. 간결·정확·구조적으로. "
        "군더더기 없이 평문으로 출력하세요. 서브태스크와 같은 언어로 응답하세요."
    )

    async def run(self, ctx: AgentContext) -> dict:
        out = await self._think(ctx, ctx.goal, temperature=0.4)
        return {"output": out}
