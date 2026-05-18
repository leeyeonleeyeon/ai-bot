from __future__ import annotations

import json

from src.agents.base import AgentContext, BaseAgent


class LeaderAgent(BaseAgent):
    role = "leader"
    system_prompt = (
        "당신은 LEADER 에이전트입니다. 사용자의 목표를 2~4개의 순차적 서브태스크로 분해하세요. "
        "필요 이상으로 잘게 쪼개지 마세요. 같은 역할(role)을 두 번 이상 쓰지 말고, "
        "보통은 planner→executor→reviewer 한 세트로 충분합니다. "
        "각 서브태스크는 정확히 하나의 역할(role)에 할당되어야 합니다:\n"
        "  - planner:  조사·사실 수집·접근 방식 설계\n"
        "  - executor: 구체적 산출물(코드·문서·초안) 작성\n"
        "  - reviewer: executor 산출물 검증과 비평\n\n"
        "오직 STRICT JSON만 출력하세요(서두 설명·마크다운 코드 펜스 모두 금지). 형식:\n"
        '{ "tasks": [ {"role": "<planner|executor|reviewer>", "description": "..."} , ... ] }\n'
        "planner 결과가 executor에 입력되도록 순서를 정하고, reviewer는 마지막에 두세요.\n\n"
        "언어 규칙: JSON 키와 role 값(planner/executor/reviewer)은 영어 그대로 두되, "
        "각 task의 description은 사용자 목표와 같은 언어로 작성하세요(사용자가 한국어면 한국어)."
    )

    async def decompose(self, goal: str, knowledge: list[str]) -> list[dict]:
        ctx = AgentContext(task_id="leader", goal=goal, knowledge=knowledge, history=[])
        raw = await self._think(
            ctx,
            f"목표:\n{goal}\n\n위 목표를 분해한 JSON을 지금 출력하세요.",
            temperature=0.1,
        )
        return _parse_tasks(raw)

    async def run(self, ctx: AgentContext) -> dict:
        tasks = await self.decompose(ctx.goal, ctx.knowledge)
        return {"tasks": tasks}


def _parse_tasks(raw: str) -> list[dict]:
    s = raw.strip()
    # markdown 코드 펜스 제거
    if s.startswith("```"):
        s = s.strip("`").strip()
        if s.lower().startswith("json"):
            s = s[4:].strip()

    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        # 본문 안에 섞여 있을 때: 첫 { 부터 마지막 } 까지만 추출
        i, j = s.find("{"), s.rfind("}")
        if i < 0 or j <= i:
            raise ValueError(f"leader produced non-JSON output: {raw[:200]!r}")
        data = json.loads(s[i : j + 1])

    tasks = data.get("tasks") or []
    if not isinstance(tasks, list):
        raise ValueError("'tasks' must be a list")
    cleaned: list[dict] = []
    for t in tasks:
        role = (t.get("role") or "").strip().lower()
        desc = (t.get("description") or "").strip()
        if role not in ("planner", "executor", "reviewer") or not desc:
            continue
        cleaned.append({"role": role, "description": desc})
    return cleaned
