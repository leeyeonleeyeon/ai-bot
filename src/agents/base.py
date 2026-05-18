from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from src.llm.gemma_client import GemmaClient


@dataclass
class AgentContext:
    task_id: str
    goal: str
    knowledge: list[str] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)


class BaseAgent(ABC):
    role: str = "base"
    system_prompt: str = ""

    def __init__(self, llm: GemmaClient, name: Optional[str] = None):
        self.llm = llm
        self.name = name or self.role

    async def _think(self, ctx: AgentContext, user_msg: str, temperature: float = 0.3) -> str:
        messages: list[dict] = [{"role": "system", "content": self.system_prompt}]
        for k in ctx.knowledge:
            messages.append({"role": "system", "content": f"[KNOWLEDGE]\n{k}"})
        messages.extend(ctx.history)
        messages.append({"role": "user", "content": user_msg})
        return await self.llm.complete(messages, temperature=temperature)

    @abstractmethod
    async def run(self, ctx: AgentContext) -> dict: ...
