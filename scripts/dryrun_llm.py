"""LLM 연결 드라이런 — Telegram 없이 GemmaClient + LeaderAgent만 검증.

실행:
    python scripts/dryrun_llm.py
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

from src.agents.leader import LeaderAgent
from src.llm.gemma_client import GemmaClient


async def main() -> int:
    load_dotenv()
    base_url = os.environ.get("GEMMA_BASE_URL", "http://localhost:11434")
    model = os.environ.get("GEMMA_MODEL", "gemma3:4b")
    api_key = os.environ.get("GEMMA_API_KEY") or None

    print(f"[1/3] GemmaClient 핑 → {base_url} model={model}")
    llm = GemmaClient(base_url=base_url, model=model, api_key=api_key, timeout=60.0)
    try:
        ping = await llm.complete(
            [{"role": "user", "content": "한 줄로 자기소개해줘."}],
            temperature=0.2,
        )
    except Exception as e:
        print(f"  ❌ LLM 호출 실패: {e}")
        return 1
    print(f"  ✅ 응답({len(ping)}자): {ping[:200]}")

    print("\n[2/3] LeaderAgent 분해")
    leader = LeaderAgent(llm)
    try:
        tasks = await leader.decompose(
            "GitHub 트렌딩에서 오늘의 파이썬 1위 저장소를 찾아 한 줄 소개와 함께 README 요약을 만들어줘.",
            knowledge=[],
        )
    except Exception as e:
        print(f"  ❌ 분해 실패: {e}")
        return 1
    print(f"  ✅ {len(tasks)}개 태스크:")
    for i, t in enumerate(tasks):
        print(f"    {i + 1}. [{t['role']}] {t['description'][:120]}")

    print("\n[3/3] 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
