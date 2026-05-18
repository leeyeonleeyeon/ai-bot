"""Gemma 4 HTTP client.

OpenAI-compatible /v1/chat/completions 엔드포인트를 호출한다.
e2b 샌드박스에서 Gemma 4를 vLLM/llama.cpp/Ollama 등으로 서빙하면 그 URL을 그대로 사용.
"""
from __future__ import annotations

import httpx
from typing import Optional


class GemmaClient:
    def __init__(
        self,
        base_url: str,
        model: str = "gemma-4",
        api_key: Optional[str] = None,
        timeout: float = 600.0,
        default_max_tokens: int = 2048,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.default_max_tokens = default_max_tokens

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
    ) -> str:
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        # OpenAI 호환 응답 형식
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"unexpected LLM response: {data}") from e
