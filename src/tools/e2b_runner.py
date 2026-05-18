"""e2b sandbox runner for the executor agent.

각 ExecutorAgent.run() 호출마다 새 샌드박스를 띄워 격리 보장.
"""
from __future__ import annotations

from typing import Any, Optional


class E2BRunResult:
    def __init__(
        self,
        stdout: str,
        stderr: str,
        error: Optional[str],
        results: list[str],
    ):
        self.stdout = stdout
        self.stderr = stderr
        self.error = error
        self.results = results

    def format_feedback(self, max_chars: int = 4000) -> str:
        parts: list[str] = []
        if self.error:
            parts.append(f"<error>\n{self.error}\n</error>")
        if self.stdout:
            parts.append(f"<stdout>\n{self.stdout}\n</stdout>")
        if self.stderr:
            parts.append(f"<stderr>\n{self.stderr}\n</stderr>")
        if self.results:
            joined = "\n".join(self.results)
            parts.append(f"<result>\n{joined}\n</result>")
        text = "\n".join(parts) if parts else "<stdout>(no output)</stdout>"
        return text[:max_chars]


class E2BRunner:
    """e2b AsyncSandbox 래퍼. async context manager로 사용."""

    def __init__(self, api_key: str, timeout: int = 300):
        self.api_key = api_key
        self.timeout = timeout
        self._sandbox: Any = None

    async def __aenter__(self) -> "E2BRunner":
        from e2b_code_interpreter import AsyncSandbox

        self._sandbox = await AsyncSandbox.create(
            api_key=self.api_key,
            timeout=self.timeout,
        )
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        if self._sandbox is not None:
            try:
                await self._sandbox.kill()
            finally:
                self._sandbox = None

    async def run(self, code: str) -> E2BRunResult:
        if self._sandbox is None:
            raise RuntimeError("E2BRunner used outside of `async with` block")

        execution = await self._sandbox.run_code(code)

        stdout = ""
        stderr = ""
        if getattr(execution, "logs", None):
            stdout = "\n".join(execution.logs.stdout or [])
            stderr = "\n".join(execution.logs.stderr or [])

        error = None
        if getattr(execution, "error", None):
            err = execution.error
            error = f"{getattr(err, 'name', 'Error')}: {getattr(err, 'value', err)}"

        results: list[str] = []
        for r in execution.results or []:
            text = getattr(r, "text", None)
            if text:
                results.append(text)

        return E2BRunResult(stdout=stdout, stderr=stderr, error=error, results=results)
