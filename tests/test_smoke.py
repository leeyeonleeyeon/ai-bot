"""LLM/Telegram/e2b/GitHub 외부 호출 없이 핵심 로직만 검증한다."""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.executor import _extract_run  # noqa: E402
from src.agents.leader import _parse_tasks  # noqa: E402
from src.orchestrator import _build_recent_goal_context, _extract_verdict, _is_trivial_input  # noqa: E402
from src.storage.github import GitHubCommitter  # noqa: E402
from src.task_queue.queue import TaskQueue, TaskStatus  # noqa: E402


def test_parse_tasks_strict() -> None:
    raw = (
        '{"tasks": [{"role": "planner", "description": "p"}, '
        '{"role": "executor", "description": "e"}]}'
    )
    tasks = _parse_tasks(raw)
    assert len(tasks) == 2
    assert tasks[0]["role"] == "planner"


def test_parse_tasks_with_fences() -> None:
    raw = '```json\n{"tasks": [{"role": "reviewer", "description": "r"}]}\n```'
    tasks = _parse_tasks(raw)
    assert tasks[0]["role"] == "reviewer"


def test_parse_tasks_drops_invalid_role() -> None:
    raw = '{"tasks": [{"role": "wizard", "description": "x"}, {"role": "executor", "description": "e"}]}'
    tasks = _parse_tasks(raw)
    assert [t["role"] for t in tasks] == ["executor"]


def test_queue_roundtrip() -> None:
    async def run() -> None:
        with tempfile.TemporaryDirectory() as d:
            q = TaskQueue(os.path.join(d, "t.db"))
            await q.init()
            gid = await q.create_goal(123, "demo goal")
            tid = await q.add_task(gid, "planner", "step 1", sequence=0)
            nxt = await q.next_pending_task(gid)
            assert nxt is not None and nxt["id"] == tid
            await q.update_task(tid, TaskStatus.DONE, "ok")
            assert await q.next_pending_task(gid) is None

            await q.add_knowledge(123, "inline", "fact A")
            ks = await q.list_knowledge(123)
            assert len(ks) == 1 and ks[0]["content"] == "fact A"

    asyncio.run(run())


def test_latest_completed_goal() -> None:
    async def run() -> None:
        with tempfile.TemporaryDirectory() as d:
            q = TaskQueue(os.path.join(d, "t.db"))
            await q.init()
            done = await q.create_goal(123, "first goal")
            await q.update_goal(done, TaskStatus.DONE, "first result")
            pending = await q.create_goal(123, "second goal")
            await q.update_goal(pending, TaskStatus.IN_PROGRESS)

            latest = await q.latest_completed_goal(123)
            assert latest is not None
            assert latest["id"] == done
            assert latest["result"] == "first result"

    asyncio.run(run())


def test_extract_run_tag() -> None:
    text = "preface\n<run_python>\nprint(1+1)\n</run_python>\nepilogue"
    assert _extract_run(text) == "print(1+1)"


def test_extract_run_none() -> None:
    assert _extract_run("just a final answer, no code") is None


def test_github_committer_validates_repo() -> None:
    try:
        GitHubCommitter(token="x", repo="bad")
    except ValueError:
        return
    raise AssertionError("expected ValueError for bad repo")


def test_extract_verdict_accept() -> None:
    txt = "본문 비평...\n\n[VERDICT: ACCEPT]"
    assert _extract_verdict(txt) == "accept"


def test_extract_verdict_revise_case_insensitive() -> None:
    assert _extract_verdict("...[verdict: revise]") == "revise"


def test_extract_verdict_missing() -> None:
    assert _extract_verdict("그냥 비평만 있고 토큰 없음") is None


def test_is_trivial_short_greeting() -> None:
    assert _is_trivial_input("안녕")
    assert _is_trivial_input("hi")
    assert _is_trivial_input("고마워!")
    assert _is_trivial_input("오늘 점심 뭐 먹지?")


def test_is_trivial_short_work_request_false() -> None:
    assert not _is_trivial_input("파이썬으로 할 일 목록 앱 설계를 간단히 작성해줘")
    assert not _is_trivial_input("README 요약해줘")


def test_is_trivial_followup_false() -> None:
    assert not _is_trivial_input("그거 더 짧게")
    assert not _is_trivial_input("더 자세히")
    assert not _is_trivial_input("표로 정리")


def test_is_trivial_long_text_false() -> None:
    long = "이 함수의 버그를 찾아서 자세히 수정해주세요 그리고 테스트도 추가해줘"
    assert not _is_trivial_input(long)


def test_is_trivial_multiline_false() -> None:
    assert not _is_trivial_input("안녕\n오늘 뭐 함?")


def test_is_trivial_code_false() -> None:
    assert not _is_trivial_input("```py\nprint(1)\n```")


def test_is_trivial_url_false() -> None:
    assert not _is_trivial_input("https://x.com 봐줘")


def test_is_trivial_empty_false() -> None:
    assert not _is_trivial_input("")
    assert not _is_trivial_input("   ")


def test_queue_revise_helpers() -> None:
    async def run() -> None:
        with tempfile.TemporaryDirectory() as d:
            q = TaskQueue(os.path.join(d, "t.db"))
            await q.init()
            gid = await q.create_goal(1, "g")
            t1 = await q.add_task(gid, "executor", "first", sequence=0)
            await q.add_task(gid, "reviewer", "rev", sequence=1)
            assert await q.find_last_done_task(gid, "executor") is None
            await q.update_task(t1, TaskStatus.DONE, "out1")
            prev = await q.find_last_done_task(gid, "executor")
            assert prev is not None and prev["output"] == "out1"
            assert await q.next_sequence(gid) == 2

    asyncio.run(run())


def test_build_recent_goal_context() -> None:
    ctx = _build_recent_goal_context(
        {
            "description": "TODO 앱 설계",
            "result": "화면: 목록\n기능: 추가/완료",
        }
    )
    assert ctx is not None
    assert "[PREVIOUS_GOAL]\nTODO 앱 설계" in ctx
    assert "[PREVIOUS_RESULT]\n화면: 목록" in ctx


if __name__ == "__main__":
    test_parse_tasks_strict()
    test_parse_tasks_with_fences()
    test_parse_tasks_drops_invalid_role()
    test_queue_roundtrip()
    test_latest_completed_goal()
    test_extract_run_tag()
    test_extract_run_none()
    test_github_committer_validates_repo()
    test_extract_verdict_accept()
    test_extract_verdict_revise_case_insensitive()
    test_extract_verdict_missing()
    test_is_trivial_short_greeting()
    test_is_trivial_short_work_request_false()
    test_is_trivial_followup_false()
    test_is_trivial_long_text_false()
    test_is_trivial_multiline_false()
    test_is_trivial_code_false()
    test_is_trivial_url_false()
    test_is_trivial_empty_false()
    test_queue_revise_helpers()
    test_build_recent_goal_context()
    print("smoke ok")
