from __future__ import annotations

import logging

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.knowledge.brain_pack import fetch_url, is_url
from src.orchestrator import Orchestrator
from src.task_queue.queue import TaskQueue

log = logging.getLogger(__name__)

_STATUS_ICON = {
    "pending": "⏳",
    "in_progress": "🔄",
    "done": "✅",
    "failed": "❌",
}


def _chunk(text: str, size: int = 3500) -> list[str]:
    if not text:
        return ["(빈 결과)"]
    return [text[i : i + size] for i in range(0, len(text), size)]


class TelegramBot:
    def __init__(self, token: str, orchestrator: Orchestrator, queue: TaskQueue):
        self.app = Application.builder().token(token).build()
        self.orchestrator = orchestrator
        self.queue = queue
        self._register()

    def _register(self) -> None:
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("goal", self.cmd_goal))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("result", self.cmd_result))
        self.app.add_handler(CommandHandler("inject", self.cmd_inject))
        self.app.add_handler(CommandHandler("reset", self.cmd_reset))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.cmd_help))

    async def cmd_start(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(
            "🤖 멀티에이전트 팀 봇 (Gemma 4 / e2b)\n\n"
            "/goal <목표>     팀에게 업무 지시\n"
            "/status          진행 상태 조회\n"
            "/result          최근 결과 보기\n"
            "/inject <URL|텍스트>  지식 주입(브레인 팩)\n"
            "/reset           진행 중 작업 취소"
        )

    async def cmd_goal(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        goal = " ".join(ctx.args).strip()
        if not goal:
            await update.message.reply_text("사용법: /goal <목표>")
            return
        chat_id = update.effective_chat.id
        await update.message.reply_text(f"🎯 목표 접수\n{goal}")

        async def progress(msg: str) -> None:
            try:
                await update.message.reply_text(msg)
            except Exception as e:
                log.warning("progress notify failed: %s", e)

        try:
            res = await self.orchestrator.run_goal(chat_id, goal, on_update=progress)
        except Exception as e:
            log.exception("run_goal failed")
            await update.message.reply_text(f"❌ 실행 실패: {e}")
            return

        result = res.get("result") or "(빈 결과)"
        await update.message.reply_text("📦 결과")
        for chunk in _chunk(result):
            await update.message.reply_text(chunk)

    async def cmd_status(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        latest = await self.queue.latest_goal(update.effective_chat.id)
        if not latest:
            await update.message.reply_text("진행 중인 목표가 없습니다.")
            return
        snap = await self.queue.goal_status(latest["id"])
        lines = [
            f"🎯 {snap['goal']['description']}",
            f"상태: {snap['goal']['status']}",
            "",
        ]
        for t in snap["tasks"]:
            mark = _STATUS_ICON.get(t["status"], "?")
            lines.append(f"{mark} [{t['assigned_role']}] {t['description']}")
        await update.message.reply_text("\n".join(lines))

    async def cmd_result(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        latest = await self.queue.latest_goal(update.effective_chat.id)
        if not latest:
            await update.message.reply_text("결과가 없습니다.")
            return
        result = latest.get("result") or "(아직 결과 없음)"
        for chunk in _chunk(result):
            await update.message.reply_text(chunk)

    async def cmd_inject(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        payload = " ".join(ctx.args).strip()
        if not payload:
            await update.message.reply_text("사용법: /inject <URL 또는 텍스트>")
            return
        chat_id = update.effective_chat.id
        if is_url(payload):
            try:
                content = await fetch_url(payload)
            except Exception as e:
                await update.message.reply_text(f"URL 가져오기 실패: {e}")
                return
            source = payload
        else:
            content = payload
            source = "inline"
        await self.queue.add_knowledge(chat_id, source, content)
        await update.message.reply_text(
            f"📚 지식 주입 완료 ({len(content)}자, 출처: {source[:60]})"
        )

    async def cmd_reset(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        n = await self.queue.reset_chat(update.effective_chat.id)
        await update.message.reply_text(f"🔄 진행 중 작업 {n}건을 초기화했습니다.")

    async def cmd_help(self, update: Update, _ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text("/start 로 명령어를 확인하세요.")

    def run(self) -> None:
        self.app.run_polling()
