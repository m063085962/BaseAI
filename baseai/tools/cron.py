from datetime import datetime
from typing import Literal
from contextvars import ContextVar

from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from baseai.cron import CronSchedule, CronService

_in_cron_context: ContextVar[bool] = ContextVar("_in_cron_context", default=False)  # 用于标记当前是否在cron任务执行的上下文中

def set_cron_context(active: bool):
    """Mark whether the tool is executing inside a cron job callback."""
    return _in_cron_context.set(active)

def reset_cron_context(token) -> None:
    """Restore previous cron context."""
    _in_cron_context.reset(token)

class CronToolInput(BaseModel):
    action: Literal["add", "list", "remove"] = Field(
        description="操作类型: add(添加定时任务), list(列出任务), remove(删除任务)"
    )
    name: str = Field(default="", description="任务名称 (add时必填)")
    message: str = Field(default="", description="任务触发时发送给你的消息，用于描述具体的任务内容 (add时必填)")
    every_seconds: int | None = Field(
        default=None, description="每隔多少秒执行一次 (与cron_expr二选一)"
    )
    cron_expr: str | None = Field(
        default=None,
        description="Cron表达式，如 '0 9 * * *' (每天9点) (与every_seconds二选一)",
    )
    tz: str | None = Field(default=None, description="时区，如 'Asia/Shanghai'")
    at: str | None = Field(
        default=None, description="一次性执行的时间，ISO格式如 '2024-01-01T09:00:00'"
    )
    job_id: str = Field(default="", description="任务ID (remove时必填)")
    delete_after_run: bool = Field(
        default=False, description="执行后是否删除 (一次性任务)"
    )


@tool(args_schema=CronToolInput)
async def cron(
    action: Literal["add", "list", "remove"],
    config: RunnableConfig,
    name: str = "",
    message: str = "",
    every_seconds: int | None = None,
    cron_expr: str | None = None,
    tz: str | None = None,
    at: str | None = None,
    job_id: str = "",
    delete_after_run: bool = False,
) -> str:
    """管理定时任务（Cron jobs）"""
    cron_service: CronService = config.get("configurable", {}).get("cron_service")
    channel = config.get("configurable", {}).get("channel")
    session = config.get("configurable", {}).get("thread_id")

    if not cron_service:
        return "Error: Cron service 未初始化"

    if action == "add":
        if _in_cron_context.get():
            return "Error: cannot schedule new jobs from within a cron job execution"
        return _handle_add(
            cron_service,
            channel,
            session,
            name,
            message,
            every_seconds,
            cron_expr,
            tz,
            at,
            delete_after_run,
        )

    if action == "list":
        return _handle_list(cron_service)

    if action == "remove":
        return _handle_remove(cron_service, job_id)

    return f"Error: 无效的操作 '{action}'"
    

def _handle_add(
    cron_service: CronService,
    channel: str,
    session: str,
    name: str,
    message: str,
    every_seconds: int | None,
    cron_expr: str | None,
    tz: str | None,
    at: str | None,
    delete_after_run: bool,
) -> str:
    """Handle add action."""
    if not name:
        name = message[:30]
    if not message:
        return "Error: 添加任务时必须指定 message"
    if not channel or not session:
        return "Error: 没有上下文信息（channel/session)"
    if tz and not cron_expr:
        return "Error: tz 只能与 cron_expr 一同使用"
    if tz:
        from zoneinfo import ZoneInfo

        try:
            ZoneInfo(tz)
        except(KeyError, Exception):
            return f"Error: 未知时区 '{tz}'"

    if every_seconds:
        schedule = CronSchedule(kind="every", every_ms=every_seconds * 1000)
    elif cron_expr:
        schedule = CronSchedule(kind="cron", expr=cron_expr, tz=tz)
    elif at:
        try:
            dt = datetime.fromisoformat(at)
        except ValueError:
            return f"Error: invalid ISO datetime format '{at}'. Expected format: YYYY-MM-DDTHH:MM:SS"
        schedule = CronSchedule(kind="at", at_ms=int(dt.timestamp() * 1000))
    else:
        return "Error: 必须指定 every_seconds, cron_expr 或 at 之一"

    job = cron_service.add_job(
        name=name,
        schedule=schedule,
        message=message,
        deliver=True,
        channel=channel,
        to=session,
        delete_after_run=delete_after_run,
    )
    return f"定时任务已添加: {job.name} (ID: {job.id})"


def _handle_list(cron_service: CronService) -> str:
    """Handle list action."""
    jobs = cron_service.list_jobs(include_disabled=True)
    if not jobs:
        return "暂无定时任务"

    lines = ["定时任务列表:"]
    for job in jobs:
        status = "启用" if job.enabled else "禁用"
        next_run = (
            datetime.fromtimestamp(job.state.next_run_at_ms / 1000).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            if job.state.next_run_at_ms
            else "N/A"
        )
        schedule_info = job.schedule.expr or job.schedule.every_ms or job.schedule.at_ms

        lines.append(f"  - {job.name} [{status}]")
        lines.append(f"    ID: {job.id}")
        lines.append(f"    下次执行: {next_run}")
        lines.append(f"    调度: {job.schedule.kind} ({schedule_info})")

    return "\n".join(lines)


def _handle_remove(cron_service: CronService, job_id: str) -> str:
    """Handle remove action."""
    if not job_id:
        return "Error: 删除任务时必须指定 job_id"

    if cron_service.remove_job(job_id):
        return f"定时任务 {job_id} 已删除"
    return f"Error: 未找到任务 {job_id}"
