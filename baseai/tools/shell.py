import asyncio
import os
import re
import locale
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, Field

DENY_PATTERNS = [
    r"\brm\s+-[rf]{1,2}\b",          # rm -r, rm -rf, rm -fr
    r"\bdel\s+/[fq]\b",              # del /f, del /q
    r"\brmdir\s+/s\b",               # rmdir /s
    r"(?:^|[;&|]\s*)format\b",       # format (as standalone command only)
    r"\b(mkfs|diskpart)\b",          # disk operations
    r"\bdd\s+if=",                   # dd
    r">\s*/dev/sd",                  # write to disk
    r"\b(shutdown|reboot|poweroff)\b",  # system power
    r":\(\)\s*\{.*\};\s*:",          # fork bomb
]


def _check(command: str) -> str | None:
    """Check if command is safe."""
    cmd_lower = command.strip().lower()

    for pattern in DENY_PATTERNS:
        if re.search(pattern, cmd_lower):
            return f"Blocked: deny pattern '{pattern}'"
        
    return None


def _handle_config(config: RunnableConfig) -> Path | None:
    """Get workspace path from config"""
    configurable = config.get("configurable", {})
    workspace = configurable.get("workspace")
    return workspace


class RunShellSchema(BaseModel):
    command: str = Field(description="要执行的 shell 命令")

@tool(args_schema=RunShellSchema)
async def run_shell(command: str, config: RunnableConfig) -> str:
    """Execute a shell command and return its output."""
    workspace = _handle_config(config)
    if not workspace:
        return "未设置工作区目录"
    cwd = workspace.as_posix()

    if check_error := _check(command):
        return check_error

    timeout = 30
    try:
        env = os.environ.copy()

        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
            return f"Error: Command timed out after {timeout} seconds"

        output = []
        encoding = locale.getpreferredencoding(False)
        if stdout:
            output.append(stdout.decode(encoding, errors="replace"))
        if stderr:
            output.append(
                f"STDERR: {stderr.decode(encoding, errors='replace')}"
            )

        if process.returncode != 0:
            output.append(f"[exit code: {process.returncode}]")
        
        result = "\n".join(output) if output else "Command executed with no output"

        max_output = 10000
        if len(result) > max_output:
            half = max_output // 2
            result = (
                result[:half]
                + f"\n\n... ({len(result) - max_output:,} chars truncated) ...\n\n"
                + result[-half:]
            )

        return result

    except PermissionError:
        return "Error: Permission denied"
    except Exception as e:
        return f"Error executing command: {e}"
