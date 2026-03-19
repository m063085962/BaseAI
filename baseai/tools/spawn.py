import uuid
from typing import Annotated

from langchain.tools import tool, BaseTool, InjectedToolCallId, ToolRuntime
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from baseai.bus import MessageBus, InputMessage

class SpawnToolInput(BaseModel):
    """Input for SpawnTool"""
    task: str = Field(..., description="The task for the subagent to perform")
    skill: str | None = Field(None, description="The skill for the subagent to use, must be selected from the `Available Skills`")
    tools: list[str] | str | None = Field(None, description="The tools for the subagent to use, must be selected from the `Available Tools`")  

@tool(args_schema=SpawnToolInput)
async def spawn_subagent(
    task: str,
    skill: str | None,
    tools: list[str] | str | None,
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> str:
    """Tool for spawning a subagent"""
    if isinstance(tools, str):
            tools = eval(tools)

    bus: MessageBus = config.get("configurable").get("bus")
    try:
        await bus.publish_input(InputMessage(
            content=task,
            channel=config.get("configurable").get("channel", "cli"),
            session_id=config.get("configurable").get("thread_id", ""),
            sender="agent",
            metadata={
                "task_id": str(uuid.uuid4()),
                "skill": skill,
                "tools": tools,
                "tool_call_id": tool_call_id,
            }
        ))
    except Exception:
        return "子代理任务发布失败"
    
    return "子代理任务已发布，任务执行中..."