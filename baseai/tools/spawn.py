import uuid

from langchain.tools import BaseTool, ToolRuntime, tool
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from baseai.bus import MessageBus, InputMessage

class SpawnToolInput(BaseModel):
    """Input for SpawnTool"""
    task: str = Field(..., description="The task for the subagent to perform")
    skill: str | None = Field(None, description="The skill for the subagent to use, must be selected from the `Available Skills`")
    tools: list[str] | str | None = Field(None, description="The tools for the subagent to use, must be selected from the `Available Tools`")  

class SpawnTool(BaseTool):
    """Tool for spawning a subagent"""

    name: str = "spawn_subagent"
    description: str = "Spawn a subagent with the given task"
    args_schema: type[BaseModel] = SpawnToolInput

    def __init__(self, bus: MessageBus, **kwargs):
        super().__init__(**kwargs)
        self._bus = bus
    
    def _run(self, task: str, skill: str | None, tools: list[str] | None,) -> str:
        """Run the tool"""
        raise NotImplementedError("SpawnTool only supports async execution")

    async def _arun(
        self, 
        task: str,
        skill: str | None,
        tools: list[str] | None,
        runtime: ToolRuntime,
    ) -> str:
        """Run the tool"""
        if isinstance(tools, str):
            tools = eval(tools)

        try:
            await self._bus.publish_input(InputMessage(
                content=task,
                channel=runtime.config.get("configurable").get("channel", "cli"),
                session_id=runtime.config.get("configurable").get("thread_id"),
                sender="agent",
                metadata={
                    "task_id": str(uuid.uuid4()),
                    "skill": skill,
                    "tools": tools,
                    "tool_call_id": runtime.tool_call_id,
                }
            ))
        except Exception:
            return "子代理任务发布失败"
        
        return "子代理任务已发布"
    

@tool(args_schema=SpawnToolInput)
async def spawn_subagent(
    task: str,
    skill: str | None,
    tools: list[str] | str | None,
    config: RunnableConfig,
    runtime: ToolRuntime,
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
                "tool_call_id": runtime.tool_call_id,
            }
        ))
    except Exception:
        return "子代理任务发布失败"
    
    return "子代理任务已发布，任务执行中..."