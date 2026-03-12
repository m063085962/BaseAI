import asyncio
from pathlib import Path
from loguru import logger

from langchain.chat_models import init_chat_model
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.base import BaseCheckpointSaver

from baseai.nodes import ModelNode, MemorizationNode, RunningMemory
from baseai.bus import InputMessage, OutputMessage, MessageBus
from baseai.tools.filesystem import WORKSPACE_DIR
from baseai.tool import ToolResgistry
from baseai.skill import SkillsLoader


class AgentServer:
    """Agent server loop"""
    def __init__(
        self,
        bus: MessageBus,
        model: str,
        provider: str = "openai",
        max_tokens: int | None = None,
        temperature: float | None = None,
        workspace: Path = WORKSPACE_DIR,
        memory_window: int = 40000,
        recursion_limit: int = 30,
        restrict_to_workspace: bool = True,
        mcp_tools: list[BaseTool] = [],
    ):
        self.bus = bus
        self.workspace = workspace
        self.memory_window = memory_window
        self.recursion_limit = recursion_limit
        self.restrict_to_workspace = restrict_to_workspace
        self.mcp_tools = mcp_tools
        self.model = init_chat_model(
            model=model,
            model_provider=provider,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        self.skills = SkillsLoader(workspace / "skills")
        self.tools = ToolResgistry()
        self._register_mcp_tools()

        self._running = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}
        self._processing_lock = asyncio.Lock()

    def _register_mcp_tools(self) -> None:
        """register MCP tools"""
        for tool in self.mcp_tools:
            self.tools.register(tool)
        
    def _build_agent(self, checkpointer: BaseCheckpointSaver) -> StateGraph:
        """build graph"""
        class AgentState(MessagesState):
            running_summary: RunningMemory

        memory_node = MemorizationNode(
            model=self.model,
            max_tokens=self.memory_window,
        )

        tools = self.tools.get_default_tools()
        tools.append(self._warp_subagent_tool())

        model_node = ModelNode(
            self.model.bind_tools(tools),
            skills=self.skills,
            tools=self.tools,
        )

        graph = (
            StateGraph(AgentState)
            .add_node("memory", memory_node)
            .add_node("model", model_node)
            .add_node("tools", ToolNode(tools))

            .add_edge(START, "memory")
            .add_edge("memory", "model")
            .add_conditional_edges("model", tools_condition, ["tools", END])
            .add_edge("tools", "model")

            .compile(checkpointer=checkpointer)
        )
        
        return graph
    
    def _warp_subagent_tool(self) -> BaseTool:
        """将子代理调度功能包装为agent可调用的工具"""
        from langchain_core.tools import tool
        from langchain_core.runnables import RunnableConfig
        from pydantic import BaseModel, Field

        class SpawnSubagentSchema(BaseModel):
            task_name: str = Field(description="为任务取一个简介名称")
            task_content: str = Field(description="任务的具体内容，包括需求分析、规划和指令等详细任务指导内容")
            skill: str  = Field(description="技能名称，只能从Available Skills中挑选, 没有则为空字符")
            tools: list[str] = Field(description="工具名称列表，只能从Available Tools中挑选，没有则为空列表")

        @tool(args_schema=SpawnSubagentSchema)
        async def spawn_subagent(
            task_name: str,
            task_content: str,
            skill: str,
            tools: list[str],
            config: RunnableConfig,
        ) -> str:
            """创建子代理任务"""
            session = config.get("configurable").get("thread_id")
            channel = config.get("configurable").get("channel")

            try:
                task = asyncio.create_task(self._run_subagent(
                    task_content=task_content,
                    skill=skill,
                    tools=tools,
                    channel=channel,
                    session=session,
                ))
                self._active_tasks.setdefault(session, []).append(task)
                task.add_done_callback(lambda t, k=session: t in self._active_tasks.get(k, []) and self._active_tasks[k].remove(t))
            except Exception:
                return f"子代理任务[{task_name}]创建失败"
                
            return f"子代理任务[{task_name}]创建成功"
        
        return spawn_subagent
        
    async def _run_subagent(
        self,
        task_content: str,
        skill: str,
        tools: list[str],
        channel: str,
        session: str,
    ) -> None:
        """执行子代理"""
        from langchain.agents import create_agent

        subagent = create_agent(
            model=self.model,
            tools=self.tools.get_with_default(tools),
            system_prompt="你是一个任务处理专家。"
        )

        messages = []
        if skill and (skill_content:=self.skills.get_content(skill)):
            messages.append(("human", skill_content))
        messages.append(("human", task_content))

        try:
            response = await subagent.ainvoke(
                {"messages": messages},
                config={"configurable": {"thread_id": None}},
            )

            if messages := response.get("messages"):
                await self.bus.publish_output(
                    OutputMessage(
                        content=messages[-1].content,
                        channel=channel,
                    )
                )
        except asyncio.CancelledError:
            logger.info(f"会话{session}的子代理任务被取消")
            raise
        except Exception:
            logger.exception(f"会话{session}的子代理任务发生错误")
            await self.bus.publish_output(OutputMessage(
                    channel=channel,
                    content="抱歉，子代理遇到了一些问题",
            ))

    async def run(self) -> None:
        """Run agent server"""
        self._running = True
        logger.info(" Agent Server 启动")
        
        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_input(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if msg.content.strip().lower() == "/stop":
                await self._handle_stop(msg)
            else:
                task = asyncio.create_task(self._dispatch(msg))
                self._active_tasks.setdefault(msg.session, []).append(task)
                task.add_done_callback(lambda t, k=msg.session: t in self._active_tasks.get(k, []) and self._active_tasks[k].remove(t))

    async def _handle_stop(self, msg: InputMessage) -> None:
        """取消所有任务和子代理"""
        tasks = self._active_tasks.pop(msg.session, [])
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        for t in tasks:
            try: 
                await t
            except(asyncio.CancelledError, Exception):
                pass
        content = f"已停止 {cancelled} 任务" if cancelled else "没有正在运行的任务"
        await self.bus.publish_output(OutputMessage(
            channel=msg.channel, content=content,
        ))

    def stop(self) -> None:
        """Stop agent server"""
        self._running = False
        logger.info(" Agent Server 停止")

    async def _dispatch(self, msg: InputMessage) -> None:
        """使用全局锁同步处理消息"""
        async with self._processing_lock:
            try:
                response = await self._process_message(msg)
                if response and isinstance(response, OutputMessage):
                    await self.bus.publish_output(response)
            except asyncio.CancelledError:
                logger.info(f"会话{msg.session}的消息任务被取消")
                raise
            except Exception:
                logger.exception(f"会话{msg.session}的消息任务发生错误")
                await self.bus.publish_output(OutputMessage(
                    channel=msg.channel,
                    content="抱歉，我遇到了一些问题",
                )) 
    
    async def _process_message(self, msg: InputMessage) -> OutputMessage:
        """处理消息"""
        config = {
            "configurable": {
                "thread_id": msg.session,
                "channel": msg.channel,
                "workspace": self.workspace,
                "restrict_to_workspace": self.restrict_to_workspace,},
            "recursion_limit": self.recursion_limit,
        }
        
        async with AsyncSqliteSaver.from_conn_string(".agent/state/state.db") as saver:
            agent = self._build_agent(saver)
            cmd = msg.content.strip().lower()
            if cmd == "/help":
                return OutputMessage(
                    content=(
                        "commands:\n"
                        "/new - 重置会话\n"
                        "/state - 获取历史状态\n"
                        "/stop - 停止当前会话任务\n"
                        "/exit - 退出当前会话\n"
                        "/help - 显示可用的命令"
                    ),
                    channel=msg.channel,
                )
            elif cmd == "/state": # 获取历史状态
                content= ""
                if state := await agent.aget_state(config):
                    for m in state.values.get("messages", []):
                        if tool_calls := getattr(m, "tool_calls", None):
                            content += f"{m.type}: tool_calls: {tool_calls}\n"
                        else:
                            content += f"{m.type}: {m.content}\n"
                return OutputMessage(
                    content=content,
                    channel=msg.channel,
                )
            elif cmd == "/new":
                await saver.adelete_thread(msg.session)
                return OutputMessage(
                    content="Succeed",
                    channel=msg.channel,
                )

            message = {"messages": ("human", msg.content)}
            response = await agent.ainvoke(message, config=config)
            
        return OutputMessage(
            content=response["messages"][-1].content,
            channel=msg.channel,
        )
    
    async def process_direct(
        self,
        content: str,
        session: str,
        channel: str = "cli",
        sender_id = "direct"
    ) -> str:
        """直接处理消息"""
        response = await self._process_message(
            InputMessage(
                content=content,
                session=session,
                channel=channel,
                sender_id=sender_id,
            )
        )

        return response.content if response else ""
