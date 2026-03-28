import asyncio
from pathlib import Path
from loguru import logger

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.base import BaseCheckpointSaver

from baseai.nodes import ModelNode, MemorizationNode, RunningMemory
from baseai.bus import InputMessage, OutputMessage, MessageBus
from baseai.tools import ToolResgistry, spawn_subagent, cron
from baseai.skill import SkillsManager
from baseai.cron import CronJob, CronService


class AgentServer:
    """Agent server loop"""
    def __init__(
        self,
        bus: MessageBus,
        model: str,
        provider: str = "openai",
        max_tokens: int | None = None,
        temperature: float | None = None,
        workspace: Path = Path(".agent/workspace"),
        memory_window: int = 10000,
        recursion_limit: int = 20,
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

        self.cron_service = CronService(Path(".agent/cron.json"),on_job=self._cron_callback)
        self.skills = SkillsManager(workspace / "skills")
        self.tools = ToolResgistry()
        self._register_mcp_tools()

        self._running = False
        self._agent_tasks: list[asyncio.Task] = []
        self._subagent_tasks: list[asyncio.Task] = []
        self._processing_lock = asyncio.Lock()

    def _register_mcp_tools(self) -> None:
        """register MCP tools"""
        for tool in self.mcp_tools:
            self.tools.register(tool)
        
    def _build_agent(self, checkpointer: BaseCheckpointSaver) -> StateGraph:
        """build graph"""
        class AgentState(MessagesState):
            running_summary: RunningMemory

        tools = self.tools.get_default_tools()
        tools.append(spawn_subagent)
        tools.append(cron)

        model_node = ModelNode(
            self.model.bind_tools(tools),
            skills=self.skills,
            tools=self.tools,
        )

        memory_node = MemorizationNode(
            model=self.model,
            max_tokens=self.memory_window,
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

    async def run(self) -> None:
        """Run agent server"""
        self._running = True
        logger.info(" Agent Server 启动")
        await self.cron_service.start()
        
        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_input(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if msg.content.strip().lower() == "/stop":
                await self._handle_stop(msg)
            else:
                await self._process_message(msg)


    async def _handle_stop(self, msg: InputMessage) -> None:
        """stop current session tasks"""
        agent_cancelled = sum(1 for t in self._agent_tasks if not t.done() and t.cancel())
        subagent_cancelled = sum(1 for t in self._subagent_tasks if not t.done() and t.cancel())
        for t in self._agent_tasks + self._subagent_tasks:
            try: 
                await t
            except(asyncio.CancelledError, Exception):
                pass
        content = f"已取消 {agent_cancelled} 消息任务和 {subagent_cancelled} 子代理任务"
        await self.bus.publish_output(OutputMessage(
            content=content, channel=msg.channel,
        ))

    def stop(self) -> None:
        """Stop agent server"""
        self.cron_service.stop()
        self._running = False
        logger.info(" Agent Server 停止")

    async def _process_message(self, msg: InputMessage) -> None:
        """Process incoming message"""
        cmd = msg.content.strip().lower()
        if cmd == "/help":
            await self.bus.publish_output(OutputMessage(
                content=(
                    "commands:\n"
                    "/new   - 重置会话\n"
                    "/state - 获取历史状态\n"
                    "/stop  - 停止当前会话任务\n"
                    "/exit  - 退出当前会话\n"
                    "/help  - 显示可用的命令"
                ),
                channel=msg.channel,
            ))
        elif cmd == "/new":
            async with AsyncSqliteSaver.from_conn_string(".agent/state/state.db") as saver:
                await saver.adelete_thread(msg.session_id)
            await self.bus.publish_output(OutputMessage(
                content="session cleaned",
                channel=msg.channel,
            ))
        elif cmd == "/state":
            content= ""
            async with AsyncSqliteSaver.from_conn_string(".agent/state/state.db") as saver:
                if state := await saver.aget_tuple({"configurable":{"thread_id":msg.session_id}}):
                    for m in state.checkpoint.get("channel_values").get("messages", []):
                        if tool_calls := getattr(m, "tool_calls", None):
                            content += f"{m.type}: tool_calls: {tool_calls}\n"
                        else:
                            content += f"{m.type}: {m.content}\n"
            await self.bus.publish_output(OutputMessage(
                content=content,
                channel=msg.channel,
            ))
        else:
            routing_tasks = self._subagent_tasks if msg.sender == "agent" else self._agent_tasks
            task = asyncio.create_task(self._dispatch(msg))
            task.add_done_callback(lambda t: routing_tasks.remove(t))
            routing_tasks.append(task)

    async def _dispatch(self, msg: InputMessage) -> None:
        """dispatch message to agent or subagent task with error handling"""
        task_type = "子代理" if msg.sender == "agent" else "消息"
        task_id = msg.metadata.get("task_id", "")
        try:
            if task_type == "子代理":
                response = await self._process_subagent_task(msg)
                if response and isinstance(response, InputMessage):
                    await self.bus.publish_input(response)
            else:
                response = await self._process_agent_task(msg)
                if response and isinstance(response, OutputMessage):
                    await self.bus.publish_output(response)
        except asyncio.CancelledError:
            logger.info(f"{task_type}任务{task_id}被取消")
            raise
        except Exception:
            logger.exception(f"{task_type}任务{task_id}发生错误")
            await self.bus.publish_output(OutputMessage(
                content="抱歉，我遇到了一些问题",
                channel=msg.channel,
            ))
            raise
            
    async def _process_agent_task(self, msg: InputMessage) -> OutputMessage:
        """process agent task with global lock to synchronize access to messages"""
        if not msg.content:
            return OutputMessage(
                content="消息内容不可以为空",
                channel=msg.channel,
            )
        
        async with self._processing_lock:
            if msg.sender == "subagent":
                message = ToolMessage(content=msg.content, tool_call_id=msg.metadata.get("tool_call_id", ""))
            else:
                message = HumanMessage(content=msg.content)

            config = {
                "configurable": {
                    "thread_id": msg.session_id,
                    "channel": msg.channel,
                    "bus": self.bus,
                    "cron_service": self.cron_service,
                    "workspace": self.workspace,
                    "restrict_to_workspace": self.restrict_to_workspace,},
                "recursion_limit": self.recursion_limit,
            }

            content = "没有内容返回"
            async with AsyncSqliteSaver.from_conn_string(".agent/state/state.db") as saver:
                agent = self._build_agent(saver)

                response = await agent.ainvoke({"messages": [message]}, config=config)
                if messages := response.get("messages"):
                    content = messages[-1].content

            return OutputMessage(
                content=content,
                channel=msg.channel,
            )

    async def _process_subagent_task(self, msg: InputMessage) -> InputMessage:
        """process subagent task"""
        if not msg.content:
            return InputMessage(
                content="任务内容不可以为空",
                channel=msg.channel,
                session_id=msg.session_id,
                sender="subagent",
            )
        
        tools = msg.metadata.get("tools")
        skill = msg.metadata.get("skill")

        messages = []
        if skill and (skill_content:=self.skills.get_content(skill)):
            messages.append(HumanMessage(content=skill_content))
        messages.append(HumanMessage(content=msg.content))

        config = {
                "configurable": {
                    "thread_id": msg.metadata.get("task_id", "1"),
                    "workspace": self.workspace,
                    "restrict_to_workspace": self.restrict_to_workspace,},
                "recursion_limit": self.recursion_limit,
            }
        
        agent_file = self.workspace / "AGENT.md"
        if agent_file.exists():
            instrution = agent_file.read_text(encoding="utf-8")

        system_prompt=f"{"你是一个任务处理专家。" or instrution}\n\n- 如需检索记忆，可使用read_file工具查看工作区中 MEMORY.md 文档"
        
        content = "子代理没有返回内容"
        async with AsyncSqliteSaver.from_conn_string(".agent/state/state.db") as saver:
            subagent = create_agent(
                model=self.model,
                tools=self.tools.get_with_default(tools),
                system_prompt=system_prompt,
                checkpointer=saver,
            )

            response = await subagent.ainvoke({"messages": messages}, config=config)
            if messages := response.get("messages"):
                content = messages[-1].content

        return InputMessage(
            content=content,
            channel=msg.channel,
            session_id=msg.session_id,
            sender="subagent",
        )
    
    async def _cron_callback(self, job: CronJob) -> None:
        """callback for cron job"""
        await self._process_message(InputMessage(
            content=job.payload.message,
            channel=job.payload.channel,
            session_id=job.payload.to,
            sender="cron",
        ))