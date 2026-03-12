import os
import asyncio
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

from baseai.bus import MessageBus, InputMessage
from baseai.agent import AgentServer
from baseai.config import Config


async def client():
    config = Config(Path(".agent/config.json").resolve())
    provider = config.get_config("provider")
    agent = config.get_config("agent")

    os.environ["OPENAI_API_KEY"] = provider.api_key
    os.environ["OPENAI_BASE_URL"] = provider.base_url

    mcp_servers = config.get_mcp_server_config()
    mcp_client = MultiServerMCPClient(mcp_servers)
    mcp_tools = await mcp_client.get_tools()

    workspace = Path(agent.workspace).resolve()

    bus = MessageBus()
    server = AgentServer(
        bus=bus,
        model=agent.model,
        workspace=workspace,
        mcp_tools=mcp_tools,
    )

    server_task = asyncio.create_task(server.run())
    stop_event = asyncio.Event()

    async def input_handler():
        """处理用户输入"""
        while not stop_event.is_set():
            # 异步获取用户输入
            user_input = await asyncio.to_thread(input)
            if user_input.strip().lower() == "/exit":
                stop_event.set()
                break
            print(f"[Input] {user_input}")
            await bus.publish_input(InputMessage(
                content=user_input,
                channel="cli",
                session="test"
            ))

    async def output_handler():
        """持续监听输出队列"""
        while not stop_event.is_set():
            try:
                response = await asyncio.wait_for(bus.consume_output(), timeout=0.5)
                if response.channel != "cli":
                    continue
                print(f"[Output] {response.content}")
            except asyncio.TimeoutError:
                continue

    # 并发运行两个任务
    await asyncio.gather(
        input_handler(),
        output_handler(),
        return_exceptions=True
    )

    # 清理
    server.stop()
    await server_task

if __name__ == "__main__":
    # 启动交互客户端
    asyncio.run(client())