from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from baseai.bus import MessageBus, OutputMessage


class SendMessageInput(BaseModel):
    content: str = Field(..., description="要发送的消息内容")
    channel: str = Field("cli", description="目标频道")


@tool(args_schema=SendMessageInput)
async def send_message(
    content: str,
    config: RunnableConfig,
    channel: str = "cli",
) -> str:
    """主动向频道发送消息"""
    bus: MessageBus = config.get("configurable").get("bus")
    target_channel = channel or config.get("configurable").get("channel", "cli")

    try:
        await bus.publish_output(
            OutputMessage(
                content=content,
                channel=target_channel,
            )
        )
        return f"消息已发送至 {target_channel}"
    except Exception:
        return "消息发送失败"

