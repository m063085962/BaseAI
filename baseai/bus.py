import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

@dataclass
class InputMessage:
    """Message received from a chat channel"""
    content: str # message content
    channel: str # which channel from
    session_id: str # session id for the conversation
    sender: Literal["agent", "subagent", "user"] = "user", # who the message is intended for, default to "agent"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class OutputMessage:
    """Message to send to chat channel"""
    content: str
    channel: str
    metadata: dict[str, Any] = field(default_factory=dict)


class MessageBus:
    """
    Async message bus that decouples chat channels from the agent core.

    Channels push messages to the input queue, and the agent processes
    them and pushes responses to the output queue.
    """
    
    def __init__(self):
        self.input: asyncio.Queue[InputMessage] = asyncio.Queue()
        self.output: asyncio.Queue[OutputMessage] = asyncio.Queue()

    async def publish_input(self, msg: InputMessage) -> None:
        await self.input.put(msg)

    async def consume_input(self) -> InputMessage:
        return await self.input.get()
    
    async def publish_output(self, msg: OutputMessage) -> None:
        await self.output.put(msg)

    async def consume_output(self) -> OutputMessage:
        return await self.output.get()
    
    @property
    def input_size(self) -> int:
        return self.input.qsize()
    
    @property
    def output_size(self) -> int:
        return self.output.qsize()