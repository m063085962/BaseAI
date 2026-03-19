from dataclasses import dataclass
from typing import Callable, Iterable, Any
from pathlib import Path
from loguru import logger

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    MessageLikeRepresentation,
    ToolMessage,
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.utils.runnable import RunnableCallable
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

TokenCounter = Callable[[Iterable[MessageLikeRepresentation]], int]

@dataclass
class RunningMemory:
    """
    存储上次记忆信息的数据对象。
    用于后续调用memorize_messages时，避免重复记忆相同消息。
    """
    summary: str
    """最新的摘要"""

    index: int
    """消息窗口的边界位置"""

    memorized_message_ids: set[str]
    """已记忆的所有消息的ID"""

    last_memorized_message_id: str | None
    """已记忆的最后一条消息的ID"""


@dataclass
class PreprocessedMessages:
    """存储待记忆消息及其相关信息的数据对象。"""

    messages_to_memorize: list[AnyMessage]
    """待记忆的消息列表。"""

    n_tokens_to_memorize: int
    """待记忆消息的token数量。"""

    n_messages_memorized: int
    """已记忆的消息总数。"""



def _preprocess_messages(
    *,
    messages: list[AnyMessage],
    running_memory: RunningMemory | None,
    max_tokens: int,
    keep_ratio: float,
    token_counter: TokenCounter,
) -> PreprocessedMessages:
    """预处理消息列表，判断是否达到触发记忆的条件"""
    if  keep_ratio < 0.0 or keep_ratio > 1.0:
        raise ValueError("keep_ratio 必须设置在 0~1 区间")
    keep_tokens = int(max_tokens * keep_ratio)
    
    if not messages:
        return PreprocessedMessages(
            messages_to_memorize=[],
            n_tokens_to_memorize=0,
            n_messages_memorized=0,
        )

    memorized_message_ids = set()
    n_messages_memorized = 0
    if running_memory:
        memorized_message_ids = running_memory.memorized_message_ids
        # 利用last_memorized_messages_id统计已记忆消息的数量
        for i, message in enumerate(messages):
                if message.id == running_memory.last_memorized_message_id:
                    n_messages_memorized = i + 1
                    break

    # 计算未记忆消息的tokens
    total_n_tokens = token_counter(messages[n_messages_memorized:])
    if total_n_tokens < max_tokens:
        return PreprocessedMessages(
            messages_to_memorize=[],
            n_tokens_to_memorize=0,
            n_messages_memorized=0,
        )

    # 遍历消息检查待记忆消息的合规性
    # 并计算tokens根据keep_tokens找到截断点
    n_tokens_to_keep = 0
    idx = len(messages)
    # 将工具调用ID映射至对应的工具消息
    tool_call_id_to_tool_message: dict[str, ToolMessage] = {}
    is_idx_located = False
    # 从最新消息往前遍历
    for i in range(len(messages)-1, n_messages_memorized-1, -1):
        message = messages[i]
        if message.id is None:
            raise ValueError("消息必须包含ID字段")

        if message.id in memorized_message_ids:
            raise ValueError(
                f"ID为{message.id}的消息已被记忆过"
            )

        # 检查累计消息是否已达到keep_tokens
        if keep_tokens and not is_idx_located:
            # 根据工具调用ID存储工具消息
            if isinstance(message, ToolMessage) and message.tool_call_id:
                tool_call_id_to_tool_message[message.tool_call_id] = message

            if n_tokens_to_keep + token_counter([message]) > keep_tokens:
                idx = i
                is_idx_located = True
                continue

            n_tokens_to_keep += token_counter([message])

    # 计算待记忆区间tokens
    n_tokens_to_memorize = total_n_tokens - n_tokens_to_keep
    messages_to_memorize = messages[n_messages_memorized : idx+1]

    # 如果最后一条消息是包含工具调用的AI消息，
    # 在待记忆消息中包含后续对应的工具消息，
    # 以避免与LLM提供商产生问题
    if(
        messages_to_memorize
        and isinstance(messages_to_memorize[-1], AIMessage)
        and (tool_calls := messages_to_memorize[-1].tool_calls)
    ):
        for tool_call in tool_calls:
            if tool_call["id"] in tool_call_id_to_tool_message:
                tool_message = tool_call_id_to_tool_message[tool_call["id"]]
                n_tokens_to_memorize += token_counter([tool_message])
                messages_to_memorize.append(tool_message)

    return PreprocessedMessages(
        messages_to_memorize=messages_to_memorize,
        n_tokens_to_memorize=n_tokens_to_memorize,
        n_messages_memorized=n_messages_memorized + len(messages_to_memorize)
    )


def _prepare_result(
    result: AIMessage,
    memory_file: Path,
    running_memory: RunningMemory | None,
    messages_to_memorize: list[AnyMessage],
    n_messages_memorized: int,
) -> RunningMemory:
    """处理记忆的结果，并返回running_memory更新状态"""
    memory = ""
    summary = ""
    if tool_calls := result.tool_calls:
        args = tool_calls[0].get("args")
        summary = args.get("summary")
        memory = args.get("memories")


    if memory:
        memory_file.write_text(memory, encoding="utf-8")
        logger.info(f"MEMORY.md已更新")

    memorized_message_ids = (
        set(running_memory.memorized_message_ids) if running_memory else set() | 
        set(message.id for message in messages_to_memorize)
    )

    return RunningMemory(
        summary=summary,
        index=n_messages_memorized,
        memorized_message_ids=memorized_message_ids,
        last_memorized_message_id=messages_to_memorize[-1].id,
    )


MEMORY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个记忆专家，负责更新对话摘要和长期记忆。请调用 `extract_memory` 工具输出结果。"
        ),
        ("placeholder", "{messages}"),
        (
            "user",
            "根据当前对话，更新已有的摘要和记忆。\n"
            "1. **摘要**：将对话核心内容压缩为1-3句话，包含主题、决策、结论、行动项等。\n"
            "2. **记忆**：以Markdown格式更新长期记忆文件，包含以下四个部分：\n"
            "   - ## User Information（用户信息）\n"
            "   - ## Preferences（偏好）\n"
            "   - ## Project Context（项目上下文）\n"
            "   - ## Important Notes（重要笔记）\n"
            "   基于已有的记忆文件内容进行更新，添加、修改或删除相关信息。\n"
            "\n已有摘要：{existing_summary}\n"
            "已有记忆：{existing_memory}"
        ),
    ]
)

class ExtractMemorySchema(BaseModel):
    summary: str = Field(description="会话内容的摘要")
    memory: str = Field(description="会话内容中提取的记忆")

@tool(args_schema=ExtractMemorySchema)
def extract_memory(summary: str, memories: str) -> None:
    """从对话中提取记忆信息"""

def memorize_messages(
    messages: list[AnyMessage],
    *,
    workspace: Path | None,
    running_memory: RunningMemory | None,
    model: BaseChatModel,
    max_tokens: int = 4096,
    keep_ratio: float = 0.3,
    token_counter: TokenCounter = count_tokens_approximately,
    memory_prompt: ChatPromptTemplate = MEMORY_PROMPT,
) -> RunningMemory:
    """Condense the context while extracting long-term memory from the session."""
    if not workspace:
        raise ValueError("workspace路径未提供")
    
    preprocessed_messages = _preprocess_messages(
        messages=messages,
        running_memory=running_memory,
        max_tokens=max_tokens,
        keep_ratio=keep_ratio,
        token_counter=token_counter,
    )

    if messages_to_memorize := preprocessed_messages.messages_to_memorize:
        model = model.bind_tools([extract_memory], tool_choice="extract_memory")

        memory_file = workspace / "MEMORY.md"
        existing_memory = ""
        if memory_file.exists():
            existing_memory = memory_file.read_text(encoding="utf-8")

        prompt = memory_prompt.invoke(
            {
                "messages": messages_to_memorize,
                "existing_summary": running_memory.summary if running_memory else "",
                "existing_memory": existing_memory,
            }
        )

        result = model.invoke(prompt)

        return _prepare_result(
            result=result,
            memory_file=memory_file,
            running_memory=running_memory,
            messages_to_memorize=messages_to_memorize,
            n_messages_memorized=preprocessed_messages.n_messages_memorized
        )

    return running_memory


async def amemorize_messages(
    messages: list[AnyMessage],
    *,
    workspace: Path | None,
    running_memory: RunningMemory | None,
    model: BaseChatModel,
    max_tokens: int = 4096,
    keep_ratio: float = 0.3,
    token_counter: TokenCounter = count_tokens_approximately,
    memory_prompt: ChatPromptTemplate = MEMORY_PROMPT,
) -> RunningMemory:
    if not workspace:
        raise ValueError("workspace路径未提供")

    preprocessed_messages = _preprocess_messages(
        messages=messages,
        running_memory=running_memory,
        max_tokens=max_tokens,
        keep_ratio=keep_ratio,
        token_counter=token_counter,
    )

    if messages_to_memorize := preprocessed_messages.messages_to_memorize:
        model.bind_tools([extract_memory], tool_choice="extract_memory")

        memory_file = workspace / "memory" / "MEMORY.md"
        existing_memory = ""
        if memory_file.exists():
            existing_memory = memory_file.read_text(encoding="utf-8")

        prompt = memory_prompt.invoke(
            {
                "messages": messages_to_memorize,
                "existing_summary": running_memory.summary if running_memory else "",
                "existing_memory": existing_memory,
            }
        )

        result = await model.ainvoke(prompt)
        
        return _prepare_result(
            result=result,
            memory_file=memory_file,
            running_memory=running_memory,
            messages_to_memorize=messages_to_memorize,
            n_messages_memorized=preprocessed_messages.n_messages_memorized
        )

    return running_memory


class MemorizationNode(RunnableCallable):
    """Momory Node of graph"""
    
    def __init__(
        self,
        *,
        model: BaseModel,
        max_tokens: int = 10000,
        keep_ratio: float = 0.3,
        token_counter: TokenCounter = count_tokens_approximately,
        memory_prompt: ChatPromptTemplate = MEMORY_PROMPT,
        input_key: str = "messages",
        output_key: str = "running_memory",
        name: str = "memorization",
    ) -> None:
        super().__init__(self._func, self._afunc, name=name, trace=False)
        self.model = model
        self.max_tokens = max_tokens
        self.keep_ratio = keep_ratio
        self.token_counter =  token_counter
        self.memory_prompt = memory_prompt
        self.input_key = input_key
        self.output_key = output_key
    
    def _parse_input(self, input: dict[str, Any] | BaseModel) -> tuple[list[AnyMessage], RunningMemory]:
        if isinstance(input, dict):
            messages = input.get(self.input_key)
            running_memory = input.get(self.output_key)
        elif isinstance(input, BaseModel):
            messages = getattr(input, self.input_key, None)
            running_memory = getattr(input, self.output_key, None)
        else:
            raise ValueError(f"Invalid input type: {type(input)}")
        
        if messages is None:
            raise ValueError(
                f"Missing required field `{self.input_key}` in the input."
            )
        return messages, running_memory
    
    def _func(self, input: dict[str, Any] | BaseModel, config: RunnableConfig) -> dict[str, Any]:
        messages, running_memory = self._parse_input(input)
        running_memory = memorize_messages(
            messages,
            workspace=config.get("configurable", {}).get("workspace"),
            running_memory=running_memory,
            model=self.model,
            max_tokens=self.max_tokens,
            keep_ratio=self.keep_ratio,
            token_counter = self.token_counter,
            memory_prompt = self.memory_prompt
        )
        return {self.output_key: running_memory}
    
    async def _afunc(self, input: dict[str, Any] | BaseModel, config: RunnableConfig) -> dict[str, Any]:
        messages, running_memory = self._parse_input(input)
        running_memory = await amemorize_messages(
            messages,
            workspace=config.get("configurable", {}).get("workspace"),
            running_memory=running_memory,
            model=self.model,
            max_tokens=self.max_tokens,
            keep_ratio=self.keep_ratio,
            token_counter = self.token_counter,
            memory_prompt = self.memory_prompt
        )
        return {self.output_key: running_memory}