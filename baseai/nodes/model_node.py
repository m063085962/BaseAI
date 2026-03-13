from typing import Any
from pathlib import Path

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AnyMessage
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.utils.runnable import RunnableCallable
from pydantic import BaseModel

from baseai.skill import SkillsLoader
from baseai.tool import ToolResgistry
from baseai.tools.filesystem import WORKSPACE_DIR

LLM_INPUT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",         
            "{agent_instruction}"
            "\n\n"
            "# Guidelines"
            "\n\n"
            "- 若用户指令模糊，主动追问以明确需求。\n"
            "- 基于已知知识提供信息，不确定或超出范围时如实告知。\n"
            "- 遇到复杂任务时，可使用spawn_subagent工具创建子代理来处理"
            "- 如需检索记忆，可使用read_file工具查看工作区中 MEMORY.md 文档"
            "\n\n"
            "# Runtime"
            "\n\n"
            "- 工作区目录为：{workspace}\n"
            "- Available Skills为可选技能，在创建子代理任务时可提供\n"
            "- Available Tools为可选工具，在创建子代理任务时可提供\n"
            "- Summary为最近对话的摘要"
            "\n\n"
            "## Available Skills""\n\n""{skills}"
            "\n\n"
            "## Available Tools""\n\n""{tools}"
            "\n\n"
            "## Summary""\n\n""{summary}"
        ),
        ("placeholder", "{messages}"),
    ]
)

class ModelNode(RunnableCallable):
    """构造上下文信息并调用模型"""

    def __init__(
        self,
        model: LanguageModelLike,
        *,
        skills: SkillsLoader,
        tools: ToolResgistry,
        messages_key: str = "messages",
        name: str = "model",
    ) -> None:
        super().__init__(self._func, self._afunc, name=name, trace=False)
        self.model = model
        self.messages_key = messages_key
        self.skills = skills
        self.tools = tools
    
    def _build_messages(self, input: dict[str, Any] | BaseModel, config: RunnableConfig) -> list[AnyMessage]:
        "整合上下文信息构造消息"
        if isinstance(input, dict):
            messages = input.get(self.messages_key)
            running_memory = input.get("running_memory")
        elif isinstance(input, BaseModel):
            messages = getattr(input, self.messages_key, None)
            running_memory = getattr(input, "running_memory", None)
        else:
            raise ValueError(f"Invalid input type: {type(input)}")
        
        if running_memory and (messages := messages[running_memory.index:]):
            return []
        
        workspace = config.get("configurable").get("workspace", WORKSPACE_DIR)
        
        agent_file = workspace / "AGENT.md"
        agent_instruction = "You are a helpful AI Assistant."
        if agent_file.exists():
            agent_instruction = agent_file.read_text(encoding="utf-8")
        
        summary = running_memory.summary if running_memory else ""

        llm_input_messages = LLM_INPUT_PROMPT.invoke(
            {
                "agent_instruction": agent_instruction,
                "workspace": workspace,
                "skills": self.skills.get_skills_summary(),
                "tools": self.tools.get_tools_summary(),
                "summary": summary,
                "messages": messages,
            }
        )
        
        return llm_input_messages
    
    def _func(self, input: dict[str, Any] | BaseModel, config: RunnableConfig) -> dict[str, Any]:
        messages = self._build_messages(input, config)
        if not messages:
            return {}
        
        response = self.model.invoke(messages)

        return {self.messages_key: [response]}
        
    
    async def _afunc(self, input: dict[str, Any] | BaseModel, config: RunnableConfig) -> dict[str, Any]:
        messages = self._build_messages(input, config)
        if not messages:
            return {}
        
        response = await self.model.ainvoke(messages)

        return {self.messages_key: [response]}