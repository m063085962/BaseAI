# AGENTS.md - Agentic Coding Guidelines

BaseAI is a lightweight AI personal assistant built on LangChain/LangGraph with an event-driven message bus architecture.

- **Language**: Python 3.13+
- **Package Manager**: uv
- **Dependencies**: langchain, langgraph, langchain-openai, loguru, pydantic

---

## Build / Lint / Test Commands

```bash
uv sync                     # Install dependencies
uv sync --group dev         # Install dev dependencies
ruff check .                # Run ruff linter
ruff check . --fix          # Fix auto-fixable issues
ruff format .               # Format code
pytest                       # Run all tests
pytest tests/test_file.py    # Run a single test file
pytest tests/test_file.py::test_function_name  # Run a single test
pytest -k "test_pattern"     # Run tests matching a pattern
mypy baseai/                 # Type checking
```

---

## Code Style Guidelines

### General Principles

Be concise and accurate. Avoid unnecessary comments. Keep functions small and focused. Prefer early returns over deeply nested conditionals.

### Imports

**Order: standard library → third-party → local (alphabetical within each group)**

```python
import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, Field

from baseai.bus import InputMessage, OutputMessage, MessageBus
from baseai.tools import ToolResgistry
```

### Formatting

- **Line length**: 88 characters (enforced by ruff)
- **Indentation**: 4 spaces
- **Trailing commas**: Use for multi-line calls

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Classes | PascalCase | `AgentServer`, `MessageBus` |
| Functions/methods | snake_case | `get_tools()`, `_register_mcp_tools()` |
| Constants | SCREAMING_SNAKE_CASE | `WORKSPACE_DIR` |
| Private methods | prefix with underscore | `_build_agent()` |
| Variables | snake_case | `agent_tasks`, `memory_window` |

### Type Annotations

Use Python 3.13+ type hints. Use `X | None` instead of `Optional[X]`. Use `dict[str, Any]` for generic dicts. Use `Literal` for string literal types.

```python
def __init__(self, model: str, provider: str = "openai", max_tokens: int | None = None): ...

@dataclass
class InputMessage:
    content: str
    channel: str
    session_id: str
    sender: Literal["agent", "subagent", "user"] = "user"
    metadata: dict[str, Any] = field(default_factory=dict)
```

### Pydantic Models

```python
class AgentConfig(BaseModel):
    model: str = "glm-4.7-flash"
    max_tokens: int = 4096
    temperature: float = 0.6
```

### Error Handling & Logging

Use try/except sparingly and specifically. Catch specific exceptions, not bare `Exception`. Use loguru: `logger.info()`, `logger.warning()`, `logger.exception()`.

---

## Project Structure

```
baseai/
├── __init__.py           # Package init
├── agent.py              # AgentServer main class
├── bus.py                # Async message bus
├── cli.py                # CLI client entry point
├── config.py             # Configuration (Pydantic)
├── skill.py              # SkillsLoader
├── nodes/                # LangGraph nodes
│   ├── model_node.py     # LLM tool call node
│   └── memorization.py   # Memory management
├── tools/                # Tool system
│   ├── registry.py       # ToolResgistry
│   ├── filesystem.py    # Built-in file tools
│   └── spawn.py         # Sub-agent spawning
└── skills/              # Built-in skills
```

---

## Key Patterns

### Message Bus
```python
await bus.publish_input(InputMessage(content="Hello", channel="discord", session_id="user123"))
msg = await bus.consume_input()
```

### Tool Registration
```python
tools = ToolResgistry()
tools.register(my_tool)
default_tools = tools.get_default_tools()
```

### Agent StateGraph
```python
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
```

---

## Configuration Files

- **pyproject.toml**: Project metadata, dependencies, ruff config
- **.agent/config.json**: Runtime agent configuration
- **.agent/workspace/**: Agent workspace (skills, memory, instructions)
