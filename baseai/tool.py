from langchain_core.tools import BaseTool

from baseai.tools import filesystem_tools

class ToolResgistry:
    """Registry for agent tools"""

    def __init__(self):
        self._default_tools: list[BaseTool] = []
        self._extend_tools: dict[str, BaseTool] = {}
        self._register_dafault_tools()

    def _register_dafault_tools(self) -> None:
        for tool in filesystem_tools:
            self._default_tools.append(tool)

    def register(self, tool: BaseTool) -> None:
        """Register a tool"""
        self._extend_tools[tool.name] = tool

    def unregister(self, name: str) -> BaseTool | None:
        """Unregister a tool by name"""
        self._extend_tools.pop(name, None)
    
    def get(self, name: str) -> BaseTool | None:
        """Get a tool by name"""
        self._extend_tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool is registered"""
        return name in self._extend_tools
    
    def get_default_tools(self) -> list[BaseTool]:
        """Get default tools"""
        return self._default_tools
    
    def get_with_default(self, names: list[str] | None) -> list[BaseTool]:
        """Get specific tools with default tools"""
        tools = []
        for tool in self._default_tools:
            tools.append(tool)
        if names:
            for name in names:
                if tool := self._extend_tools.get(name):
                    tools.append(tool)
        
        return tools
    
    def get_tools_summary(self) -> str:
        """Get a summary of extend tools including name and description"""
        tools = []
        for tool in self._extend_tools.values():
            tools.append(f"- {tool.name}: {tool.description}")
        
        return "\n".join(tools)
    

