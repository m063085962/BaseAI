import json

from typing import Literal, Union
from pydantic import BaseModel, Field
from pathlib import Path

class ProviderConfig(BaseModel):
    """LLM provider configuraion"""
    api_key: str = ""
    base_url: str | None = None

class AgentConfig(BaseModel):
    """Model configuration"""
    model: str = "glm-4.7-flash"
    max_tokens: int  = 4096
    temperature: float = 0.6
    workspace: str = ".agent/workspace"
    memory_window: int = 40000
    recursion_limit: int = 30

class HTTPMCPServerConfig(BaseModel):
    """MCP server configuration"""
    transport: Literal["http"] = "http"
    url: str = ""
    headers: dict[str, str] = Field(default_factory=dict)

class StdioMCPServerConfig(BaseModel):
    transport: Literal["stdio"] = "stdio"
    command: str | None = None
    args: list[str] = Field(default_factory=list)

class Configuration(BaseModel):
    """Root configuraion for baseai"""
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    mcp_servers: dict[str, Union[HTTPMCPServerConfig, StdioMCPServerConfig]] = Field(default_factory={})

class Config:

    def __init__(self, config_path: Path = Path(".agent/config.json")):
        self._config_path = config_path
        self._config = self._load()

    def set_config_path(self, path: Path) -> None:
        """Set the config path"""
        self._config_path = path

    def get_config_path(self) -> Path:
        """Get the configuration file path"""
        return self._config_path

    def _load(self) -> Configuration:
        """Load configuration from file or create default"""
        if self._config_path.exists():
            try:
                with open(self._config_path, encoding="utf-8") as f:
                    config = json.load(f)
                    return Configuration.model_validate(config)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Warning: Failed to load config from {self._config_path}: {e}")
                print("Using default configuration")
        
        return Configuration()
    
    def save(self, config: Configuration) -> None:
        """Save configuraion to file"""
        self._config_path.parent.mkdir(parents=True, exist_ok=True)

        data = config.model_dump()

        with open(self._config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_config(self, name: Literal["agent", "provider"]) -> Configuration:
        """Get the configuration by name"""
        if name == "agent":
            return self._config.agent
        elif name == "provider":
            return self._config.provider
        else:
            raise ValueError(f"Unknown config name: {name}")

    def get_mcp_server_config(self) -> dict:
        """Get MCP server configuration"""
        mcp_servers = {}
        for k in self._config.mcp_servers.keys():
            mcp_servers[k] = self._config.mcp_servers.get(k).model_dump()
        
        return mcp_servers

        