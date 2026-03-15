<h1 align="center">BaseAI: Simple Personal AI Assistant</h1>

基于LangChain/LangGraph构建的轻量级 AI 个人助手，采用事件驱动的消息总线架构，支持技能(Skill)、MCP 工具集成、对话记忆管理以及子代理任务分派。

## 核心特性 

- 🔌 **解耦设计**：消息总线隔离输入渠道与处理逻辑
- 🤖 **多智能体**：主代理任务分派与子代理异步执行
- 🧠 **智能记忆**：对话窗口管理、上下文压缩与长期记忆
- 🔧 **工具系统**：内置文件系统工具，支持 MCP 工具
- 📚 **技能框架**：实现技能渐进披露，支持热插拔

## 项目结构

```
baseai/
├── agent.py              # AgentServer 主类实现
├── bus.py                # 异步消息总线
├── config.py             # 配置模型与加载器
├── skill.py              # 技能加载器
├── nodes/
│   ├── model_node.py     # LLM 工具调用节点
│   ├── memorization.py   # 对话历史与长期记忆管理
└── tools/
    ├── Registry.py       # 工具注册表
    ├── filesystem.py     # 内置文件系统工具
    └── spawn.py          # 调用子代理工具
```
