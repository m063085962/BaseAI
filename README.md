# BaseAI


基于LangChain/LangGraph构建的轻量级 AI 个人助手，采用事件驱动的消息总线架构，支持动态技能加载、MCP 工具集成、对话记忆管理以及子代理任务分派。

## 核心特性 

- 🔌 **解耦设计**：消息总线隔离输入渠道与处理逻辑
- 🧠 **智能记忆**：对话摘要、长期记忆与 token 窗口管理
- 🔧 **工具系统**：内置文件系统工具，支持 MCP 工具动态注入
- 📚 **技能框架**：工作区与内置技能管理，支持 YAML 元数据定义
- 🎯 **状态管理**：基于 LangGraph 的可靠状态转移与检查点保存
- 🚀 **子代理支持**：灵活的任务分派与多代理编排


## 项目结构

```
baseai/
├── agent.py              # AgentServer 主类实现
├── bus.py                # 异步消息总线
├── config.py             # 配置模型与加载器
├── skill.py              # 技能加载器
├── tool.py               # 工具注册表
├── nodes/
│   ├── model_node.py     # LLM 工具调用节点
│   ├── memorization.py   # 对话历史与长期记忆管理
├── tools/
│   └── filesystem.py     # 内置文件系统工具集
```
