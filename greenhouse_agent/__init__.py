"""智慧大棚 LangGraph Agent 模块。

模块结构：
    state   —— LangGraph 状态模式定义
    llm     —— DeepSeek V4 Pro 模型封装
    config  —— 路径与配置
    tools   —— MCP 工具加载
    nodes   —— 各功能节点实现
    graph   —— LangGraph 图构建与编译

使用方式：
    from greenhouse_agent.graph import build_greenhouse_graph

    app = build_greenhouse_graph()
    result = app.invoke({"action": "天气预报", "city": "Qingdao"})
"""

from greenhouse_agent.graph import build_greenhouse_graph

__all__ = ["build_greenhouse_graph"]
