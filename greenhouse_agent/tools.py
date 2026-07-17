"""MCP 工具加载与缓存。

LangGraph 节点可以通过两种方式获取天气数据：

1. **直接 Python 调用**：通过 :func:`call_weather_tool` 同步调用本地 MCP 服务
   中定义的工具函数，性能最佳，进程内通信。
2. **MCP 协议调用**：通过 :func:`get_mcp_tools` 启动 MCP 子进程并把工具转成
   LangChain Tool，用于跨进程或跨服务集成场景。

默认采用方式 1，避免 Streamlit + asyncio 的复杂度。
"""

import asyncio
import os
import sys
from functools import lru_cache
from typing import Any, Dict, List

from greenhouse_agent import config


def _run_async(coro):
    """在同步上下文中执行协程；若已有事件循环则使用新线程运行。"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 已在事件循环中（如 Streamlit 的某些场景），用新线程运行
            import threading

            result: Dict[str, Any] = {}

            def runner():
                new_loop = asyncio.new_event_loop()
                try:
                    result["value"] = new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()

            t = threading.Thread(target=runner)
            t.start()
            t.join()
            return result.get("value")
    except RuntimeError:
        # 没有事件循环，正常路径
        pass

    return asyncio.run(coro)


def call_weather_tool(tool_name: str, **kwargs) -> Any:
    """直接调用 weather_mcp 服务中定义的工具函数（同步、进程内）。

    Args:
        tool_name: 工具名，如 ``get_current_weather`` / ``get_weather_forecast``。
        **kwargs: 工具参数。

    Returns:
        工具返回值。
    """
    # 确保环境变量已经加载到当前进程（MCP 服务依赖）
    os.environ.setdefault(
        "OPENWEATHER_API_KEY", config.OPENWEATHER_API_KEY
    )

    # 延迟导入，避免循环导入
    from weather_mcp import server as weather_server

    tool_map = {
        "get_current_weather": weather_server.get_current_weather,
        "get_weather_forecast": weather_server.get_weather_forecast,
    }
    fn = tool_map.get(tool_name)
    if fn is None:
        raise ValueError(f"未知的天气工具：{tool_name}")
    return fn(**kwargs)


@lru_cache(maxsize=1)
def _get_mcp_client():
    """创建并缓存 MCP 客户端（stdio 模式启动 weather_mcp 子进程）。

    注意：返回的客户端 ``get_tools()`` 是协程，需要配合 ``_run_async`` 使用。
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    server_script = (
        config.PROJECT_ROOT / "weather_mcp" / "server.py"
    ).as_posix()

    connections = {
        "weather": {
            "command": sys.executable,
            "args": ["-m", "weather_mcp.server"],
            "transport": "stdio",
            "env": {
                "OPENWEATHER_API_KEY": config.OPENWEATHER_API_KEY,
                "PYTHONPATH": config.PROJECT_ROOT.as_posix(),
            },
        }
    }
    return MultiServerMCPClient(connections)


def get_mcp_tools_sync():
    """同步获取 MCP 工具列表（跨进程 stdio 模式）。

    首次调用会启动 weather_mcp 子进程并加载工具。
    后续调用走缓存。
    """
    client = _get_mcp_client()
    return _run_async(client.get_tools())


def call_mcp_tool_sync(tool_name: str, **kwargs) -> Any:
    """通过 MCP 协议（stdio）调用天气工具。

    适用于需要严格走 MCP 协议的场景。日常使用请优先用
    :func:`call_weather_tool` 避免子进程开销。
    """
    tools = get_mcp_tools_sync()
    for tool in tools:
        if tool.name == tool_name:
            return tool.invoke(kwargs)
    raise ValueError(f"MCP 服务中未找到工具：{tool_name}")
