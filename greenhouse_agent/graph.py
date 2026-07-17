"""LangGraph 图构建。

图结构：

    START
      ↓
    router (按 state['action'] 分支)
      ↓
    ┌── basic_info ──┐
    ├── weather_forecast ──┐
    ├── realtime_monitoring ──┤
    ├── greenhouse_status ──┤
    ├── memo ──┤
    ├── optimization_plan ──┤
    └── pest_disease ──┘
                             ↓
                           END

每个业务节点执行完毕后直接结束，不再循环回 router。
如果未来需要多步协作（如先查天气再生成方案），可在此处扩展为多跳路径。
"""

from __future__ import annotations

from typing import Callable, Literal

from langgraph.graph import END, START, StateGraph

from greenhouse_agent import nodes
from greenhouse_agent.state import GreenhouseState


# =====================================================================
# 路由
# =====================================================================

def route_by_action(state: GreenhouseState) -> str:
    """根据 ``state['action']`` 选择下一个节点名。

    Returns:
        节点名（与下方 ``add_node`` 注册的名字一一对应）。
        若 action 缺失或未知，返回 ``END``。
    """
    action = (state.get("action") or "").strip()
    action_to_node = {
        "基本信息": "basic_info",
        "天气预报": "weather_forecast",
        "实时监测": "realtime_monitoring",
        "大棚情况": "greenhouse_status",
        "备忘录": "memo",
        "优化种植方案": "optimization_plan",
        "病虫害处理": "pest_disease",
    }
    return action_to_node.get(action, END)


# =====================================================================
# 构建图
# =====================================================================

def build_greenhouse_graph():
    """构建并编译智慧大棚 LangGraph。

    Returns:
        编译后的 ``CompiledGraph``，可通过 ``.invoke(state)`` 调用。
    """
    builder = StateGraph(GreenhouseState)

    # ---- 注册节点 ----
    builder.add_node("basic_info", nodes.basic_info_node)
    builder.add_node("weather_forecast", nodes.weather_forecast_node)
    builder.add_node("realtime_monitoring", nodes.realtime_monitoring_node)
    builder.add_node("greenhouse_status", nodes.greenhouse_status_node)
    builder.add_node("memo", nodes.memo_node)
    builder.add_node("optimization_plan", nodes.optimization_plan_node)
    builder.add_node("pest_disease", nodes.pest_disease_node)

    # ---- 入口 ----
    builder.add_conditional_edges(
        START,
        route_by_action,
        {
            "basic_info": "basic_info",
            "weather_forecast": "weather_forecast",
            "realtime_monitoring": "realtime_monitoring",
            "greenhouse_status": "greenhouse_status",
            "memo": "memo",
            "optimization_plan": "optimization_plan",
            "pest_disease": "pest_disease",
            END: END,
        },
    )

    # ---- 所有业务节点都直接结束 ----
    for node_name in (
        "basic_info",
        "weather_forecast",
        "realtime_monitoring",
        "greenhouse_status",
        "memo",
        "optimization_plan",
        "pest_disease",
    ):
        builder.add_edge(node_name, END)

    return builder.compile()
