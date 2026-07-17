"""LangGraph 状态模式定义。

整个智慧大棚 Agent 的所有节点共享 ``GreenhouseState``。
每个字段对应一个业务功能产生的中间数据或最终结果。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class GreenhouseState(TypedDict, total=False):
    """智慧大棚 Agent 共享状态。

    ``total=False`` 表示所有字段都是可选的，节点只需返回自己更新的字段。
    """

    # ===== 路由：用户选择的业务动作 =====
    # 取值见 config.ACTIONS
    action: str

    # ===== 基本信息 =====
    city: str
    crop_variety: str
    agrotype: str

    # ===== 天气数据（来自 weather MCP）=====
    current_weather: Optional[Dict[str, Any]]
    weather_forecast: Optional[List[Dict[str, Any]]]
    weather_error: Optional[str]

    # ===== 大棚传感器数据 =====
    # pandas DataFrame 不能直接放进 TypedDict 的严格类型，统一用 Any
    sensor_data: Any
    sensor_data_today: Any
    sensor_error: Optional[str]

    # ===== 备忘录 =====
    memos: List[Dict[str, Any]]
    memo_text: str
    memo_action: str  # "add" / "view" / "delete"
    memo_filter_date: str  # "%Y-%m-%d"
    memo_message: str  # 操作反馈信息

    # ===== 优化种植方案（LLM 生成）=====
    optimization_query: str
    optimization_plan: str

    # ===== 病虫害处理 =====
    disease_name: str
    disease_treatment: Optional[str]
    disease_prevention: Optional[str]
    # 当 SQLite 没查到时，由 LLM 兜底生成
    disease_solution: str

    # ===== 错误信息 =====
    error: Optional[str]

    # ===== 消息日志（节点 append 形式累积，便于调试）=====
    logs: Annotated[List[str], operator.add]
