"""LangGraph 节点实现。

每个节点是一个 ``Callable[[GreenhouseState], dict]``，返回需要更新的字段。
节点之间通过 :mod:`greenhouse_agent.state` 中定义的 ``GreenhouseState`` 共享数据。

设计要点：
    - 所有文件路径通过 :mod:`greenhouse_agent.config` 集中管理，使用绝对路径，
      避免因 Streamlit 启动目录不同导致路径错乱。
    - 天气数据通过 :mod:`greenhouse_agent.tools` 调用 weather_mcp 服务。
    - LLM 调用通过 :mod:`greenhouse_agent.llm` 走 DeepSeek V4 Pro。
    - 节点只负责数据与业务逻辑，不直接渲染 UI（UI 由 Streamlit 侧处理）。
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from greenhouse_agent import config, llm, tools
from greenhouse_agent.state import GreenhouseState


# =====================================================================
# 工具函数
# =====================================================================

def _log(state: GreenhouseState, msg: str) -> None:
    """向 state.logs 追加一条日志（不会覆盖已有日志）。"""
    state.setdefault("logs", []).append(msg)


def _read_sensor_data(file_path) -> pd.DataFrame:
    """读取大棚传感器 Excel 文件。"""
    if not file_path.exists():
        raise FileNotFoundError(
            f"未找到传感器数据文件：{file_path}，请确认文件位于项目根目录。"
        )
    return pd.read_excel(file_path, sheet_name="Sheet1")


def _save_weather_forecast_csv(
    forecasts: List[Dict[str, Any]], filename
) -> None:
    """把天气预报写入 CSV 文件。"""
    fieldnames = [
        "城市", "日期", "温度", "气压", "湿度", "天气状况", "风力"
    ]
    with open(
        filename, mode="w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in forecasts:
            writer.writerow(row)


def _save_current_weather_csv(
    weather: Dict[str, Any], filename
) -> None:
    """把实时天气写入 CSV 文件。"""
    fieldnames = [
        "城市", "日期", "温度", "气压", "湿度", "天气状况", "风速"
    ]
    with open(
        filename, mode="w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(weather)


# =====================================================================
# 1. 基本信息
# =====================================================================

def basic_info_node(state: GreenhouseState) -> Dict[str, Any]:
    """记录基本信息，并把品种/土壤类型写回 sensor_data Excel。"""
    _log(state, f"[basic_info] city={state.get('city')}, "
                f"crop={state.get('crop_variety')}, soil={state.get('agrotype')}")

    updates: Dict[str, Any] = {}

    city = state.get("city", "")
    crop_variety = state.get("crop_variety", "")
    agrotype = state.get("agrotype", "")

    if not (city and crop_variety and agrotype):
        updates["error"] = "请填写城市、作物种类、土壤类型后再提交。"
        return updates

    try:
        # 把品种/土壤类型写回 Excel（与原系统行为保持一致）
        for file_path in (
            config.SENSOR_DATA_FILE,
            config.SENSOR_DATA_TODAY_FILE,
        ):
            if file_path.exists():
                df = pd.read_excel(file_path)
                df["品种"] = crop_variety
                df["土壤类型"] = agrotype
                df.to_excel(file_path, index=False)
    except Exception as e:
        updates["error"] = f"写入基本信息失败：{e}"
        return updates

    updates["city"] = city
    updates["crop_variety"] = crop_variety
    updates["agrotype"] = agrotype
    updates["error"] = None
    return updates


# =====================================================================
# 2. 天气预报
# =====================================================================

def weather_forecast_node(state: GreenhouseState) -> Dict[str, Any]:
    """通过 weather MCP 工具获取未来五天天气预报，并保存为 CSV。"""
    city = state.get("city") or "Qingdao"
    _log(state, f"[weather_forecast] city={city}")

    updates: Dict[str, Any] = {"city": city}
    try:
        forecasts = tools.call_weather_tool(
            "get_weather_forecast", city_name=city
        )
        if forecasts and isinstance(forecasts, list) and "error" not in forecasts[0]:
            _save_weather_forecast_csv(forecasts, config.WEATHER_FORECAST_FILE)
            updates["weather_forecast"] = forecasts
            updates["weather_error"] = None
        else:
            err = forecasts[0].get("error") if forecasts else "未获取到数据"
            updates["weather_forecast"] = []
            updates["weather_error"] = err
    except Exception as e:
        updates["weather_forecast"] = []
        updates["weather_error"] = f"获取天气预报失败：{e}"

    return updates


# =====================================================================
# 3. 实时监测（实时天气 + 大棚传感器数据）
# =====================================================================

def realtime_monitoring_node(state: GreenhouseState) -> Dict[str, Any]:
    """获取实时天气 + 大棚传感器数据，组装为监测面板所需的数据。"""
    city = state.get("city") or "Qingdao"
    _log(state, f"[realtime_monitoring] city={city}")

    updates: Dict[str, Any] = {"city": city}

    # --- 实时天气 ---
    try:
        weather = tools.call_weather_tool(
            "get_current_weather", city_name=city
        )
        if isinstance(weather, dict) and "error" not in weather:
            _save_current_weather_csv(weather, config.REALTIME_WEATHER_FILE)
            updates["current_weather"] = weather
            updates["weather_error"] = None
        else:
            updates["current_weather"] = None
            updates["weather_error"] = weather.get("error") if weather else "无数据"
    except Exception as e:
        updates["current_weather"] = None
        updates["weather_error"] = f"获取实时天气失败：{e}"

    # --- 大棚传感器数据（当前小时 vs 上一小时）---
    try:
        df = _read_sensor_data(config.SENSOR_DATA_FILE)
        now = datetime.now()
        current_hour = now.hour
        prev_hour = (now - timedelta(hours=1)).hour

        current_data = df[df["时间"] == current_hour]
        previous_data = df[df["时间"] == prev_hour]

        if not previous_data.empty and not current_data.empty:
            cur = current_data.iloc[0]
            prev = previous_data.iloc[0]
            metrics = {
                "品种": str(cur["品种"]),
                "土壤湿度": float(cur["土壤湿度"]),
                "土壤湿度变化": float(cur["土壤湿度"] - prev["土壤湿度"]),
                "光照": float(cur["光照"]),
                "光照变化": float(cur["光照"] - prev["光照"]),
                "无机盐浓度": float(cur["无机盐浓度"]),
                "无机盐浓度变化": float(cur["无机盐浓度"] - prev["无机盐浓度"]),
                "CO2浓度": float(cur["CO2浓度"]),
                "CO2浓度变化": float(cur["CO2浓度"] - prev["CO2浓度"]),
                "大棚温度": float(cur["大棚温度"]),
                "大棚温度变化": float(cur["大棚温度"] - prev["大棚温度"]),
            }
            updates["sensor_data"] = metrics
            updates["sensor_error"] = None
        else:
            updates["sensor_data"] = None
            updates["sensor_error"] = "没有找到当前时间的数据。"
    except Exception as e:
        updates["sensor_data"] = None
        updates["sensor_error"] = f"读取传感器数据失败：{e}"

    return updates


# =====================================================================
# 4. 大棚情况（读取一周 / 当日数据用于绘图）
# =====================================================================

def greenhouse_status_node(state: GreenhouseState) -> Dict[str, Any]:
    """读取一周 / 当日大棚数据，供 Streamlit 绘制图表。"""
    _log(state, "[greenhouse_status] 读取大棚数据")

    updates: Dict[str, Any] = {}

    # 一周数据
    try:
        df_week = _read_sensor_data(config.SENSOR_DATA_FILE)
        updates["sensor_data"] = df_week
        updates["sensor_error"] = None
    except Exception as e:
        updates["sensor_error"] = f"读取一周数据失败：{e}"
        updates["sensor_data"] = None

    # 当日数据
    try:
        df_today = _read_sensor_data(config.SENSOR_DATA_TODAY_FILE)
        df_today["日期"] = pd.to_datetime(df_today["日期"], format="%Y-%m-%d")
        today = datetime.today().date()
        filtered = df_today[df_today["日期"].dt.date == today]
        updates["sensor_data_today"] = filtered
    except Exception as e:
        updates["sensor_data_today"] = None
        if updates.get("sensor_error"):
            updates["sensor_error"] += f" | 读取当日数据失败：{e}"
        else:
            updates["sensor_error"] = f"读取当日数据失败：{e}"

    return updates


# =====================================================================
# 5. 备忘录
# =====================================================================

def _load_memos() -> List[Dict[str, Any]]:
    if not config.MEMO_FILE.exists():
        config.MEMO_FILE.write_text("[]", encoding="utf-8")
    try:
        return json.loads(config.MEMO_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        return []


def _save_memos(memos: List[Dict[str, Any]]) -> None:
    config.MEMO_FILE.write_text(
        json.dumps(memos, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )


def memo_node(state: GreenhouseState) -> Dict[str, Any]:
    """备忘录增删查。根据 ``state['memo_action']`` 分支处理。"""
    action = state.get("memo_action", "view")
    _log(state, f"[memo] action={action}")

    updates: Dict[str, Any] = {}
    memos = _load_memos()

    if action == "add":
        text = (state.get("memo_text") or "").strip()
        if not text:
            updates["memo_message"] = "备忘录内容不能为空。"
        else:
            memo = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "content": text,
            }
            memos.append(memo)
            _save_memos(memos)
            updates["memo_message"] = "备忘录已添加。"
        updates["memos"] = memos

    elif action == "delete":
        idx = state.get("memo_filter_date")  # 复用字段存放索引字符串
        try:
            idx_int = int(idx) if idx is not None else -1
        except (TypeError, ValueError):
            idx_int = -1
        if 0 <= idx_int < len(memos):
            removed = memos.pop(idx_int)
            _save_memos(memos)
            updates["memo_message"] = f"已删除备忘录：{removed.get('content', '')[:30]}..."
        else:
            updates["memo_message"] = "索引无效，无法删除。"
        updates["memos"] = memos

    else:  # view
        sel_date = state.get("memo_filter_date")
        if sel_date:
            filtered = [m for m in memos if m.get("date") == sel_date]
            updates["memos"] = filtered
            updates["memo_message"] = f"找到 {len(filtered)} 条备忘录（{sel_date}）。"
        else:
            updates["memos"] = memos
            updates["memo_message"] = f"共 {len(memos)} 条备忘录。"

    return updates


# =====================================================================
# 6. 优化种植方案（DeepSeek V4 Pro 生成）
# =====================================================================

def _build_optimization_query(state: GreenhouseState) -> str:
    """根据当前大棚数据 + 未来五天天气，组装优化种植方案的 prompt。"""
    # 计算当前时刻对应的 Excel 行索引（与原系统保持一致）
    t = time_str = datetime.now().strftime("%A %b %d %H:%M:%S %Y")
    parts = t.split()
    week_map = {
        "Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4,
        "Friday": 5, "Saturday": 6, "Sunday": 7,
    }
    week = week_map.get(parts[0], 1)
    hms = parts[3].split(":")
    hour = int(hms[0])
    moment = int(hms[2])
    if moment > 30:
        hour += 1
    if hour == 0:
        hour = 24
        week -= 1

    # 大棚数据
    df = pd.read_excel(config.SENSOR_DATA_FILE)
    index = (week - 1) * 24 + hour
    if index >= len(df):
        index = len(df) - 1
    row = df.iloc[index]
    data = {k: str(v) for k, v in row.items()}

    greenhouse_condition = (
        f"当地时刻{parts[3]}，大棚内种有{data.get('品种', '未知')}"
        f"，大棚温度为{data.get('大棚温度', '未知')}℃"
        f"，土壤为{data.get('土壤类型', '未知')}"
        f"，土壤湿度为{data.get('土壤湿度', '未知')}%"
        f"，土壤EC值为{data.get('无机盐浓度', '未知')}"
        f"，大棚内二氧化碳浓度为{data.get('CO2浓度', '未知')}ppm"
        f"，此时的光照强度为{data.get('光照', '未知')}万勒克斯。"
    )

    # 天气预报
    future_condition = ""
    if config.ALL_CLIMATE_FILE.exists():
        climate = pd.read_csv(config.ALL_CLIMATE_FILE)
        climate["Avg Temperature"] = climate["Avg Temperature"].astype(str) + "℃"
        climate["Avg Pressure"] = climate["Avg Pressure"].astype(str) + "hpa"
        climate["Avg Humidity"] = climate["Avg Humidity"].astype(str) + "%"
        climate["Temperature Range"] = (
            climate["Temperature Range"].astype(str) + " ℃"
        )
        avg_temp = "，".join(climate["Avg Temperature"].tail(5))
        avg_pres = "， ".join(climate["Avg Pressure"].tail(5))
        avg_hum = "， ".join(climate["Avg Humidity"].tail(5))
        desc = "，".join(climate["Description"].tail(5))
        t_range = "，".join(climate["Temperature Range"].tail(5))
        future_condition = (
            f"未来五天室外平均气温分别为{avg_temp}"
            f"，温度区间分别为{t_range}"
            f"，平均气压分别为{avg_pres}"
            f"，平均湿度分别为{avg_hum}"
            f"，天气情况分别为{desc}。"
        )

    return f"{greenhouse_condition}{future_condition}请给出后续的优化种植方案。"


def optimization_plan_node(state: GreenhouseState) -> Dict[str, Any]:
    """调用 DeepSeek V4 Pro 生成优化种植方案。"""
    _log(state, "[optimization_plan] 生成优化种植方案")

    updates: Dict[str, Any] = {}
    try:
        query = state.get("optimization_query") or _build_optimization_query(state)
        updates["optimization_query"] = query

        system_prompt = (
            "你是一位经验丰富的农业种植专家与大棚管理顾问。"
            "请根据用户提供的大棚当前状态与未来五天天气预报，"
            "给出具体、可执行的优化种植方案，包括温度、湿度、光照、"
            "通风、灌溉、施肥等方面的调整建议。"
            "请使用 Markdown 格式输出，结构清晰，便于阅读。"
        )
        plan = llm.invoke_llm(query, system=system_prompt)
        updates["optimization_plan"] = plan
        updates["error"] = None
    except Exception as e:
        updates["optimization_plan"] = ""
        updates["error"] = f"生成优化种植方案失败：{e}"

    return updates


# =====================================================================
# 7. 病虫害处理（语义搜索 + DeepSeek 兜底 + 自学习）
# =====================================================================

# PestSearcher 单例（延迟初始化，首次调用时加载 BGE 模型）
_pest_searcher = None


def _get_pest_searcher():
    global _pest_searcher
    if _pest_searcher is not None:
        return _pest_searcher
    from greenhouse_agent.knowledge import PestSearcher

    _pest_searcher = PestSearcher()
    _pest_searcher.init()
    return _pest_searcher


def _render_structured_solution(sol) -> str:
    """把 StructuredSolution 渲染为统一格式的 Markdown 文本。

    DeepSeek 兜底与自学习命中均复用此函数，保证展示风格一致。
    """
    parts = [f"### 诊断\n{sol.diagnosis}"]
    if sol.cause:
        parts.append(f"\n**病因分析**：{sol.cause}")
    if sol.treatments:
        parts.append("\n#### 处理方法")
        for t in sol.treatments:
            parts.append(f"- **{t.drug_name}**（{t.drug_type}）")
            if t.dosage:
                parts.append(f"  - 用量：{t.dosage}")
            if t.application_method:
                parts.append(f"  - 施用方式：{t.application_method}")
    if sol.preventions:
        parts.append("\n#### 预防方法")
        for p in sol.preventions:
            parts.append(f"- {p.measure}")
            if p.timing:
                parts.append(f"  - 时机：{p.timing}")
    if sol.notes:
        parts.append(f"\n**注意事项**：{sol.notes}")
    return "\n".join(parts)


def _build_pest_json_prompt(disease_name: str):
    """构造让 DeepSeek 输出结构化 JSON 的 system / user prompt。"""
    system_prompt = (
        "你是一位植物保护专家。针对用户描述的农作物病虫害，给出结构化方案。"
        "必须严格输出 JSON，字段如下：\n"
        "{\n"
        '  "diagnosis": "诊断结果/病害判断",\n'
        '  "cause": "病因/病原分析",\n'
        '  "treatments": [\n'
        '    {"drug_name": "药剂/措施名", "drug_type": "化学/生物/物理", '
        '"dosage": "用量与稀释倍数", "application_method": "施用方式"}\n'
        "  ],\n"
        '  "preventions": [\n'
        '    {"measure": "预防措施", "timing": "执行时机"}\n'
        "  ],\n"
        '  "notes": "注意事项"\n'
        "}\n"
        "treatments 与 preventions 至少各给 1 条，多多益善。"
    )
    user_prompt = f"作物及病害名称：{disease_name}。请给出结构化方案（JSON）。"
    return system_prompt, user_prompt


def pest_disease_node(state: GreenhouseState) -> Dict[str, Any]:
    """病虫害处理：语义搜索知识库 → 自学习命中 → DeepSeek 兜底。"""
    disease_name = (state.get("disease_name") or "").strip()
    _log(state, f"[pest_disease] disease_name={disease_name}")

    updates: Dict[str, Any] = {"disease_name": disease_name}
    if not disease_name:
        updates["disease_solution"] = ""
        updates["error"] = "请输入病虫害名称。"
        return updates

    # 1. 语义搜索知识库
    try:
        searcher = _get_pest_searcher()
        result = searcher.search(disease_name)
    except Exception as e:
        _log(state, f"[pest_disease] 知识库搜索失败: {e}")
        result = None

    # 2. 命中正式知识库
    if result and result.matched and result.disease:
        d = result.disease
        parts = [
            f"### {d.crop_type} · {d.disease_name}",
            f"- **类别**：{d.disease_category}",
            f"- **症状**：{d.symptoms}",
            f"- **病原**：{d.cause}",
            f"- **相似度**：{result.similarity:.1%}",
        ]
        if d.treatments:
            parts.append("\n#### 处理方法")
            for t in d.treatments:
                parts.append(f"- **{t.drug_name}**（{t.drug_type}）")
                if t.dosage:
                    parts.append(f"  - 用量：{t.dosage}")
                if t.application_method:
                    parts.append(f"  - 施用方式：{t.application_method}")
        if d.preventions:
            parts.append("\n#### 预防方法")
            for p in d.preventions:
                parts.append(f"- {p.measure}")
                if p.timing:
                    parts.append(f"  - 时机：{p.timing}")
        parts.append(f"\n> 数据来源：知识库")

        updates["disease_treatment"] = d.treatments[0].dosage if d.treatments else None
        updates["disease_prevention"] = d.preventions[0].measure if d.preventions else None
        updates["disease_solution"] = "\n".join(parts)
        updates["error"] = None
        return updates

    # 3. 命中自学习表（之前 DeepSeek 生成过的方案）
    if result and result.matched and result.llm_solution:
        sol = result.llm_solution
        structured = sol.structured
        if structured:
            # 结构化渲染（新数据）
            body = _render_structured_solution(structured)
            updates["disease_treatment"] = (
                structured.treatments[0].dosage if structured.treatments else None
            )
            updates["disease_prevention"] = (
                structured.preventions[0].measure if structured.preventions else None
            )
        else:
            # 向后兼容：老数据无 solution_json，用原文
            body = sol.solution
            updates["disease_treatment"] = None
            updates["disease_prevention"] = None
        updates["disease_solution"] = (
            f"> *此问题此前已由 AI 解答（相似度 {result.similarity:.1%}），"
            f"自动从自学习库命中：*\n\n{body}"
        )
        updates["error"] = None
        return updates

    # 4. 未命中，调用 DeepSeek V4 Pro 兜底（优先结构化 JSON 输出）
    try:
        from greenhouse_agent.knowledge.models import StructuredSolution

        system_prompt, user_prompt = _build_pest_json_prompt(disease_name)
        structured: Optional[StructuredSolution] = None
        body: str

        # 优先尝试 JSON 结构化输出
        try:
            data = llm.invoke_llm_json(user_prompt, system=system_prompt)
            structured = StructuredSolution.model_validate(data)
            body = _render_structured_solution(structured)
            _log(state, "[pest_disease] DeepSeek 返回结构化 JSON 成功")
        except Exception as json_err:
            # Fallback：JSON 失败时退回纯文本
            _log(state, f"[pest_disease] JSON 模式失败，回退纯文本: {json_err}")
            text_system = (
                "你是一位植物保护专家。请针对用户描述的农作物病虫害，"
                "给出具体、可执行的处理方法与预防方案。"
                "请使用 Markdown 格式输出，分为「处理方法」与「预防方法」两部分。"
            )
            body = llm.invoke_llm(
                f"作物及病害名称：{disease_name}。请给出处理方法与预防方案。",
                system=text_system,
            )

        updates["disease_treatment"] = (
            structured.treatments[0].dosage
            if structured and structured.treatments else None
        )
        updates["disease_prevention"] = (
            structured.preventions[0].measure
            if structured and structured.preventions else None
        )
        updates["disease_solution"] = (
            f"> *本地知识库未收录此病害，以下方案由 DeepSeek V4 Pro 生成：*\n\n"
            + body
        )
        updates["error"] = None

        # 5. 自学习：保存方案到知识库（含结构化数据）
        try:
            searcher = _get_pest_searcher()
            searcher.save_solution(
                disease_name, body,
                confidence=result.similarity if result else 0,
                structured=structured,
            )
            tag = "（含结构化 JSON）" if structured else "（纯文本，JSON 失败）"
            _log(state, f"[pest_disease] AI 方案已保存到自学习库{tag}")
        except Exception as save_err:
            _log(state, f"[pest_disease] 保存自学习方案失败: {save_err}")

    except Exception as e:
        updates["disease_solution"] = ""
        updates["error"] = f"病虫害处理失败：{e}"

    return updates
