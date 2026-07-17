"""大棚管理页面。

所有业务功能（基本信息 / 天气预报 / 实时监测 / 大棚情况 / 备忘录 /
优化种植方案 / 病虫害处理）均通过 LangGraph Agent 执行，本页面只负责
用户交互与结果渲染。

LLM: DeepSeek V4 Pro（greenhouse_agent.llm）
天气: weather MCP（greenhouse_agent.tools）
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# 把项目根目录加入 sys.path，使 greenhouse_agent 包可被导入
# app/pages/greenhouse_manage.py → app/pages/ → app/ → 项目根目录
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 与原系统一致，CWD 设为项目根目录，便于读写 sensor_data.xlsx 等文件
os.chdir(_PROJECT_ROOT)

from greenhouse_agent import config  # noqa: E402
from greenhouse_agent.graph import build_greenhouse_graph  # noqa: E402


# ----------------------------------------------------------------------
# 全局初始化
# ----------------------------------------------------------------------

# 缓存编译后的 LangGraph，避免每次 rerun 都重新构建
@st.cache_resource
def get_greenhouse_graph():
    return build_greenhouse_graph()


def init_session_state():
    """初始化 Streamlit session_state 中的全局变量。"""
    defaults = {
        "city_name": "",
        "last_result": None,  # 最近一次 LangGraph 执行结果
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()

# 侧边栏：业务功能选择
add_selectbox = st.sidebar.selectbox(
    "智慧大棚管理系统",
    (
        "基本信息",
        "天气预报",
        "实时监测",
        "大棚情况",
        "备忘录",
        "优化种植方案",
        "病虫害处理",
    ),
)


def run_graph(state: dict) -> dict:
    """运行 LangGraph 并把结果缓存到 session_state。"""
    graph = get_greenhouse_graph()
    result = graph.invoke(state)
    st.session_state["last_result"] = result
    return result


def show_last_error():
    """如果最近一次执行有错误，在顶部展示。"""
    last = st.session_state.get("last_result") or {}
    err = last.get("error")
    if err:
        st.error(err)


# ----------------------------------------------------------------------
# 1. 基本信息
# ----------------------------------------------------------------------

if add_selectbox == "基本信息":
    st.title("智慧大棚管理系统 - 基本信息")

    with st.form(key="basic_info_form"):
        city = st.text_input(
            "请输入您所在的城市（输入城市名拼音） 例：Qingdao",
            value=st.session_state["city_name"],
            max_chars=100,
            help="精确到地级市",
        )
        crop_variety = st.text_input(
            "请输入作物种类", max_chars=100, help="输入一种作物种类"
        )
        agrotype = st.text_input(
            "请输入土壤类型", max_chars=100, help="输入一种土壤类型"
        )
        submit = st.form_submit_button(label="提交基本信息")

    if submit:
        result = run_graph(
            {
                "action": "基本信息",
                "city": city,
                "crop_variety": crop_variety,
                "agrotype": agrotype,
            }
        )
        if result.get("error"):
            st.error(result["error"])
        else:
            st.session_state["city_name"] = city
            st.write("您所在的城市是", city)
            st.write("您种植的品种是", crop_variety)
            st.write("您大棚的土壤类型是", agrotype)
            st.success("输入成功")


# ----------------------------------------------------------------------
# 2. 天气预报
# ----------------------------------------------------------------------

elif add_selectbox == "天气预报":
    st.title("智慧大棚管理系统 - 天气预报")
    city = st.text_input(
        "城市（拼音）", value=st.session_state["city_name"] or "Qingdao"
    )

    if st.button("获取天气预报"):
        with st.spinner("正在通过 weather MCP 获取天气预报 ..."):
            result = run_graph({"action": "天气预报", "city": city})

        forecasts = result.get("weather_forecast") or []
        err = result.get("weather_error")
        if err:
            st.error(err)
        elif forecasts:
            st.subheader("详细天气预报")
            st.dataframe(pd.DataFrame(forecasts))

            # 按日期筛选
            sel_date = st.date_input("选择一个日期", value=datetime.today())
            filtered = [
                f
                for f in forecasts
                if f.get("日期", "").startswith(
                    sel_date.strftime("%Y-%m-%d")
                )
            ]
            if filtered:
                st.subheader(f"{sel_date} 的天气预报")
                st.dataframe(pd.DataFrame(filtered))
            else:
                st.info(f"{sel_date} 无天气预报数据。")
        else:
            st.warning("未能获取天气预报数据。")


# ----------------------------------------------------------------------
# 3. 实时监测
# ----------------------------------------------------------------------

elif add_selectbox == "实时监测":
    st.title("智慧大棚管理系统 - 实时监测")
    city = st.text_input(
        "城市（拼音）", value=st.session_state["city_name"] or "Qingdao"
    )

    if st.button("刷新实时监测"):
        with st.spinner("正在获取实时天气与大棚数据 ..."):
            result = run_graph({"action": "实时监测", "city": city})

    # 如果有上次结果，直接展示（避免每次 rerun 都重新拉取）
    result = st.session_state.get("last_result") or {}
    if result.get("action") == "实时监测":
        st.header("实时天气")
        weather = result.get("current_weather")
        if result.get("weather_error"):
            st.error(result["weather_error"])
        elif weather:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("城市", weather.get("城市", ""))
            with col2:
                st.metric("温度", f"{weather.get('温度')} °C")
            with col3:
                st.metric("气压", f"{weather.get('气压')} hPa")
            with col1:
                st.metric("天气状况", weather.get("天气状况", ""))
            with col2:
                st.metric("风速", f"{weather.get('风速')} m/s")
            with col3:
                st.metric("湿度", f"{weather.get('湿度')} %")
        else:
            st.info("点击上方按钮获取实时天气。")

        st.header("大棚数据")
        metrics = result.get("sensor_data")
        if result.get("sensor_error"):
            st.warning(result["sensor_error"])
        elif metrics:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("品种", metrics["品种"])
            with col2:
                st.metric(
                    "土壤湿度",
                    f"{metrics['土壤湿度']} %",
                    f"{metrics['土壤湿度变化']:.2f} %",
                )
            with col3:
                st.metric(
                    "光照",
                    f"{metrics['光照']} lux",
                    f"{metrics['光照变化']:.2f} lux",
                )
            with col1:
                st.metric(
                    "无机盐浓度",
                    f"{metrics['无机盐浓度']} ppm",
                    f"{metrics['无机盐浓度变化']:.2f} ppm",
                )
            with col2:
                st.metric(
                    "CO2浓度",
                    f"{metrics['CO2浓度']} ppm",
                    f"{metrics['CO2浓度变化']:.2f} ppm",
                )
            with col3:
                st.metric(
                    "大棚温度",
                    f"{metrics['大棚温度']} °C",
                    f"{metrics['大棚温度变化']:.2f} °C",
                )
        else:
            st.info("暂无大棚传感器数据。")
    else:
        st.info("点击上方按钮获取实时监测数据。")


# ----------------------------------------------------------------------
# 4. 大棚情况（图表展示）
# ----------------------------------------------------------------------

elif add_selectbox == "大棚情况":
    st.title("智慧大棚管理系统 - 大棚情况")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        load_week = st.button("获取一周内大棚图表数据")
    with col_btn2:
        load_today = st.button("获取当日大棚图表数据")

    if load_week:
        with st.spinner("读取一周大棚数据 ..."):
            result = run_graph({"action": "大棚情况"})

    if load_today:
        # 当日数据复用同一节点（同一份 state），简化为只刷新当日
        with st.spinner("读取当日大棚数据 ..."):
            result = run_graph({"action": "大棚情况"})

    result = st.session_state.get("last_result") or {}
    if result.get("action") == "大棚情况":
        if result.get("sensor_error"):
            st.warning(result["sensor_error"])

        # 一周数据图表
        df_week = result.get("sensor_data")
        if df_week is not None and not df_week.empty:
            st.subheader("一周 - 大棚温度")
            st.line_chart(df_week["大棚温度"])
            st.subheader("一周 - 土壤湿度")
            st.line_chart(df_week["土壤湿度"])
            st.subheader("一周 - 光照")
            st.area_chart(df_week["光照"])
            st.subheader("一周 - CO2浓度")
            st.bar_chart(df_week["CO2浓度"])
            st.subheader("一周 - 无机盐浓度")
            st.bar_chart(df_week["无机盐浓度"])

        # 当日数据图表
        df_today = result.get("sensor_data_today")
        if df_today is not None and not df_today.empty:
            st.subheader("当日 - 大棚温度")
            st.line_chart(df_today["大棚温度"])
            st.subheader("当日 - 土壤湿度")
            st.line_chart(df_today["土壤湿度"])
            st.subheader("当日 - 光照")
            st.area_chart(df_today["光照"])
            st.subheader("当日 - CO2浓度")
            st.bar_chart(df_today["CO2浓度"])
            st.subheader("当日 - 无机盐浓度")
            st.bar_chart(df_today["无机盐浓度"])

        if (df_week is None or df_week.empty) and (
            df_today is None or df_today.empty
        ):
            st.info(
                "未读取到数据，请确认 `data/` 目录下存在 "
                "`sensor_data.xlsx` 与 `sensor_data_today.xlsx`。"
            )
    else:
        st.info("点击按钮获取大棚图表数据。")


# ----------------------------------------------------------------------
# 5. 备忘录
# ----------------------------------------------------------------------

elif add_selectbox == "备忘录":
    st.title("智慧大棚管理系统 - 备忘录")

    # 添加备忘录
    with st.form(key="memo_add_form"):
        memo_text = st.text_area("输入备忘录内容：")
        add_submitted = st.form_submit_button("添加备忘录")

    if add_submitted and memo_text.strip():
        result = run_graph(
            {
                "action": "备忘录",
                "memo_action": "add",
                "memo_text": memo_text,
            }
        )
        if result.get("memo_message"):
            st.success(result["memo_message"])

    # 查看备忘录
    sel_date = st.date_input("选择日期", datetime.today())
    if st.button("查看备忘录列表"):
        result = run_graph(
            {
                "action": "备忘录",
                "memo_action": "view",
                "memo_filter_date": sel_date.strftime("%Y-%m-%d"),
            }
        )
        memos = result.get("memos", [])
        if memos:
            st.header("备忘录列表")
            for i, m in enumerate(memos):
                st.write(f"{m.get('date')}: {m.get('content')}")
                if st.button(f"删除 #{i + 1}", key=f"del_{i}"):
                    del_result = run_graph(
                        {
                            "action": "备忘录",
                            "memo_action": "delete",
                            "memo_filter_date": str(i),
                        }
                    )
                    if del_result.get("memo_message"):
                        st.success(del_result["memo_message"])
        else:
            st.warning("暂无备忘录")


# ----------------------------------------------------------------------
# 6. 优化种植方案
# ----------------------------------------------------------------------

elif add_selectbox == "优化种植方案":
    st.title("智慧大棚管理系统 - 优化种植方案")

    st.markdown(
        "由 **DeepSeek V4 Pro** 根据当前大棚数据与未来五天天气预报生成。"
    )

    if st.button("获取优化种植方案"):
        with st.spinner("DeepSeek V4 Pro 正在生成方案 ..."):
            result = run_graph({"action": "优化种植方案"})

        if result.get("error"):
            st.error(result["error"])

        plan = result.get("optimization_plan")
        if plan:
            with st.expander("查看发送给模型的 Prompt", expanded=False):
                st.write(result.get("optimization_query", ""))
            st.markdown("---")
            st.markdown(plan)


# ----------------------------------------------------------------------
# 7. 病虫害处理
# ----------------------------------------------------------------------

elif add_selectbox == "病虫害处理":
    st.title("智慧大棚管理系统 - 病虫害处理")

    disease_name = st.text_input(
        "请输入作物以及病的名称: 例：草莓褐斑病",
        max_chars=100,
        help="输入病的名称即可",
    )

    if st.button("获取解决办法"):
        if not disease_name.strip():
            st.warning("请输入病虫害名称。")
        else:
            with st.spinner("正在查询本地数据库 / 调用 DeepSeek V4 Pro ..."):
                result = run_graph(
                    {
                        "action": "病虫害处理",
                        "disease_name": disease_name,
                    }
                )

            if result.get("error"):
                st.error(result["error"])

            solution = result.get("disease_solution")
            if solution:
                # 数据库命中时显示处理 / 预防方法
                treatment = result.get("disease_treatment")
                prevention = result.get("disease_prevention")
                if treatment and prevention:
                    st.write(f"病虫害名称：{disease_name}")
                    st.write(f"处理方法：{treatment}")
                    st.write(f"预防方法：{prevention}")
                else:
                    # DeepSeek 兜底，直接渲染 Markdown
                    st.write(f"病虫害名称：{disease_name}")
                    st.markdown(solution)
