"""自定义数据分析页面。

使用 pygwalker 对用户上传的 CSV / Excel 文件进行交互式可视化探索。
此功能属于纯 UI 组件，不涉及 LLM，因此保留在 Streamlit 侧实现。
"""

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from pygwalker.api.streamlit import StreamlitRenderer

# 把项目根目录加入 sys.path，便于读取项目内数据文件
# app/pages/data_analysis.py → app/pages/ → app/ → 项目根目录
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

# pygwalker 配置文件路径
_PYGWALKER_SPEC = str(_PROJECT_ROOT / "config" / "pygwalker.json")

st.title("对选定数据进行分析")
st.header("文件读取示例")
uploaded_file = st.file_uploader("选择文件", type=["txt", "csv", "xlsx"])


@st.cache_resource
def get_pyg_renderer(file) -> "StreamlitRenderer | None":
    """缓存 pygwalker 渲染器，避免内存爆炸。"""
    if file is None:
        return None
    try:
        df = pd.read_csv(file)
    except UnicodeDecodeError:
        file.seek(0)
        df = pd.read_csv(file, encoding="latin1")
    return StreamlitRenderer(df, spec=_PYGWALKER_SPEC, spec_io_mode="rw")


if uploaded_file:
    renderer = get_pyg_renderer(uploaded_file)
    if renderer:
        renderer.explorer()
    else:
        st.error("未能生成渲染器，请上传有效文件。")
else:
    st.info("请上传一个文件进行分析。")
