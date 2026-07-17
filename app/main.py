"""智慧大棚管理系统主入口。

基于 Streamlit 多页面应用 + LangGraph Agent 后端。
大模型：DeepSeek V4 Pro（通过 langchain-deepseek 接入）。
天气数据：通过 weather_mcp 服务获取。
"""

import os
import sys
from pathlib import Path

import streamlit as st

# 把项目根目录加入 sys.path，使 greenhouse_agent 包可被导入
# app/main.py → app/ → 项目根目录
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 把项目根目录设为 CWD，便于读写 data/ 下文件
os.chdir(_PROJECT_ROOT)

st.set_page_config(
    page_title="智慧大棚管理系统",
    page_icon="🌱",
    layout="wide",
)

# 使用 st.navigation 统一管理页面（英文文件名 + 中文显示标题）
pg = st.navigation(
    [
        st.Page("pages/greenhouse_manage.py", title="大棚管理", icon="🌱", default=True),
        st.Page("pages/data_analysis.py", title="自定义数据分析", icon="📊"),
        st.Page("pages/knowledge_admin.py", title="病虫害知识库", icon="📚"),
    ]
)
pg.run()
