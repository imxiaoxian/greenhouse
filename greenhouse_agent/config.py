"""配置与路径管理。"""

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 数据目录（运行时生成的 csv / json / db 都放这里）
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 自动加载 .env
load_dotenv(PROJECT_ROOT / ".env")

# ===== API Key =====
DEEPSEEK_API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", "")
OPENWEATHER_API_KEY: str = os.environ.get("OPENWEATHER_API_KEY", "")

# ===== DeepSeek 模型配置 =====
DEEPSEEK_MODEL: str = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")
DEEPSEEK_BASE_URL: str = os.environ.get(
    "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
)
DEEPSEEK_TEMPERATURE: float = float(
    os.environ.get("DEEPSEEK_TEMPERATURE", "0.5")
)
DEEPSEEK_MAX_TOKENS: int = int(
    os.environ.get("DEEPSEEK_MAX_TOKENS", "4096")
)

# ===== MCP 服务配置 =====
WEATHER_MCP_TRANSPORT: str = os.environ.get(
    "WEATHER_MCP_TRANSPORT", "stdio"
).lower()
WEATHER_MCP_HOST: str = os.environ.get("WEATHER_MCP_HOST", "127.0.0.1")
WEATHER_MCP_PORT: int = int(
    os.environ.get("WEATHER_MCP_PORT", "8765")
)

# ===== 数据文件路径（统一放在 data/ 目录下）=====
SENSOR_DATA_FILE: Path = DATA_DIR / "sensor_data.xlsx"
SENSOR_DATA_TODAY_FILE: Path = DATA_DIR / "sensor_data_today.xlsx"
WEATHER_FORECAST_FILE: Path = DATA_DIR / "weather_forecast.csv"
REALTIME_WEATHER_FILE: Path = DATA_DIR / "realtime_weather.csv"
ALL_CLIMATE_FILE: Path = DATA_DIR / "all_climate.csv"
MEMO_FILE: Path = DATA_DIR / "memos.json"
PEST_DISEASE_DB_FILE: Path = DATA_DIR / "pest_disease.db"

# ===== 病虫害知识库配置 =====
# 后端选择：sqlite（轻量级）或 postgres（企业级）
KNOWLEDGE_DB_BACKEND: str = os.environ.get(
    "KNOWLEDGE_DB_BACKEND", "sqlite"
).lower()

# PostgreSQL 连接参数（仅当 KNOWLEDGE_DB_BACKEND=postgres 时使用）
PG_HOST: str = os.environ.get("PG_HOST", "127.0.0.1")
PG_PORT: int = int(os.environ.get("PG_PORT", "5432"))
PG_DATABASE: str = os.environ.get("PG_DATABASE", "greenhouse")
PG_USER: str = os.environ.get("PG_USER", "postgres")
PG_PASSWORD: str = os.environ.get("PG_PASSWORD", "")

# 语义搜索参数
SIMILARITY_THRESHOLD: float = float(
    os.environ.get("SIMILARITY_THRESHOLD", "0.65")
)
EMBEDDING_MODEL: str = os.environ.get(
    "EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"
)

# ===== 可选的业务动作（与 Streamlit 侧边栏选项一一对应） =====
ACTIONS: List[str] = [
    "基本信息",
    "天气预报",
    "实时监测",
    "大棚情况",
    "备忘录",
    "优化种植方案",
    "病虫害处理",
]


def ensure_runtime_files() -> None:
    """确保运行时所需的文件存在（不存在则创建空文件）。"""
    if not MEMO_FILE.exists():
        MEMO_FILE.write_text("[]", encoding="utf-8")
