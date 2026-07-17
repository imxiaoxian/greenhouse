"""天气 MCP 服务。

把 OpenWeatherMap API 封装为 MCP 工具，提供：
  - get_current_weather: 获取指定城市的实时天气
  - get_weather_forecast: 获取指定城市未来五天天气预报

运行方式：
    # stdio 模式（默认，供 LangGraph 同进程调用）
    python -m weather_mcp.server

    # SSE 模式（独立服务，供远程调用）
    WEATHER_MCP_TRANSPORT=sse python -m weather_mcp.server
"""

import os
from typing import Any, Dict, List

import requests
from mcp.server.fastmcp import FastMCP

# 英文 -> 中文 天气状况映射
WEATHER_MAP: Dict[str, str] = {
    "clear sky": "晴",
    "few clouds": "少云",
    "scattered clouds": "晴间多云",
    "broken clouds": "多云",
    "overcast clouds": "阴",
    "shower rain": "阵雨",
    "rain": "雨",
    "thunderstorm": "雷阵雨",
    "snow": "雪",
    "mist": "雾",
}

# 创建 MCP 服务器
mcp = FastMCP("weather-mcp")


def _get_api_key() -> str:
    """从环境变量读取 OpenWeatherMap API Key。"""
    api_key = os.environ.get("OPENWEATHER_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "未配置 OPENWEATHER_API_KEY 环境变量，请在 .env 文件中设置。"
        )
    return api_key


@mcp.tool()
def get_current_weather(city_name: str) -> Dict[str, Any]:
    """获取指定城市的实时天气。

    Args:
        city_name: 城市名拼音，例如 "Qingdao"、"Beijing"。

    Returns:
        包含城市、日期、温度、气压、湿度、天气状况、风速的字典；
        失败时返回 {"error": "..."}。
    """
    api_key = _get_api_key()
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"q": city_name, "appid": api_key, "units": "metric"}

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        return {"error": f"获取实时天气失败: {e}"}

    from datetime import datetime

    main = data["main"]
    weather = data["weather"][0]
    wind = data["wind"]
    weather_desc_en = weather["description"]
    weather_desc = WEATHER_MAP.get(weather_desc_en, weather_desc_en)

    return {
        "城市": city_name,
        "日期": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "温度": main["temp"],
        "气压": main["pressure"],
        "湿度": main["humidity"],
        "天气状况": weather_desc,
        "风速": wind["speed"],
    }


@mcp.tool()
def get_weather_forecast(city_name: str) -> List[Dict[str, Any]]:
    """获取指定城市未来五天的天气预报。

    Args:
        city_name: 城市名拼音，例如 "Qingdao"、"Beijing"。

    Returns:
        天气预报条目列表，每项包含城市、日期、温度、气压、湿度、天气状况、风力；
        失败时返回 [{"error": "..."}]。
    """
    api_key = _get_api_key()
    base_url = "http://api.openweathermap.org/data/2.5/forecast"
    params = {"q": city_name, "appid": api_key, "units": "metric"}

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        return [{"error": f"获取天气预报失败: {e}"}]

    from datetime import datetime

    forecasts: List[Dict[str, Any]] = []
    for item in data.get("list", []):
        date = datetime.fromtimestamp(item["dt"]).strftime("%Y-%m-%d %H:%M:%S")
        forecasts.append(
            {
                "城市": city_name,
                "日期": date,
                "温度": item["main"]["temp"],
                "气压": item["main"]["pressure"],
                "湿度": item["main"]["humidity"],
                "天气状况": item["weather"][0]["description"],
                "风力": item["wind"]["speed"],
            }
        )
    return forecasts


def main() -> None:
    """启动 MCP 服务，根据环境变量选择传输方式。"""
    transport = os.environ.get("WEATHER_MCP_TRANSPORT", "stdio").lower()

    if transport == "sse":
        host = os.environ.get("WEATHER_MCP_HOST", "127.0.0.1")
        port = int(os.environ.get("WEATHER_MCP_PORT", "8765"))
        print(f"[weather-mcp] 启动 SSE 服务：http://{host}:{port}/sse")
        mcp.run(transport="sse", host=host, port=port)
    else:
        print("[weather-mcp] 启动 stdio 服务")
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
