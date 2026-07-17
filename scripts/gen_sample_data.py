"""生成示例传感器数据文件，用于测试优化种植方案与大棚情况节点。

生成：
- data/sensor_data.xlsx      一周（7天 × 24小时 = 168 行）大棚传感器数据
- data/sensor_data_today.xlsx 当日（24 行）数据
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from greenhouse_agent import config  # noqa: E402

# 生成一周数据（168 行 = 7 天 × 24 小时）
days = 7
hours_per_day = 24
total_rows = days * hours_per_day

rows = []
base_time = datetime(2026, 7, 15, 0, 0, 0)
weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

for i in range(total_rows):
    day_idx = i // hours_per_day  # 0-6
    hour = i % hours_per_day      # 0-23
    ts = base_time + timedelta(days=day_idx, hours=hour)

    # 模拟昼夜温差：白天温度高，夜间低
    temp_base = 28 + 8 * max(0, -((hour - 14) ** 2) / 50 + 1)
    greenhouse_temp = round(temp_base + (day_idx * 0.3), 1)

    # 土壤湿度缓慢下降（需要灌溉）
    soil_moisture = round(65 - day_idx * 1.5 + (hour % 6) * 0.5, 1)

    # 光照强度（万勒克斯）：白天有光，夜间为 0
    if 6 <= hour <= 18:
        light = round(5 * max(0, -((hour - 12) ** 2) / 36 + 1), 2)
    else:
        light = 0.0

    # CO2 浓度：白天光合作用消耗，夜间呼吸释放
    co2 = 400 + (200 if hour < 6 or hour > 20 else -100) + (day_idx * 5)

    # 无机盐浓度（ppm）
    ec = 1200 + (hour % 8) * 10 - day_idx * 5

    rows.append({
        "品种": "番茄",
        "土壤类型": "砂壤土",
        "大棚温度": greenhouse_temp,
        "土壤湿度": soil_moisture,
        "光照": light,
        "CO2浓度": co2,
        "无机盐浓度": ec,
    })

df_week = pd.DataFrame(rows)
df_week.to_excel(config.SENSOR_DATA_FILE, sheet_name="Sheet1", index=False)
print(f"已生成 {config.SENSOR_DATA_FILE.name}（{len(df_week)} 行）")

# 生成当日数据（24 行）
today_str = datetime.today().strftime("%Y-%m-%d")
rows_today = []
for hour in range(hours_per_day):
    row = rows[hour].copy()
    row["日期"] = today_str
    rows_today.append(row)

df_today = pd.DataFrame(rows_today)
df_today.to_excel(config.SENSOR_DATA_TODAY_FILE, sheet_name="Sheet1", index=False)
print(f"已生成 {config.SENSOR_DATA_TODAY_FILE.name}（{len(df_today)} 行）")

print("\n前 3 行示例：")
print(df_week.head(3).to_string(index=False))
