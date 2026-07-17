# CLAUDE.md — Smart Greenhouse Management System

## Quick Start
```bash
cd greenhouse
pip install -r requirements.txt
cp .env.example .env  # Add DEEPSEEK_API_KEY + OPENWEATHER_API_KEY
streamlit run app/main.py
```

## Architecture
- **Agent**: LangGraph with 7 nodes (router → basic_info/weather_forecast/realtime_monitoring/greenhouse_status/memo/optimization_plan/pest_disease → END)
- **Weather**: MCP protocol via `weather_mcp/server.py` (FastMCP + OpenWeatherMap)
- **LLM**: DeepSeek V4 Pro via `langchain-deepseek`
- **Knowledge Base**: BGE embeddings + sqlite-vec (SQLite) or pgvector (PostgreSQL)
- **Frontend**: Streamlit 3-page app (st.navigation)

## Key Files
- `greenhouse_agent/graph.py` — StateGraph definition
- `greenhouse_agent/nodes.py` — 7 business node functions
- `greenhouse_agent/tools.py` — MCP weather tool wrapper (direct + protocol)
- `greenhouse_agent/knowledge/searcher.py` — Semantic search + self-learning
- `weather_mcp/server.py` — MCP weather service

## CI
- `.github/workflows/ci.yml` — Python 3.11/3.12 lint + import checks

## Notes
- BGE model auto-downloaded from HuggingFace on first run
- China users: set `HF_ENDPOINT=https://hf-mirror.com`
- `.conda-env/` is gitignored
- `data/bge_model/` is gitignored (users download on first run)
