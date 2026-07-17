"""病虫害知识库模块。

提供语义检索、自学习积累、管理后台后端支持。

两套存储后端：
- SQLite + sqlite-vec（轻量级，默认）
- PostgreSQL + pgvector（企业级）

通过 .env 中 KNOWLEDGE_DB_BACKEND=sqlite|postgres 切换。
"""

from greenhouse_agent.knowledge.searcher import PestSearcher

__all__ = ["PestSearcher"]
