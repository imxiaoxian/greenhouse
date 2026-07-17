"""病虫害语义搜索 + 自学习引擎。

核心流程：
1. 用户输入症状/病害名 → BGE 编码为向量
2. 先在 disease_vectors 表搜索（正式知识库）
3. 再在 llm_solution_vectors 表搜索（自学习积累的 AI 方案）
4. 相似度 > 阈值 → 返回结构化结果
5. 相似度 < 阈值 → 返回空结果，调用方触发 DeepSeek 兜底
6. DeepSeek 方案生成后 → save_solution() 写入自学习表
"""

import logging
from typing import Optional

from greenhouse_agent import config
from greenhouse_agent.knowledge.embeddings import encode
from greenhouse_agent.knowledge.models import SearchResult
from greenhouse_agent.knowledge.repository import (
    PestDiseaseRepository,
    get_repository,
)

logger = logging.getLogger(__name__)


class PestSearcher:
    """病虫害知识库搜索引擎。

    封装语义搜索 + 自学习逻辑，对上层（nodes.py）提供简洁接口。
    """

    def __init__(self, repo: Optional[PestDiseaseRepository] = None):
        self._repo = repo or get_repository()

    @property
    def repo(self) -> PestDiseaseRepository:
        return self._repo

    def init(self) -> None:
        """初始化数据库 schema（建表 + 向量索引）。首次使用时调用。"""
        self._repo.init_schema()

    def get_stats(self):
        """获取知识库统计信息（委托给 repository）。"""
        return self._repo.get_stats()

    def search(self, query: str, top_k: int = 5) -> SearchResult:
        """语义搜索病虫害知识库。

        Args:
            query: 用户输入的症状描述或病害名称
            top_k: 返回的最相似结果数

        Returns:
            SearchResult: 包含 disease（若命中）和 similarity
        """
        embedding = encode(query)

        # 1. 先搜正式知识库
        hits = self._repo.search_by_vector(embedding, top_k=top_k)
        if hits:
            best_id, best_sim, best_text = hits[0]
            if best_sim >= config.SIMILARITY_THRESHOLD:
                disease = self._repo.get_disease(best_id)
                if disease:
                    return SearchResult(
                        disease=disease, similarity=best_sim, matched=True,
                    )

        # 2. 再搜自学习表（之前 DeepSeek 生成过的方案）
        llm_hits = self._repo.search_llm_solutions_by_vector(embedding, top_k=1)
        if llm_hits:
            sol_id, sol_sim, sol_text = llm_hits[0]
            if sol_sim >= config.SIMILARITY_THRESHOLD:
                from greenhouse_agent.knowledge.models import LLMSolution

                solutions = self._repo.list_llm_solutions(limit=100)
                for sol in solutions:
                    if sol.id == sol_id:
                        return SearchResult(
                            disease=None, similarity=sol_sim, matched=True,
                            llm_solution=sol,
                        )

        # 3. 未命中，返回最高相似度但不匹配的结果
        best_sim = hits[0][1] if hits else 0.0
        return SearchResult(similarity=best_sim, matched=False)

    def save_solution(
        self,
        query_text: str,
        solution: str,
        model: str = "deepseek-v4-pro",
        confidence: Optional[float] = None,
        structured: Optional["StructuredSolution"] = None,
    ) -> int:
        """保存 DeepSeek 生成的方案到自学习表。

        下次相同问题来时，向量搜索可直接命中，无需再调 DeepSeek。

        Args:
            query_text: 原始查询
            solution: AI 生成的方案原文（Markdown，向后兼容展示）
            model: 模型名
            confidence: 置信度
            structured: 结构化方案对象（可选），序列化为 JSON 一并存入

        Returns:
            新记录 id
        """
        from greenhouse_agent.knowledge.models import StructuredSolution  # noqa: F811

        embedding = encode(query_text)
        solution_json = (
            structured.model_dump_json(exclude_none=True) if structured else None
        )
        return self._repo.save_llm_solution(
            query_text, solution, model, embedding, confidence, solution_json,
        )

    def add_disease(
        self,
        crop_type: str,
        disease_name: str,
        symptoms: str = "",
        cause: str = "",
        disease_category: str = "",
        treatments: list = None,
        preventions: list = None,
    ) -> int:
        """添加病害到知识库（含向量索引）。

        供管理后台和初始化脚本调用。
        """
        from greenhouse_agent.knowledge.models import Disease, Treatment, Prevention

        # 构造用于 embedding 的文本（症状 + 病名 + 病原）
        embed_text = f"{crop_type} {disease_name} {symptoms} {cause}"
        embedding = encode(embed_text)

        disease = Disease(
            crop_type=crop_type,
            disease_name=disease_name,
            disease_category=disease_category,
            symptoms=symptoms,
            cause=cause,
            treatments=treatments or [],
            preventions=preventions or [],
        )
        return self._repo.create_disease(disease, embed_text, embedding)
