"""病虫害知识库 Repository 抽象接口。

定义统一的 CRUD + 语义搜索接口，SQLite 与 PostgreSQL 后端分别实现。
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd

from greenhouse_agent import config
from greenhouse_agent.knowledge.models import (
    Disease,
    KnowledgeStats,
    LLMSolution,
    Prevention,
    SearchResult,
    Treatment,
)


class PestDiseaseRepository(ABC):
    """知识库存储抽象层。"""

    # ===== Schema 管理 =====

    @abstractmethod
    def init_schema(self) -> None:
        """创建表、索引、向量扩展。可重复调用（IF NOT EXISTS）。"""
        ...

    # ===== 向量搜索 =====

    @abstractmethod
    def search_by_vector(
        self, embedding: List[float], top_k: int = 5
    ) -> List[tuple]:
        """向量搜索最相似的病害。

        返回 [(disease_id, similarity, text_content), ...]
        similarity 范围 0-1，1 表示完全相同。
        """
        ...

    @abstractmethod
    def search_llm_solutions_by_vector(
        self, embedding: List[float], top_k: int = 3
    ) -> List[tuple]:
        """在自学习方案表中做向量搜索。

        返回 [(solution_id, similarity, solution_text), ...]
        """
        ...

    # ===== 病害 CRUD =====

    @abstractmethod
    def create_disease(
        self,
        disease: Disease,
        embedding_text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
    ) -> int:
        """创建病害记录，返回新 id。

        如果提供了 embedding_text + embedding，同时写入向量索引。
        """
        ...

    @abstractmethod
    def get_disease(self, disease_id: int) -> Optional[Disease]:
        """获取病害详情（含 treatments 和 preventions）。"""
        ...

    @abstractmethod
    def update_disease(self, disease: Disease) -> None:
        """更新病害信息。"""
        ...

    @abstractmethod
    def delete_disease(self, disease_id: int) -> None:
        """删除病害及其关联数据（treatments / preventions / embeddings）。"""
        ...

    @abstractmethod
    def list_diseases(
        self, crop_type: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> List[Disease]:
        """分页列出病害，可按作物筛选。"""
        ...

    # ===== Treatment / Prevention =====

    @abstractmethod
    def add_treatment(self, disease_id: int, treatment: Treatment) -> int:
        """添加处理方法。"""
        ...

    @abstractmethod
    def add_prevention(self, disease_id: int, prevention: Prevention) -> int:
        """添加预防措施。"""
        ...

    @abstractmethod
    def delete_treatment(self, treatment_id: int) -> None:
        """删除处理方法。"""
        ...

    @abstractmethod
    def delete_prevention(self, prevention_id: int) -> None:
        """删除预防措施。"""
        ...

    # ===== 自学习（LLM Solutions）=====

    @abstractmethod
    def save_llm_solution(
        self,
        query_text: str,
        solution: str,
        model: str,
        embedding: List[float],
        confidence: Optional[float] = None,
        solution_json: Optional[str] = None,
    ) -> int:
        """保存 DeepSeek 生成的方案，返回新 id。

        Args:
            solution: Markdown 原文（向后兼容展示用）
            solution_json: 结构化方案的 JSON 字符串（可选）
        """
        ...

    @abstractmethod
    def list_llm_solutions(
        self, promoted_only: bool = False, limit: int = 50
    ) -> List[LLMSolution]:
        """列出 AI 方案，可只看未提升的。"""
        ...

    @abstractmethod
    def promote_llm_solution(
        self, solution_id: int, disease_id: int
    ) -> None:
        """将 AI 方案标记为已提升（关联到正式病害记录）。"""
        ...

    @abstractmethod
    def clear_llm_solutions(self) -> None:
        """清空自学习表（测试用，慎用）。"""
        ...

    # ===== 统计 =====

    @abstractmethod
    def get_stats(self) -> KnowledgeStats:
        """获取知识库统计信息。"""
        ...

    # ===== 导入导出 =====

    @abstractmethod
    def export_diseases(self) -> pd.DataFrame:
        """导出所有病害为 DataFrame（含 treatments 文本拼接）。"""
        ...

    @abstractmethod
    def import_diseases(self, df: pd.DataFrame) -> int:
        """从 DataFrame 批量导入病害，返回导入数量。"""
        ...


def get_repository() -> PestDiseaseRepository:
    """根据配置创建对应的 Repository 实例。"""
    backend = config.KNOWLEDGE_DB_BACKEND
    if backend == "postgres":
        from greenhouse_agent.knowledge.postgres_repo import PostgresRepository

        return PostgresRepository()
    else:
        from greenhouse_agent.knowledge.sqlite_repo import SQLiteRepository

        return SQLiteRepository()
