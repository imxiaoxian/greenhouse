"""病虫害知识库数据模型。

使用 Pydantic v2 定义结构化模型，统一用于 SQLite 与 PostgreSQL 两种后端。
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class Treatment(BaseModel):
    """处理方法（药剂/物理/生物防治）。"""

    id: Optional[int] = None
    disease_id: Optional[int] = None
    drug_name: str = Field(..., description="药剂/措施名称")
    drug_type: str = Field(default="化学", description="类型：化学/生物/物理")
    dosage: str = Field(default="", description="用量与稀释倍数")
    application_method: str = Field(default="", description="施用方式（喷雾/灌根/涂抹等）")
    timing: str = Field(default="", description="施用时机")
    rotation_period: str = Field(default="", description="抗药性轮换周期")
    cost_estimate: str = Field(default="", description="成本估算")
    precautions: str = Field(default="", description="注意事项")


class Prevention(BaseModel):
    """预防措施。"""

    id: Optional[int] = None
    disease_id: Optional[int] = None
    measure: str = Field(..., description="预防措施描述")
    timing: str = Field(default="", description="预防时机")
    frequency: str = Field(default="", description="执行频率")


class Disease(BaseModel):
    """病害主记录。"""

    id: Optional[int] = None
    crop_type: str = Field(..., description="作物类型（番茄/黄瓜/西瓜等）")
    disease_name: str = Field(..., description="病害名称")
    disease_category: str = Field(default="", description="类别（真菌/细菌/病毒/虫害/生理性）")
    symptoms: str = Field(default="", description="典型症状描述")
    cause: str = Field(default="", description="病原/病因")
    favorable_conditions: str = Field(default="", description="发病有利条件（温湿度等）")
    season: str = Field(default="", description="高发季节")
    severity_level: int = Field(default=3, ge=1, le=5, description="严重度等级 1-5")
    source: str = Field(default="manual", description="数据来源：manual/llm_promoted")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # 关联数据（查询时填充）
    treatments: List[Treatment] = Field(default_factory=list)
    preventions: List[Prevention] = Field(default_factory=list)


class LLMSolution(BaseModel):
    """DeepSeek 生成的方案（自学习积累）。"""

    id: Optional[int] = None
    query_text: str = Field(..., description="用户查询原文")
    disease_id: Optional[int] = Field(default=None, description="关联病害（可为空）")
    solution: str = Field(..., description="AI 生成的方案原文（Markdown）")
    solution_json: Optional[str] = Field(
        default=None,
        description="结构化方案 JSON 字符串（与 solution 同时保存）",
    )
    model: str = Field(default="deepseek-v4-pro", description="生成模型")
    confidence: Optional[float] = Field(default=None, description="匹配置信度")
    promoted: bool = Field(default=False, description="是否已被提升为正式记录")
    created_at: Optional[datetime] = None

    @property
    def structured(self) -> Optional["StructuredSolution"]:
        """运行时反序列化为结构化对象（若 solution_json 存在且可解析）。"""
        if not self.solution_json:
            return None
        try:
            return StructuredSolution.model_validate_json(self.solution_json)
        except Exception:
            return None


class StructuredTreatment(BaseModel):
    """AI 生成的处理方法（结构化）。"""

    drug_name: str = Field(..., description="药剂/措施名称")
    drug_type: str = Field(default="化学", description="化学/生物/物理")
    dosage: str = Field(default="", description="用量与稀释倍数")
    application_method: str = Field(default="", description="施用方式")


class StructuredPrevention(BaseModel):
    """AI 生成的预防措施（结构化）。"""

    measure: str = Field(..., description="预防措施描述")
    timing: str = Field(default="", description="执行时机")


class StructuredSolution(BaseModel):
    """DeepSeek 生成的结构化方案。

    用于自学习保存：DeepSeek 输出 JSON，解析为该对象后序列化存入
    ``llm_solutions.solution_json`` 字段，前端可结构化渲染。
    """

    diagnosis: str = Field(..., description="诊断结果/病害判断")
    cause: str = Field(default="", description="病因/病原分析")
    treatments: List[StructuredTreatment] = Field(
        default_factory=list, description="处理方法列表"
    )
    preventions: List[StructuredPrevention] = Field(
        default_factory=list, description="预防措施列表"
    )
    notes: str = Field(default="", description="注意事项")


class SearchResult(BaseModel):
    """语义搜索结果。"""

    disease: Optional[Disease] = None
    similarity: float = Field(default=0.0, description="相似度 0-1")
    matched: bool = Field(default=False, description="是否达到阈值")
    llm_solution: Optional[LLMSolution] = Field(default=None, description="自学习命中的 AI 方案（若有）")


class KnowledgeStats(BaseModel):
    """知识库统计信息。"""

    disease_count: int = 0
    treatment_count: int = 0
    prevention_count: int = 0
    llm_solution_count: int = 0
    promoted_count: int = 0
