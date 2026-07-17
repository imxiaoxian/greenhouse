"""端到端测试：语义搜索 + 自学习全流程。

测试场景：
    1. 搜索已知病害症状（应命中知识库）
    2. 搜索未登录的新问题（应不命中）
    3. 模拟 DeepSeek 生成方案并保存（自学习）
    4. 再次搜索相同 query（应命中已保存的 AI 方案）
"""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from greenhouse_agent.knowledge import PestSearcher
from greenhouse_agent.knowledge.embeddings import get_embedding_dimension


def main() -> int:
    print("=== 知识库端到端测试 ===\n")

    # 0. 模型维度
    dim = get_embedding_dimension()
    print(f"[0] BGE 向量维度: {dim}")
    assert dim == 512, f"期望 512，实际 {dim}"
    print()

    searcher = PestSearcher()
    # 确保 schema 最新（含 solution_json 列）
    searcher.init()
    # 清理之前测试残留的 AI 方案，保证测试可重复
    searcher.repo.clear_llm_solutions()
    print("[预清理] 已清空 llm_solutions 表\n")

    # 1. 搜索已知病害症状（番茄早疫病典型症状）
    print("[1] 搜索已知病害：'番茄叶子有同心轮纹的暗褐色病斑'")
    result = searcher.search("番茄叶子有同心轮纹的暗褐色病斑")
    print(f"    matched={result.matched}, similarity={result.similarity:.4f}")
    if result.disease:
        print(f"    命中病害: {result.disease.crop_type} / {result.disease.disease_name}")
        print(f"    类别: {result.disease.disease_category}")
        print(f"    处理方法数: {len(result.disease.treatments)}")
        print(f"    预防措施数: {len(result.disease.preventions)}")
    assert result.matched, "应该命中已知病害"
    print()

    # 2. 搜索未登录的虚构问题
    print("[2] 搜索未登录问题：'火星作物紫色斑点扩散'")
    result2 = searcher.search("火星作物紫色斑点扩散")
    print(f"    matched={result2.matched}, similarity={result2.similarity:.4f}")
    assert not result2.matched, "不应该命中"
    print()

    # 3. 模拟 DeepSeek 生成的方案并保存（自学习 + 结构化）
    print("[3] 保存 AI 方案（模拟 DeepSeek 兜底，结构化 JSON）...")
    query = "火星作物紫色斑点扩散"
    fake_solution = (
        "## 火星作物紫斑病处理方案\n\n"
        "1. 隔离病株，避免扩散\n"
        "2. 调整大棚大气成分，降低 CO2 浓度\n"
    )
    from greenhouse_agent.knowledge.models import (
        StructuredSolution, StructuredTreatment, StructuredPrevention,
    )
    fake_structured = StructuredSolution(
        diagnosis="火星作物紫斑病（疑似真菌感染）",
        cause="低气压高湿环境诱发未知真菌",
        treatments=[
            StructuredTreatment(
                drug_name="多菌灵可湿性粉剂",
                drug_type="化学",
                dosage="稀释 800 倍",
                application_method="叶面喷雾",
            ),
            StructuredTreatment(
                drug_name="UV-C 紫外线照射",
                drug_type="物理",
                dosage="每日 15 分钟",
                application_method="近距离照射",
            ),
        ],
        preventions=[
            StructuredPrevention(
                measure="调整大棚大气成分，降低 CO2 浓度",
                timing="发病初期每日",
            ),
        ],
        notes="火星作物无地球病害直接对照，建议送回地球实验室分析",
    )
    sol_id = searcher.save_solution(
        query_text=query,
        solution=fake_solution,
        model="deepseek-v4-pro",
        confidence=result2.similarity,
        structured=fake_structured,
    )
    print(f"    已保存结构化 AI 方案，id={sol_id}")
    print()

    # 4. 再次搜索相同 query，应命中已保存的 AI 方案
    print("[4] 再次搜索相同 query，验证自学习命中（含结构化数据）...")
    result3 = searcher.search(query)
    print(f"    matched={result3.matched}, similarity={result3.similarity:.4f}")
    assert result3.llm_solution is not None, "应该命中已保存的 AI 方案"
    sol = result3.llm_solution
    print(f"    命中 AI 方案 id={sol.id} (model={sol.model})")
    # 验证结构化字段
    assert sol.solution_json is not None, "solution_json 应非空"
    structured_back = sol.structured
    assert structured_back is not None, "应能反序列化为 StructuredSolution"
    print(f"    诊断: {structured_back.diagnosis}")
    print(f"    病因: {structured_back.cause}")
    print(f"    处理方法数: {len(structured_back.treatments)}")
    for i, t in enumerate(structured_back.treatments, 1):
        print(f"      [{i}] {t.drug_name}（{t.drug_type}）- {t.dosage} - {t.application_method}")
    print(f"    预防措施数: {len(structured_back.preventions)}")
    for p in structured_back.preventions:
        print(f"      - {p.measure}（{p.timing}）")
    print(f"    注意事项: {structured_back.notes[:50]}")
    assert len(structured_back.treatments) == 2, "应有 2 条处理方法"
    assert len(structured_back.preventions) == 1, "应有 1 条预防措施"
    print()

    # 5. 统计
    print("[5] 知识库统计:")
    stats = searcher.get_stats()
    print(f"    病害数: {stats.disease_count}")
    print(f"    处理方法数: {stats.treatment_count}")
    print(f"    预防措施数: {stats.prevention_count}")
    print(f"    AI 方案数: {stats.llm_solution_count}")
    print(f"    已提升数: {stats.promoted_count}")
    print()

    print("=== 全部测试通过 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
