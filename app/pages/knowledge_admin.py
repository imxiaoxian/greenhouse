"""病虫害知识库管理后台。

功能：
- 病害列表：分页浏览、按作物筛选
- 新增/编辑病害：结构化录入（症状、药剂、预防）
- AI 方案审核：查看自学习库中的 DeepSeek 方案，提升为正式记录
- 搜索测试：输入症状描述，测试语义检索效果
- 导入导出：Excel 批量管理
"""

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# 项目根目录
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

from greenhouse_agent.knowledge.models import Disease, Prevention, Treatment
from greenhouse_agent.knowledge.repository import get_repository
from greenhouse_agent.knowledge.searcher import PestSearcher


@st.cache_resource
def _get_searcher() -> PestSearcher:
    s = PestSearcher()
    s.init()
    return s


def _render_disease_card(d: Disease):
    """渲染病害卡片。"""
    st.markdown(f"#### {d.crop_type} · {d.disease_name}")
    cols = st.columns([1, 1, 1, 1])
    cols[0].metric("类别", d.disease_category or "-")
    cols[1].metric("严重度", f"{'★' * d.severity_level}")
    cols[2].metric("季节", d.season or "-")
    cols[3].metric("药剂数", len(d.treatments))
    if d.symptoms:
        st.markdown(f"**症状**：{d.symptoms}")
    if d.cause:
        st.markdown(f"**病原**：{d.cause}")
    if d.treatments:
        st.markdown("**处理方法**：")
        for t in d.treatments:
            st.markdown(f"- {t.drug_name}（{t.drug_type}）：{t.dosage}")
    if d.preventions:
        st.markdown("**预防方法**：")
        for p in d.preventions:
            st.markdown(f"- {p.measure}")
    st.divider()


def page_disease_list(searcher: PestSearcher):
    """病害列表页。"""
    st.subheader("病害列表")
    repo = searcher.repo

    # 筛选
    col1, col2 = st.columns(2)
    with col1:
        crop_filter = st.selectbox(
            "按作物筛选",
            ["全部"] + [d.crop_type for d in repo.list_diseases(limit=10000)],
        )
    with col2:
        page_size = st.selectbox("每页显示", [10, 20, 50], index=0)

    diseases = repo.list_diseases(
        crop_type=None if crop_filter == "全部" else crop_filter,
        limit=10000,
    )
    total = len(diseases)
    if total == 0:
        st.info("知识库为空，请先运行 `python scripts/init_pest_db.py` 初始化数据。")
        return

    total_pages = (total + page_size - 1) // page_size
    page = st.number_input("页码", 1, total_pages, 1) - 1
    start = page * page_size
    end = start + page_size

    st.caption(f"共 {total} 条，第 {start + 1}-{min(end, total)} 条")
    for d in diseases[start:end]:
        _render_disease_card(d)


def page_add_disease(searcher: PestSearcher):
    """新增病害页。"""
    st.subheader("新增病害")
    with st.form("add_disease_form"):
        col1, col2 = st.columns(2)
        crop = col1.text_input("作物类型 *", placeholder="如：番茄")
        name = col2.text_input("病害名称 *", placeholder="如：早疫病")
        col3, col4 = st.columns(2)
        category = col3.selectbox("类别", ["真菌", "细菌", "病毒", "虫害", "生理性"])
        severity = col4.slider("严重度", 1, 5, 3)
        symptoms = st.text_area("症状描述", placeholder="叶片出现...")
        cause = st.text_input("病原/病因")

        st.markdown("**处理方法（药剂）**")
        drug_name = st.text_input("药剂名称", placeholder="如：80%戊唑醇")
        drug_dosage = st.text_area("用量与稀释", placeholder="如：1000倍液喷雾")

        st.markdown("**预防方法**")
        prevention = st.text_area("预防措施", placeholder="如：合理密植，及时排水")

        submitted = st.form_submit_button("提交")
        if submitted:
            if not crop or not name:
                st.error("作物类型和病害名称为必填项")
            else:
                try:
                    did = searcher.add_disease(
                        crop_type=crop, disease_name=name,
                        disease_category=category, symptoms=symptoms, cause=cause,
                        treatments=[Treatment(
                            drug_name=drug_name, dosage=drug_dosage,
                            application_method="喷雾",
                        )] if drug_name else [],
                        preventions=[Prevention(measure=prevention)] if prevention else [],
                    )
                    st.success(f"已添加：{crop} {name}（id={did}）")
                except Exception as e:
                    st.error(f"添加失败：{e}")


def page_ai_solutions(searcher: PestSearcher):
    """AI 方案审核页。"""
    st.subheader("AI 方案审核（自学习库）")
    repo = searcher.repo
    solutions = repo.list_llm_solutions(promoted_only=False, limit=50)

    if not solutions:
        st.info("自学习库为空。当用户查询未命中知识库时，DeepSeek 生成的方案会自动保存到这里。")
        return

    st.caption(f"共 {len(solutions)} 条 AI 方案（含结构化字段）")
    for sol in solutions:
        status = "已提升" if sol.promoted else "待审核"
        structured = sol.structured
        title = f"[{status}] {sol.query_text}"
        if structured:
            title += f" · {structured.diagnosis[:30]}"
        with st.expander(f"{title} ({sol.created_at})"):
            if structured:
                # 结构化展示
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.metric("置信度", f"{sol.confidence:.2%}" if sol.confidence else "-")
                    st.metric("模型", sol.model)
                with col2:
                    st.markdown(f"**诊断**：{structured.diagnosis}")
                    if structured.cause:
                        st.markdown(f"**病因**：{structured.cause}")

                if structured.treatments:
                    st.markdown("#### 处理方法")
                    df_t = pd.DataFrame([
                        {
                            "药剂/措施": t.drug_name,
                            "类型": t.drug_type,
                            "用量": t.dosage,
                            "施用方式": t.application_method,
                        }
                        for t in structured.treatments
                    ])
                    st.dataframe(df_t, use_container_width=True, hide_index=True)

                if structured.preventions:
                    st.markdown("#### 预防措施")
                    df_p = pd.DataFrame([
                        {"措施": p.measure, "时机": p.timing}
                        for p in structured.preventions
                    ])
                    st.dataframe(df_p, use_container_width=True, hide_index=True)

                if structured.notes:
                    st.info(f"**注意事项**：{structured.notes}")

                with st.expander("查看原始 JSON"):
                    st.code(sol.solution_json, language="json")
            else:
                # 向后兼容：老数据无结构化字段
                st.markdown(
                    sol.solution[:500] + "..." if len(sol.solution) > 500 else sol.solution
                )

            if not sol.promoted:
                st.divider()
                if st.button("提升为正式记录", key=f"promote_{sol.id}"):
                    # 创建正式病害记录（若结构化，则带 treatments/preventions）
                    treatments = [
                        Treatment(
                            drug_name=t.drug_name, drug_type=t.drug_type,
                            dosage=t.dosage, application_method=t.application_method,
                        )
                        for t in structured.treatments
                    ] if structured else []
                    preventions = [
                        Prevention(measure=p.measure, timing=p.timing)
                        for p in structured.preventions
                    ] if structured else []
                    did = searcher.add_disease(
                        crop_type="未知",
                        disease_name=sol.query_text,
                        symptoms=structured.diagnosis if structured else sol.query_text,
                        cause=structured.cause if structured else "",
                        treatments=treatments,
                        preventions=preventions,
                    )
                    repo.promote_llm_solution(sol.id, did)
                    st.success(f"已提升为正式记录（disease_id={did}）")
                    st.rerun()


def page_search_test(searcher: PestSearcher):
    """搜索测试页。"""
    st.subheader("语义搜索测试")
    query = st.text_input(
        "输入症状或病害名",
        placeholder="如：番茄叶子上有黄色轮纹斑点",
    )
    if query:
        with st.spinner("搜索中...（首次需加载 BGE 模型）"):
            result = searcher.search(query)
        if result.matched:
            if result.disease:
                st.success(f"命中知识库：{result.disease.crop_type} {result.disease.disease_name}（相似度 {result.similarity:.1%}）")
                _render_disease_card(result.disease)
            elif result.llm_solution:
                st.info(f"命中自学习库（相似度 {result.similarity:.1%}）")
                sol = result.llm_solution
                structured = sol.structured
                if structured:
                    st.markdown(f"**诊断**：{structured.diagnosis}")
                    if structured.cause:
                        st.markdown(f"**病因**：{structured.cause}")
                    if structured.treatments:
                        st.markdown("**处理方法**：")
                        for t in structured.treatments:
                            line = f"- {t.drug_name}（{t.drug_type}）"
                            if t.dosage:
                                line += f" · 用量 {t.dosage}"
                            if t.application_method:
                                line += f" · {t.application_method}"
                            st.markdown(line)
                    if structured.preventions:
                        st.markdown("**预防措施**：")
                        for p in structured.preventions:
                            st.markdown(f"- {p.measure}" + (f"（{p.timing}）" if p.timing else ""))
                    if structured.notes:
                        st.info(f"注意事项：{structured.notes}")
                else:
                    st.markdown(sol.solution[:1000])
        else:
            st.warning(f"未命中（最高相似度 {result.similarity:.1%}）")
            st.info("在实际使用中，此查询会触发 DeepSeek V4 Pro 生成方案并自动保存到自学习库。")


def page_import_export(searcher: PestSearcher):
    """导入导出页。"""
    st.subheader("导入 / 导出")
    repo = searcher.repo
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 导出")
        if st.button("导出为 Excel"):
            df = repo.export_diseases()
            if df.empty:
                st.warning("知识库为空")
            else:
                st.dataframe(df)
                st.download_button(
                    "下载 Excel",
                    df.to_excel(index=False, engine="openpyxl"),
                    file_name="pest_diseases.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

    with col2:
        st.markdown("#### 导入")
        uploaded = st.file_uploader("上传 Excel/CSV", type=["xlsx", "csv"])
        if uploaded and st.button("开始导入"):
            try:
                if uploaded.name.endswith(".csv"):
                    df = pd.read_csv(uploaded)
                else:
                    df = pd.read_excel(uploaded)
                count = repo.import_diseases(df)
                st.success(f"成功导入 {count} 条记录")
            except Exception as e:
                st.error(f"导入失败：{e}")


# ===== 页面入口 =====
st.title("病虫害知识库管理")

searcher = _get_searcher()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "病害列表", "新增病害", "AI 方案审核", "搜索测试", "导入导出"
])

with tab1:
    page_disease_list(searcher)
with tab2:
    page_add_disease(searcher)
with tab3:
    page_ai_solutions(searcher)
with tab4:
    page_search_test(searcher)
with tab5:
    page_import_export(searcher)
