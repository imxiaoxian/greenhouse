#!/usr/bin/env python
# coding: utf-8
"""从旧版 SQLite 数据库迁移到企业级知识库。

旧版表结构：pest_disease_methods(disease_name, treatment_method, prevention_method)
新版表结构：diseases + treatments + preventions + disease_embeddings（含 BGE 向量）

此脚本会：
1. 读取旧表中的所有记录
2. 解析作物类型（从病害名前缀提取）
3. 通过 PestSearcher 写入新 schema（自动生成向量索引）
4. 报告迁移结果

用法：
    python scripts/migrate_pest_db.py
"""

import sqlite3
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from greenhouse_agent import config  # noqa: E402
from greenhouse_agent.knowledge.models import Prevention, Treatment  # noqa: E402
from greenhouse_agent.knowledge.searcher import PestSearcher  # noqa: E402

# 旧 DB 路径（迁移前备份）
OLD_DB = config.PEST_DISEASE_DB_FILE


def _extract_crop(disease_name: str) -> tuple:
    """从病害名中提取作物类型。

    例如 "西瓜细菌性叶斑病" → ("西瓜", "细菌性叶斑病")
    常见作物前缀：西瓜/茄子/牡丹/蔷薇/番茄/甜瓜/辣椒/黄瓜/花椰菜/马铃薯/苹果/南瓜/草莓
    """
    crops = ["西瓜", "茄子", "牡丹", "蔷薇", "番茄", "甜瓜", "辣椒",
             "黄瓜", "花椰菜", "马铃薯", "苹果", "南瓜", "草莓", "瓜类"]
    for c in crops:
        if disease_name.startswith(c):
            return c, disease_name[len(c):]
    return "其他", disease_name


def main():
    if not OLD_DB.exists():
        print(f"数据库文件不存在: {OLD_DB}")
        print("请先运行 init_pest_db.py 创建新数据库。")
        return

    # 检查旧表是否存在
    conn = sqlite3.connect(str(OLD_DB))
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='pest_disease_methods'"
        )
        if not cursor.fetchone():
            print("旧表 pest_disease_methods 不存在（可能已经是新 schema）。")
            return

        cursor.execute("SELECT disease_name, treatment_method, prevention_method FROM pest_disease_methods")
        old_records = cursor.fetchall()
    finally:
        conn.close()

    print(f"从旧表读取到 {len(old_records)} 条记录")
    print()

    # 备份旧 DB
    backup = OLD_DB.with_suffix(".db.bak")
    if not backup.exists():
        import shutil
        shutil.copy2(OLD_DB, backup)
        print(f"旧数据库已备份: {backup}")

    # 删除旧 DB，重建新 schema
    OLD_DB.unlink()
    print("旧数据库已删除，准备创建新 schema...\n")

    searcher = PestSearcher()
    searcher.init()

    migrated = 0
    for disease_name, treatment_text, prevention_text in old_records:
        crop, name = _extract_crop(disease_name)
        disease_id = searcher.add_disease(
            crop_type=crop,
            disease_name=name,
            disease_category="",
            symptoms="",
            cause="",
            treatments=[Treatment(
                drug_name=treatment_text[:50] + "..." if len(treatment_text) > 50 else treatment_text,
                drug_type="化学",
                dosage=treatment_text,
                application_method="喷雾",
            )],
            preventions=[Prevention(measure=prevention_text)],
        )
        migrated += 1
        print(f"  [{migrated:2d}] {crop} {name} → id={disease_id}")

    print(f"\n迁移完成：{migrated} 条记录已导入新 schema（含 BGE 向量索引）")

    # 验证搜索
    print("\n=== 搜索验证 ===")
    test_query = old_records[0][0] if old_records else "番茄病害"
    result = searcher.search(test_query)
    if result.matched and result.disease:
        print(f"命中: {result.disease.crop_type} {result.disease.disease_name} (相似度: {result.similarity:.1%})")
    else:
        print(f"未命中（最高相似度: {result.similarity:.1%}）")


if __name__ == "__main__":
    main()
