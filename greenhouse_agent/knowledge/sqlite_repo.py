"""SQLite + sqlite-vec 后端实现。

表结构：
- diseases / treatments / preventions：结构化数据（常规表）
- disease_vectors / llm_solution_vectors：向量索引（sqlite-vec 虚拟表）
- llm_solutions：自学习方案（常规表）

所有数据存储在 data/pest_disease.db 单个文件中，零运维成本。
"""

import hashlib
import sqlite3
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd

from greenhouse_agent import config
from greenhouse_agent.knowledge.models import (
    Disease,
    KnowledgeStats,
    LLMSolution,
    Prevention,
    Treatment,
)
from greenhouse_agent.knowledge.repository import PestDiseaseRepository

# 向量维度（bge-small-zh = 512）
_VEC_DIM: Optional[int] = None


def _get_vec_dim() -> int:
    """延迟获取向量维度（需要先加载 BGE 模型）。"""
    global _VEC_DIM
    if _VEC_DIM is None:
        from greenhouse_agent.knowledge.embeddings import get_embedding_dimension

        _VEC_DIM = get_embedding_dimension()
    return _VEC_DIM


def _l2_to_similarity(l2_distance: float) -> float:
    """L2 距离转余弦相似度（仅对归一化向量有效）。

    L2^2 = 2(1 - cos_sim)  →  cos_sim = 1 - L2^2 / 2
    """
    sim = 1.0 - (l2_distance ** 2) / 2.0
    return max(0.0, min(1.0, sim))


def _pack_vector(vec: List[float]) -> bytes:
    """将 float 列表打包为 sqlite-vec 需要的 BLOB 格式。"""
    import struct

    return struct.pack(f"{len(vec)}f", *vec)


class SQLiteRepository(PestDiseaseRepository):
    """SQLite + sqlite-vec 实现。"""

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = str(db_path or config.PEST_DISEASE_DB_FILE)
        self._conn: Optional[sqlite3.Connection] = None

    @property
    def conn(self) -> sqlite3.Connection:
        """获取数据库连接（延迟初始化，加载 sqlite-vec 扩展）。"""
        if self._conn is not None:
            return self._conn
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        # 加载 sqlite-vec 扩展
        try:
            import sqlite_vec

            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
        except Exception as e:
            raise RuntimeError(
                f"无法加载 sqlite-vec 扩展，请运行 pip install sqlite-vec: {e}"
            ) from e
        return self._conn

    def init_schema(self) -> None:
        c = self.conn
        dim = _get_vec_dim()

        # 结构化表
        c.executescript("""
            CREATE TABLE IF NOT EXISTS diseases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                crop_type TEXT NOT NULL,
                disease_name TEXT NOT NULL,
                disease_category TEXT DEFAULT '',
                symptoms TEXT DEFAULT '',
                cause TEXT DEFAULT '',
                favorable_conditions TEXT DEFAULT '',
                season TEXT DEFAULT '',
                severity_level INTEGER DEFAULT 3,
                source TEXT DEFAULT 'manual',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS treatments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                disease_id INTEGER NOT NULL REFERENCES diseases(id) ON DELETE CASCADE,
                drug_name TEXT NOT NULL,
                drug_type TEXT DEFAULT '化学',
                dosage TEXT DEFAULT '',
                application_method TEXT DEFAULT '',
                timing TEXT DEFAULT '',
                rotation_period TEXT DEFAULT '',
                cost_estimate TEXT DEFAULT '',
                precautions TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS preventions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                disease_id INTEGER NOT NULL REFERENCES diseases(id) ON DELETE CASCADE,
                measure TEXT NOT NULL,
                timing TEXT DEFAULT '',
                frequency TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS llm_solutions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT NOT NULL,
                disease_id INTEGER REFERENCES diseases(id),
                solution TEXT NOT NULL,
                solution_json TEXT,
                model TEXT DEFAULT 'deepseek-v4-pro',
                confidence REAL,
                promoted INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS disease_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                disease_id INTEGER NOT NULL REFERENCES diseases(id) ON DELETE CASCADE,
                text_content TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                model_name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # 向量虚拟表（sqlite-vec）
        c.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS disease_vectors "
            f"USING vec0(disease_id INTEGER PRIMARY KEY, embedding FLOAT[{dim}])"
        )
        c.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS llm_solution_vectors "
            f"USING vec0(solution_id INTEGER PRIMARY KEY, embedding FLOAT[{dim}])"
        )

        # 兼容旧库：若 llm_solutions 缺少 solution_json 列，则补加
        cols = [row["name"] for row in c.execute("PRAGMA table_info(llm_solutions)").fetchall()]
        if "solution_json" not in cols:
            c.execute("ALTER TABLE llm_solutions ADD COLUMN solution_json TEXT")

        c.commit()

    # ===== 向量搜索 =====

    def search_by_vector(
        self, embedding: List[float], top_k: int = 5
    ) -> List[Tuple[int, float, str]]:
        c = self.conn
        rows = c.execute(
            "SELECT disease_id, distance FROM disease_vectors "
            "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (_pack_vector(embedding), top_k),
        ).fetchall()
        results = []
        for row in rows:
            disease_id = row["disease_id"]
            sim = _l2_to_similarity(row["distance"])
            # 取对应的文本内容
            text_row = c.execute(
                "SELECT text_content FROM disease_embeddings WHERE disease_id = ?",
                (disease_id,),
            ).fetchone()
            text = text_row["text_content"] if text_row else ""
            results.append((disease_id, sim, text))
        return results

    def search_llm_solutions_by_vector(
        self, embedding: List[float], top_k: int = 3
    ) -> List[Tuple[int, float, str]]:
        c = self.conn
        rows = c.execute(
            "SELECT solution_id, distance FROM llm_solution_vectors "
            "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (_pack_vector(embedding), top_k),
        ).fetchall()
        results = []
        for row in rows:
            sol_id = row["solution_id"]
            sim = _l2_to_similarity(row["distance"])
            text_row = c.execute(
                "SELECT solution FROM llm_solutions WHERE id = ?",
                (sol_id,),
            ).fetchone()
            text = text_row["solution"] if text_row else ""
            results.append((sol_id, sim, text))
        return results

    # ===== 病害 CRUD =====

    def create_disease(
        self,
        disease: Disease,
        embedding_text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
    ) -> int:
        c = self.conn
        now = datetime.now().isoformat(sep=" ", timespec="seconds")
        cursor = c.execute(
            """INSERT INTO diseases
               (crop_type, disease_name, disease_category, symptoms, cause,
                favorable_conditions, season, severity_level, source, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                disease.crop_type, disease.disease_name, disease.disease_category,
                disease.symptoms, disease.cause, disease.favorable_conditions,
                disease.season, disease.severity_level, disease.source, now, now,
            ),
        )
        disease_id = cursor.lastrowid

        # 插入 treatments
        for t in disease.treatments:
            c.execute(
                """INSERT INTO treatments
                   (disease_id, drug_name, drug_type, dosage, application_method,
                    timing, rotation_period, cost_estimate, precautions)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (disease_id, t.drug_name, t.drug_type, t.dosage,
                 t.application_method, t.timing, t.rotation_period,
                 t.cost_estimate, t.precautions),
            )

        # 插入 preventions
        for p in disease.preventions:
            c.execute(
                """INSERT INTO preventions (disease_id, measure, timing, frequency)
                   VALUES (?,?,?,?)""",
                (disease_id, p.measure, p.timing, p.frequency),
            )

        # 插入向量
        if embedding_text and embedding:
            text_hash = hashlib.md5(embedding_text.encode()).hexdigest()
            c.execute(
                "INSERT INTO disease_embeddings (disease_id, text_content, text_hash, model_name, created_at) "
                "VALUES (?,?,?,?,?)",
                (disease_id, embedding_text, text_hash, config.EMBEDDING_MODEL, now),
            )
            c.execute(
                "INSERT INTO disease_vectors (disease_id, embedding) VALUES (?, ?)",
                (disease_id, _pack_vector(embedding)),
            )

        c.commit()
        return disease_id

    def get_disease(self, disease_id: int) -> Optional[Disease]:
        c = self.conn
        row = c.execute(
            "SELECT * FROM diseases WHERE id = ?", (disease_id,)
        ).fetchone()
        if not row:
            return None
        disease = Disease(
            id=row["id"], crop_type=row["crop_type"], disease_name=row["disease_name"],
            disease_category=row["disease_category"], symptoms=row["symptoms"],
            cause=row["cause"], favorable_conditions=row["favorable_conditions"],
            season=row["season"], severity_level=row["severity_level"],
            source=row["source"],
        )
        # treatments
        for t in c.execute(
            "SELECT * FROM treatments WHERE disease_id = ?", (disease_id,)
        ).fetchall():
            disease.treatments.append(Treatment(
                id=t["id"], disease_id=disease_id, drug_name=t["drug_name"],
                drug_type=t["drug_type"], dosage=t["dosage"],
                application_method=t["application_method"], timing=t["timing"],
                rotation_period=t["rotation_period"], cost_estimate=t["cost_estimate"],
                precautions=t["precautions"],
            ))
        # preventions
        for p in c.execute(
            "SELECT * FROM preventions WHERE disease_id = ?", (disease_id,)
        ).fetchall():
            disease.preventions.append(Prevention(
                id=p["id"], disease_id=disease_id, measure=p["measure"],
                timing=p["timing"], frequency=p["frequency"],
            ))
        return disease

    def update_disease(self, disease: Disease) -> None:
        c = self.conn
        now = datetime.now().isoformat(sep=" ", timespec="seconds")
        c.execute(
            """UPDATE diseases SET crop_type=?, disease_name=?, disease_category=?,
               symptoms=?, cause=?, favorable_conditions=?, season=?,
               severity_level=?, updated_at=? WHERE id=?""",
            (disease.crop_type, disease.disease_name, disease.disease_category,
             disease.symptoms, disease.cause, disease.favorable_conditions,
             disease.season, disease.severity_level, now, disease.id),
        )
        c.commit()

    def delete_disease(self, disease_id: int) -> None:
        c = self.conn
        c.execute("DELETE FROM diseases WHERE id = ?", (disease_id,))
        c.execute("DELETE FROM disease_vectors WHERE disease_id = ?", (disease_id,))
        c.execute("DELETE FROM disease_embeddings WHERE disease_id = ?", (disease_id,))
        c.commit()

    def list_diseases(
        self, crop_type: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> List[Disease]:
        c = self.conn
        if crop_type:
            rows = c.execute(
                "SELECT * FROM diseases WHERE crop_type = ? ORDER BY id DESC LIMIT ? OFFSET ?",
                (crop_type, limit, offset),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM diseases ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [Disease(
            id=r["id"], crop_type=r["crop_type"], disease_name=r["disease_name"],
            disease_category=r["disease_category"], symptoms=r["symptoms"],
            cause=r["cause"], severity_level=r["severity_level"], source=r["source"],
        ) for r in rows]

    # ===== Treatment / Prevention =====

    def add_treatment(self, disease_id: int, treatment: Treatment) -> int:
        c = self.conn
        cursor = c.execute(
            """INSERT INTO treatments
               (disease_id, drug_name, drug_type, dosage, application_method,
                timing, rotation_period, cost_estimate, precautions)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (disease_id, treatment.drug_name, treatment.drug_type, treatment.dosage,
             treatment.application_method, treatment.timing, treatment.rotation_period,
             treatment.cost_estimate, treatment.precautions),
        )
        c.commit()
        return cursor.lastrowid

    def add_prevention(self, disease_id: int, prevention: Prevention) -> int:
        c = self.conn
        cursor = c.execute(
            "INSERT INTO preventions (disease_id, measure, timing, frequency) VALUES (?,?,?,?)",
            (disease_id, prevention.measure, prevention.timing, prevention.frequency),
        )
        c.commit()
        return cursor.lastrowid

    def delete_treatment(self, treatment_id: int) -> None:
        self.conn.execute("DELETE FROM treatments WHERE id = ?", (treatment_id,))
        self.conn.commit()

    def delete_prevention(self, prevention_id: int) -> None:
        self.conn.execute("DELETE FROM preventions WHERE id = ?", (prevention_id,))
        self.conn.commit()

    # ===== 自学习（LLM Solutions）=====

    def save_llm_solution(
        self,
        query_text: str,
        solution: str,
        model: str,
        embedding: List[float],
        confidence: Optional[float] = None,
        solution_json: Optional[str] = None,
    ) -> int:
        c = self.conn
        cursor = c.execute(
            "INSERT INTO llm_solutions (query_text, solution, solution_json, model, confidence) "
            "VALUES (?,?,?,?,?)",
            (query_text, solution, solution_json, model, confidence),
        )
        sol_id = cursor.lastrowid
        c.execute(
            "INSERT INTO llm_solution_vectors (solution_id, embedding) VALUES (?, ?)",
            (sol_id, _pack_vector(embedding)),
        )
        c.commit()
        return sol_id

    def list_llm_solutions(
        self, promoted_only: bool = False, limit: int = 50
    ) -> List[LLMSolution]:
        c = self.conn
        if promoted_only:
            rows = c.execute(
                "SELECT * FROM llm_solutions WHERE promoted = 1 ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM llm_solutions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [LLMSolution(
            id=r["id"], query_text=r["query_text"], disease_id=r["disease_id"],
            solution=r["solution"], solution_json=r["solution_json"],
            model=r["model"], confidence=r["confidence"],
            promoted=bool(r["promoted"]), created_at=r["created_at"],
        ) for r in rows]

    def promote_llm_solution(self, solution_id: int, disease_id: int) -> None:
        self.conn.execute(
            "UPDATE llm_solutions SET promoted = 1, disease_id = ? WHERE id = ?",
            (disease_id, solution_id),
        )
        self.conn.commit()

    def clear_llm_solutions(self) -> None:
        """清空自学习表（测试用，慎用）。"""
        c = self.conn
        c.execute("DELETE FROM llm_solution_vectors")
        c.execute("DELETE FROM llm_solutions")
        c.commit()

    # ===== 统计 =====

    def get_stats(self) -> KnowledgeStats:
        c = self.conn
        d = c.execute("SELECT COUNT(*) as n FROM diseases").fetchone()["n"]
        t = c.execute("SELECT COUNT(*) as n FROM treatments").fetchone()["n"]
        p = c.execute("SELECT COUNT(*) as n FROM preventions").fetchone()["n"]
        l = c.execute("SELECT COUNT(*) as n FROM llm_solutions").fetchone()["n"]
        pr = c.execute("SELECT COUNT(*) as n FROM llm_solutions WHERE promoted = 1").fetchone()["n"]
        return KnowledgeStats(
            disease_count=d, treatment_count=t, prevention_count=p,
            llm_solution_count=l, promoted_count=pr,
        )

    # ===== 导入导出 =====

    def export_diseases(self) -> pd.DataFrame:
        diseases = self.list_diseases(limit=10000)
        rows = []
        for d in diseases:
            treatments_text = " | ".join(t.drug_name for t in d.treatments) if d.treatments else ""
            rows.append({
                "作物": d.crop_type, "病害名称": d.disease_name,
                "类别": d.disease_category, "症状": d.symptoms,
                "病原": d.cause, "发病条件": d.favorable_conditions,
                "季节": d.season, "严重度": d.severity_level,
                "推荐药剂": treatments_text,
            })
        return pd.DataFrame(rows)

    def import_diseases(self, df: pd.DataFrame) -> int:
        from greenhouse_agent.knowledge.embeddings import encode, encode_batch

        count = 0
        texts = []
        for _, row in df.iterrows():
            text = f"{row.get('作物', '')} {row.get('病害名称', '')} {row.get('症状', '')} {row.get('病原', '')}"
            texts.append(text)

        embeddings = encode_batch(texts) if texts else []

        for idx, (_, row) in enumerate(df.iterrows()):
            disease = Disease(
                crop_type=str(row.get("作物", "")),
                disease_name=str(row.get("病害名称", "")),
                disease_category=str(row.get("类别", "")),
                symptoms=str(row.get("症状", "")),
                cause=str(row.get("病原", "")),
                favorable_conditions=str(row.get("发病条件", "")),
                season=str(row.get("季节", "")),
                severity_level=int(row.get("严重度", 3)),
            )
            # 药剂拆分
            drugs_text = str(row.get("推荐药剂", ""))
            if drugs_text:
                for drug in drugs_text.split("|"):
                    drug = drug.strip()
                    if drug:
                        disease.treatments.append(Treatment(drug_name=drug))

            self.create_disease(disease, embeddings[idx], embeddings[idx])
            count += 1

        return count

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
