"""PostgreSQL + pgvector 后端实现。

适用于多用户并发、大数据量的企业级部署。
需要 PostgreSQL 14+ 并安装 pgvector 扩展。

配置方式（.env）：
    KNOWLEDGE_DB_BACKEND=postgres
    PG_HOST=127.0.0.1
    PG_PORT=5432
    PG_DATABASE=greenhouse
    PG_USER=postgres
    PG_PASSWORD=your_password
"""

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

_VEC_DIM: Optional[int] = None


def _get_vec_dim() -> int:
    global _VEC_DIM
    if _VEC_DIM is None:
        from greenhouse_agent.knowledge.embeddings import get_embedding_dimension

        _VEC_DIM = get_embedding_dimension()
    return _VEC_DIM


def _vec_to_pg(vec: List[float]) -> str:
    """将向量转为 PostgreSQL vector 字面量格式 '[0.1,0.2,...]'。"""
    return "[" + ",".join(f"{v:.7f}" for v in vec) + "]"


class PostgresRepository(PestDiseaseRepository):
    """PostgreSQL + pgvector 实现。"""

    def __init__(self):
        self._pool = None

    @property
    def conn(self):
        """获取连接（每次从连接池取出）。"""
        if self._pool is None:
            import psycopg2.pool

            self._pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1, maxconn=5,
                host=config.PG_HOST, port=config.PG_PORT,
                dbname=config.PG_DATABASE, user=config.PG_USER,
                password=config.PG_PASSWORD,
            )
        return self._pool.getconn()

    def _release(self, conn):
        """归还连接。"""
        if self._pool:
            self._pool.putconn(conn)

    def init_schema(self) -> None:
        conn = self.conn
        try:
            with conn.cursor() as cur:
                dim = _get_vec_dim()
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS diseases (
                        id SERIAL PRIMARY KEY,
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
                        id SERIAL PRIMARY KEY,
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
                        id SERIAL PRIMARY KEY,
                        disease_id INTEGER NOT NULL REFERENCES diseases(id) ON DELETE CASCADE,
                        measure TEXT NOT NULL,
                        timing TEXT DEFAULT '',
                        frequency TEXT DEFAULT ''
                    );

                    CREATE TABLE IF NOT EXISTS disease_embeddings (
                        id SERIAL PRIMARY KEY,
                        disease_id INTEGER NOT NULL REFERENCES diseases(id) ON DELETE CASCADE,
                        text_content TEXT NOT NULL,
                        text_hash TEXT NOT NULL,
                        embedding vector(%s) NOT NULL,
                        model_name TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE TABLE IF NOT EXISTS llm_solutions (
                        id SERIAL PRIMARY KEY,
                        query_text TEXT NOT NULL,
                        disease_id INTEGER REFERENCES diseases(id),
                        solution TEXT NOT NULL,
                        solution_json TEXT,
                        model TEXT DEFAULT 'deepseek-v4-pro',
                        confidence REAL,
                        promoted INTEGER DEFAULT 0,
                        embedding vector(%s),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """, (dim, dim))

                # 向量搜索索引（ivfflat）
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_disease_emb
                    ON disease_embeddings USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_llm_sol_emb
                    ON llm_solutions USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """)
            conn.commit()
        finally:
            self._release(conn)

    # ===== 向量搜索 =====

    def search_by_vector(
        self, embedding: List[float], top_k: int = 5
    ) -> List[Tuple[int, float, str]]:
        conn = self.conn
        try:
            vec_str = _vec_to_pg(embedding)
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT d.disease_id, 1 - (d.embedding <=> %s::vector) as similarity,
                              d.text_content
                       FROM disease_embeddings d
                       ORDER BY d.embedding <=> %s::vector
                       LIMIT %s""",
                    (vec_str, vec_str, top_k),
                )
                return [(r[0], float(r[1]), r[2]) for r in cur.fetchall()]
        finally:
            self._release(conn)

    def search_llm_solutions_by_vector(
        self, embedding: List[float], top_k: int = 3
    ) -> List[Tuple[int, float, str]]:
        conn = self.conn
        try:
            vec_str = _vec_to_pg(embedding)
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT id, 1 - (embedding <=> %s::vector) as similarity, solution
                       FROM llm_solutions
                       WHERE embedding IS NOT NULL
                       ORDER BY embedding <=> %s::vector
                       LIMIT %s""",
                    (vec_str, vec_str, top_k),
                )
                return [(r[0], float(r[1]), r[2]) for r in cur.fetchall()]
        finally:
            self._release(conn)

    # ===== 病害 CRUD =====

    def create_disease(
        self, disease: Disease,
        embedding_text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
    ) -> int:
        conn = self.conn
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO diseases
                       (crop_type, disease_name, disease_category, symptoms, cause,
                        favorable_conditions, season, severity_level, source)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                    (disease.crop_type, disease.disease_name, disease.disease_category,
                     disease.symptoms, disease.cause, disease.favorable_conditions,
                     disease.season, disease.severity_level, disease.source),
                )
                disease_id = cur.fetchone()[0]
                for t in disease.treatments:
                    cur.execute(
                        """INSERT INTO treatments
                           (disease_id, drug_name, drug_type, dosage, application_method,
                            timing, rotation_period, cost_estimate, precautions)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                        (disease_id, t.drug_name, t.drug_type, t.dosage,
                         t.application_method, t.timing, t.rotation_period,
                         t.cost_estimate, t.precautions),
                    )
                for p in disease.preventions:
                    cur.execute(
                        "INSERT INTO preventions (disease_id, measure, timing, frequency) VALUES (%s,%s,%s,%s)",
                        (disease_id, p.measure, p.timing, p.frequency),
                    )
                if embedding_text and embedding:
                    import hashlib
                    text_hash = hashlib.md5(embedding_text.encode()).hexdigest()
                    cur.execute(
                        """INSERT INTO disease_embeddings (disease_id, text_content, text_hash, embedding, model_name)
                           VALUES (%s,%s,%s,%s::vector,%s)""",
                        (disease_id, embedding_text, text_hash, _vec_to_pg(embedding), config.EMBEDDING_MODEL),
                    )
            conn.commit()
            return disease_id
        finally:
            self._release(conn)

    def get_disease(self, disease_id: int) -> Optional[Disease]:
        conn = self.conn
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM diseases WHERE id = %s", (disease_id,))
                row = cur.fetchone()
                if not row:
                    return None
                disease = Disease(
                    id=row[0], crop_type=row[1], disease_name=row[2],
                    disease_category=row[3], symptoms=row[4], cause=row[5],
                    favorable_conditions=row[6], season=row[7],
                    severity_level=row[8], source=row[9],
                )
                cur.execute("SELECT * FROM treatments WHERE disease_id = %s", (disease_id,))
                for t in cur.fetchall():
                    disease.treatments.append(Treatment(
                        id=t[0], disease_id=disease_id, drug_name=t[2], drug_type=t[3],
                        dosage=t[4], application_method=t[5], timing=t[6],
                        rotation_period=t[7], cost_estimate=t[8], precautions=t[9],
                    ))
                cur.execute("SELECT * FROM preventions WHERE disease_id = %s", (disease_id,))
                for p in cur.fetchall():
                    disease.preventions.append(Prevention(
                        id=p[0], disease_id=disease_id, measure=p[2],
                        timing=p[3], frequency=p[4],
                    ))
            return disease
        finally:
            self._release(conn)

    def update_disease(self, disease: Disease) -> None:
        conn = self.conn
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """UPDATE diseases SET crop_type=%s, disease_name=%s, disease_category=%s,
                       symptoms=%s, cause=%s, favorable_conditions=%s, season=%s,
                       severity_level=%s, updated_at=CURRENT_TIMESTAMP WHERE id=%s""",
                    (disease.crop_type, disease.disease_name, disease.disease_category,
                     disease.symptoms, disease.cause, disease.favorable_conditions,
                     disease.season, disease.severity_level, disease.id),
                )
            conn.commit()
        finally:
            self._release(conn)

    def delete_disease(self, disease_id: int) -> None:
        conn = self.conn
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM diseases WHERE id = %s", (disease_id,))
            conn.commit()
        finally:
            self._release(conn)

    def list_diseases(
        self, crop_type: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> List[Disease]:
        conn = self.conn
        try:
            with conn.cursor() as cur:
                if crop_type:
                    cur.execute(
                        "SELECT * FROM diseases WHERE crop_type = %s ORDER BY id DESC LIMIT %s OFFSET %s",
                        (crop_type, limit, offset),
                    )
                else:
                    cur.execute(
                        "SELECT * FROM diseases ORDER BY id DESC LIMIT %s OFFSET %s",
                        (limit, offset),
                    )
                return [Disease(
                    id=r[0], crop_type=r[1], disease_name=r[2],
                    disease_category=r[3], symptoms=r[4], cause=r[5],
                    severity_level=r[8], source=r[9],
                ) for r in cur.fetchall()]
        finally:
            self._release(conn)

    # ===== Treatment / Prevention =====

    def add_treatment(self, disease_id: int, treatment: Treatment) -> int:
        conn = self.conn
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO treatments
                       (disease_id, drug_name, drug_type, dosage, application_method,
                        timing, rotation_period, cost_estimate, precautions)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                    (disease_id, treatment.drug_name, treatment.drug_type, treatment.dosage,
                     treatment.application_method, treatment.timing, treatment.rotation_period,
                     treatment.cost_estimate, treatment.precautions),
                )
                rid = cur.fetchone()[0]
            conn.commit()
            return rid
        finally:
            self._release(conn)

    def add_prevention(self, disease_id: int, prevention: Prevention) -> int:
        conn = self.conn
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO preventions (disease_id, measure, timing, frequency) VALUES (%s,%s,%s,%s) RETURNING id",
                    (disease_id, prevention.measure, prevention.timing, prevention.frequency),
                )
                rid = cur.fetchone()[0]
            conn.commit()
            return rid
        finally:
            self._release(conn)

    def delete_treatment(self, treatment_id: int) -> None:
        conn = self.conn
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM treatments WHERE id = %s", (treatment_id,))
            conn.commit()
        finally:
            self._release(conn)

    def delete_prevention(self, prevention_id: int) -> None:
        conn = self.conn
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM preventions WHERE id = %s", (prevention_id,))
            conn.commit()
        finally:
            self._release(conn)

    # ===== 自学习 =====

    def save_llm_solution(
        self, query_text: str, solution: str, model: str,
        embedding: List[float], confidence: Optional[float] = None,
        solution_json: Optional[str] = None,
    ) -> int:
        conn = self.conn
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO llm_solutions
                       (query_text, solution, solution_json, model, confidence, embedding)
                       VALUES (%s,%s,%s,%s,%s,%s::vector) RETURNING id""",
                    (query_text, solution, solution_json, model, confidence, _vec_to_pg(embedding)),
                )
                rid = cur.fetchone()[0]
            conn.commit()
            return rid
        finally:
            self._release(conn)

    def list_llm_solutions(
        self, promoted_only: bool = False, limit: int = 50
    ) -> List[LLMSolution]:
        conn = self.conn
        try:
            with conn.cursor() as cur:
                if promoted_only:
                    cur.execute(
                        "SELECT * FROM llm_solutions WHERE promoted = 1 ORDER BY id DESC LIMIT %s",
                        (limit,),
                    )
                else:
                    cur.execute(
                        "SELECT * FROM llm_solutions ORDER BY id DESC LIMIT %s", (limit,)
                    )
                return [LLMSolution(
                    id=r[0], query_text=r[1], disease_id=r[2], solution=r[3], solution_json=r[4],
                    model=r[5], confidence=r[6], promoted=bool(r[7]), created_at=r[8],
                ) for r in cur.fetchall()]
        finally:
            self._release(conn)

    def promote_llm_solution(self, solution_id: int, disease_id: int) -> None:
        conn = self.conn
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE llm_solutions SET promoted = 1, disease_id = %s WHERE id = %s",
                    (disease_id, solution_id),
                )
            conn.commit()
        finally:
            self._release(conn)

    def clear_llm_solutions(self) -> None:
        """清空自学习表（测试用，慎用）。"""
        conn = self.conn
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM llm_solution_vectors")
                cur.execute("DELETE FROM llm_solutions")
            conn.commit()
        finally:
            self._release(conn)

    # ===== 统计 =====

    def get_stats(self) -> KnowledgeStats:
        conn = self.conn
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM diseases")
                d = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM treatments")
                t = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM preventions")
                p = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM llm_solutions")
                l = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM llm_solutions WHERE promoted = 1")
                pr = cur.fetchone()[0]
            return KnowledgeStats(
                disease_count=d, treatment_count=t, prevention_count=p,
                llm_solution_count=l, promoted_count=pr,
            )
        finally:
            self._release(conn)

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
        from greenhouse_agent.knowledge.embeddings import encode_batch

        texts = []
        for _, row in df.iterrows():
            text = f"{row.get('作物', '')} {row.get('病害名称', '')} {row.get('症状', '')} {row.get('病原', '')}"
            texts.append(text)
        embeddings = encode_batch(texts) if texts else []
        count = 0
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
            drugs_text = str(row.get("推荐药剂", ""))
            if drugs_text:
                for drug in drugs_text.split("|"):
                    drug = drug.strip()
                    if drug:
                        disease.treatments.append(Treatment(drug_name=drug))
            self.create_disease(disease, texts[idx], embeddings[idx])
            count += 1
        return count
