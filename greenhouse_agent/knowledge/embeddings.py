"""BGE 本地 Embedding 服务。

使用 sentence-transformers 加载 BAAI/bge-small-zh-v1.5 模型，
将中文文本编码为 512 维向量，用于病虫害知识库语义检索。

模型约 100MB，首次加载自动从 HuggingFace 下载到 ``data/bge_model/`` 目录
（使用 copy 模式而非 symlink，避免 Windows 权限问题）。
"""

import threading
from pathlib import Path
from typing import List, Optional

from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

from greenhouse_agent import config

# 单例
_model: Optional[SentenceTransformer] = None
_lock = threading.Lock()


def _get_model() -> SentenceTransformer:
    """获取 BGE 模型单例（线程安全）。

    将模型下载到项目内 ``data/bge_model/`` 目录，
    避免 Windows 下默认缓存目录的符号链接权限问题。
    """
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        # 本地缓存目录（项目内，避免 Windows symlink 权限问题）
        local_dir = config.DATA_DIR / "bge_model"
        local_dir.mkdir(parents=True, exist_ok=True)
        # 若本地目录已有完整模型（modules.json 存在），直接加载；否则下载
        if (local_dir / "modules.json").exists():
            model_path = str(local_dir)
        else:
            model_path = snapshot_download(
                repo_id=config.EMBEDDING_MODEL,
                local_dir=str(local_dir),
            )
        _model = SentenceTransformer(
            model_path,
            device="cpu",
        )
    return _model


def get_embedding_dimension() -> int:
    """返回模型输出维度（bge-small-zh = 512）。"""
    return _get_model().get_embedding_dimension()


def encode(text: str) -> List[float]:
    """将单条文本编码为向量。"""
    model = _get_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()


def encode_batch(texts: List[str]) -> List[List[float]]:
    """批量编码文本为向量。"""
    model = _get_model()
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vecs.tolist()
