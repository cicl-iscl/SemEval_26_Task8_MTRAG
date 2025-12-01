import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = PROJECT_ROOT / "dataset"
INDEX_ROOT   = PROJECT_ROOT / "indexes"
QUERIES_ROOT = PROJECT_ROOT / "queries data"
RUNS_ROOT    = PROJECT_ROOT / "runs"   # 新建一个目录存检索结果
RUNS_ROOT.mkdir(exist_ok=True)

DATASETS = ["clapnq", "cloud", "fiqa", "govt"]


def corpus_path(dataset: str) -> Path:
    return DATASET_ROOT / dataset / "corpus.jsonl"


def queries_path(dataset: str, variant: str = "questions") -> Path:
    """
    variant: "questions" or "lastturn"
    """
    fname = f"{dataset}_{variant}.jsonl"
    return QUERIES_ROOT / fname


def bm25_jsonl_dir(dataset: str) -> Path:
    return INDEX_ROOT / f"{dataset}-jsonl"


def bm25_lucene_dir(dataset: str) -> Path:
    return INDEX_ROOT / f"{dataset}-lucene"


def dense_index_dir(dataset: str, model_tag: str) -> Path:
    """
    model_tag: "bge-base" or "bge-icl"
    """
    return INDEX_ROOT / f"{dataset}-{model_tag}-faiss"
