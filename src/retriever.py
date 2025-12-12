from typing import Dict

import faiss
import numpy as np
import json, re
import ollama
from pathlib import Path

from src.run_retrieval_eval import load_qrels, evaluate

BASE_DIR = Path("..").resolve()
CORPUS_PATH = f"{BASE_DIR}/dataset/clapnq/corpus.jsonl"
INDEX_DIR   = f"{BASE_DIR}/indexes/clapnq-qwen-ollama-faiss"
INDEX_PATH  = f"{INDEX_DIR}/index.faiss"
EMB_PATH    = f"{INDEX_DIR}/emb.npy"
QUERIES_PATH = f"{BASE_DIR}/history_selected_rewrite_queries/rewritten_last_turn_qwen3_30B.jsonl"
QRELS_PATH = f"{BASE_DIR}/dataset/clapnq/qrels/train.tsv"

corpus_id = []
with open(CORPUS_PATH, "r") as f:
    for line in f:
        obj = json.loads(line)
        corpus_id.append(obj["_id"])

index = faiss.read_index(INDEX_PATH)

MODEL = "qwen3-embedding:4b"

def encode_query(text):
    emb = ollama.embeddings(model=MODEL, prompt=text)["embedding"]
    return np.array(emb, dtype="float32").reshape(1, -1)

def retrieve(QUERIES_PATH, k:int = 10) -> Dict[str, Dict[str, float]]:
    USER_LINE_RE = re.compile(r'^\|user\|\s*:\s*(.*)\s*$', re.M)
    result = {}
    with open(QUERIES_PATH, "r") as f:
        for line in f:
            obj = json.loads(line)
            qid = obj["_id"]
            query_list = USER_LINE_RE.findall(obj.get("text", ""))
            if not query_list:
                continue
            query = query_list[0]
            encoded_query = encode_query(query)
            l2_query = faiss.normalize_L2(encoded_query)
            distances, indices = index.search(l2_query, k)
            hits = {}
            for idx, score in zip(indices[0], distances[0]):
                docid = corpus_id[idx]
                hits[docid] = float(score)

            result[qid] = hits
    return result


if __name__ == "__main__":
    res = retrieve(QUERIES_PATH)
    qrels = load_qrels(QRELS_PATH)
    scores, ndcg, _map, recall, precision = evaluate(qrels, res, [1, 3, 5, 10])
    print(ndcg, recall)




