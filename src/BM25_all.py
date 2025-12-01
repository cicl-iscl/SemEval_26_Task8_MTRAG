import os
import csv
import json
import random
import subprocess
from pathlib import Path

import pandas as pd
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from pyserini.search.lucene import LuceneSearcher

#=========some system setting and parameter setting======
BASE_DIR = Path(__file__).resolve().parent
DATASETS = {
    "clapnq": {
        "data_dir": BASE_DIR / "dataset" / "clapnq",
        "index_dir": BASE_DIR / "indexes" / "clapnq-lucene",
        "json_dir": BASE_DIR / "indexes" / "clapnq-jsonl",
    },
    "cloud": {
        "data_dir": BASE_DIR / "dataset" / "cloud",
        "index_dir": BASE_DIR / "indexes" / "cloud-lucene",
        "json_dir": BASE_DIR / "indexes" / "cloud-jsonl",
    },
    "fiqa": {
        "data_dir": BASE_DIR / "dataset" / "fiqa",
        "index_dir": BASE_DIR / "indexes" / "fiqa-lucene",
        "json_dir": BASE_DIR / "indexes" / "fiqa-jsonl",
    },
    "govt": {
        "data_dir": BASE_DIR / "dataset" / "govt",
        "index_dir": BASE_DIR / "indexes" / "govt-lucene",
        "json_dir": BASE_DIR / "indexes" / "govt-jsonl",
    },
}
SPLIT_SOURCE = "train"
TRAIN_RATIO = 0.8
TOPK = 10
GRID_K1 = [0.6, 0.9, 1.2, 1.5]
GRID_B = [0.2, 0.4, 0.6, 0.8]
SEED = 42
#==========================================

random.seed(SEED)

"""
create contents for pyserini, dict is the corpus, return string
"""
def build_contents(d: dict) -> str:
    title = (d.get("title") or "").strip()
    text = (d.get("text") or "").strip()
    if title and text.startswith(title):
        return text
    merged = (f"{title}\n{text}").strip()
    return merged or title or text

"""
transfer beir_repo corpus to pyserini jsoncollection
"""
def prepare_jsonl(json_dir:Path, data_dir: Path) -> None:
    json_dir.mkdir(parents=True, exist_ok=True)
    out_file = json_dir / "docs.jsonl"
    if out_file.exists():
        return
    corpus = data_dir/ "corpus.jsonl"
    n = 0
    with open(corpus, "r", encoding="utf-8") as fin, open(out_file, "w", encoding="utf-8") as fout:
        for line in fin:
            d = json.loads(line)
            did = d.get("_id") or d.get("id")
            if not did:
                continue
            contents = build_contents(d)
            fout.write(json.dumps({"id": str(did), "contents": contents}, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n:,} docs to {out_file}")

"""
create index using lucene, if the indexes are already exist, just skip it
"""
def ensure_lucene_index(json_dir:Path, index_dir:Path) -> None:
    if index_dir.exists() and any(p.name.startswith("segments") for p in index_dir.iterdir()):
        return #the index is already made
    cmd = f'''python -m pyserini.index.lucene \
      --collection JsonCollection \
      --input "{json_dir}" \
      --index "{index_dir}" \
      --generator DefaultLuceneDocumentGenerator \
      --threads 8 \
      --storePositions --storeDocvectors --storeRaw'''
    print(f"Building lucene index {index_dir}")
    subprocess.run(cmd, shell=True, check=True)
    print(f"finished building")

"""
split the queries to train and dev sets
"""
def memory_split_queries(queries_all: dict, qrels_all: dict, train_ration: float):
    qids = list(qrels_all.keys())
    random.shuffle(qids)
    n_tr = int(train_ration * len(qids))
    train_ids = qids[:n_tr]
    dev_ids = qids[n_tr:]
    queries_train = {qid: queries_all[qid] for qid in train_ids}
    queries_dev = {qid: queries_all[qid] for qid in dev_ids}
    qrels_train = {qid: qrels_all[qid] for qid in train_ids}
    qrels_dev = {qid:qrels_all[qid] for qid in dev_ids}
    return queries_train, queries_dev, qrels_train, qrels_dev

"""
searching base on BM25
"""
def search_with_bm25(index_dir: Path, queries: dict, k1:float, b:float, k: int) -> dict:
    searcher = LuceneSearcher(str(index_dir))
    searcher.set_bm25(k1, b)
    results = {}
    for qid, qtext in queries.items():
        hits = searcher.search(qtext, k=k)
        results[qid] = {h.docid: float(h.score) for h in hits}
    return results

"""
do the evaluate for ndcg, recall
"""
def evaluate(qrels: dict, results:dict, k_values):
    evaluator = EvaluateRetrieval(None)
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, k_values = k_values)
    return ndcg, recall

"""
tune the hyperparameters for BM25 on train using Grid Search
"""
def tune_on_train(index_dir:Path, queries_tr:dict, qrels_tr: dict):
    best = None
    for k1 in GRID_K1:
        for b in GRID_B:
            rlt = search_with_bm25(index_dir, queries_tr, k1, b, k=TOPK)
            ndcg, recall = evaluate(qrels_tr, rlt, [1,3,5,10])
            score = (ndcg['NDCG@3'], recall['Recall@3'])
            if best is None or score > best[0]:
                best = [score, k1, b]
            print(f"  k1={k1:.2f}, b={b:.2f}: nDCG@3={ndcg['NDCG@3']:.4f}, recall@3={recall['Recall@3']:.4f}")
    (_, k1_best, b_best) = best
    print(f"Best on train k1:{k1_best}, b:{b_best}")
    return k1_best, b_best

def main():
    summary = []

    for name, cfg in DATASETS.items():
        data_dir  = cfg["data_dir"]
        json_dir  = cfg["json_dir"]
        index_dir = cfg["index_dir"]

        print("\n" + "="*80)
        print(f"Dataset: {name}")

        # indexing
        prepare_jsonl(json_dir, data_dir)
        ensure_lucene_index(json_dir, index_dir)

        # read data
        corpus_all, queries_all, qrels_all = GenericDataLoader(str(data_dir)).load(split=SPLIT_SOURCE)
        print(f"Loaded: |D|={len(corpus_all):,} |Q|={len(queries_all):,} |QRels|={len(qrels_all):,}")

        # 4) split dataset
        queries_tr, queries_dv, qrels_tr, qrels_dv = memory_split_queries(queries_all, qrels_all, TRAIN_RATIO)
        print(f"SPLIT → TRAIN Q={len(queries_tr):,}, DEV2 Q={len(queries_dv):,}")

        # 5) tune parameter
        k1_star, b_star = tune_on_train(index_dir, queries_tr, qrels_tr)

        # 6) evaluate from dev set
        res_dev = search_with_bm25(index_dir, queries_dv, k1_star, b_star, k=TOPK)
        ks = [1, 3, 5, 10]
        ndcg_dev, recall_dev = evaluate(qrels_dv, res_dev, ks)
        ndcg10 = ndcg_dev['NDCG@10']
        rec10  = recall_dev['Recall@10']
        print(f"DEV2  nDCG@10={ndcg10:.4f}  Recall@10={rec10:.4f}")

        row = {
            "dataset": name,
            "k1": k1_star,
            "b": b_star,
        }
        for k in ks:
            row[f"nDCG@{k}"] = round(ndcg_dev[f'NDCG@{k}'], 4)
            row[f"Recall@{k}"] = round(recall_dev[f'Recall@{k}'], 4)

        summary.append(row)


    # print the summary and results

    print("\n" + "="*80)
    print("Summary")
    print(f"{'dataset':10s}  {'k1':>4s}  {'b':>4s}  {'nDCG@1':>8s} {'nDCG@3':>8s} {'nDCG@5':>8s} {'nDCG@10':>8s}  {'R@1':>10s} {'R@3':>10s} {'R@5':>10s} {'R@10':>10s}")
    for row in summary:
        print(f"{row['dataset']:10s}  {row['k1']:>4}  {row['b']:>4}  {row['nDCG@1']:>8} {row['nDCG@3']:>8} {row['nDCG@5']:>8} {row['nDCG@10']:>8} {row['Recall@1']:>10} {row['Recall@3']:>10} {row['Recall@5']:>10} {row['Recall@10']:>10}")

    ndcg1_values = [d["nDCG@1"] for d in summary]
    ndcg1_macro = sum(ndcg1_values) / len(ndcg1_values)
    print(f"nDCG@1 marco: {ndcg1_macro}")
    ndcg3_values = [d["nDCG@3"] for d in summary]
    ndcg3_macro = sum(ndcg3_values) / len(ndcg3_values)
    print(f"nDCG@3 marco: {ndcg3_macro}")
    ndcg5_values = [d["nDCG@5"] for d in summary]
    ndcg5_macro = sum(ndcg5_values) / len(ndcg5_values)
    print(f"nDCG@5 marco: {ndcg5_macro}")
    ndcg10_values = [d["nDCG@10"] for d in summary]
    ndcg10_macro = sum(ndcg10_values) / len(ndcg10_values)
    print(f"nDCG@10 marco: {ndcg10_macro}")
    recall1_val = [d["Recall@1"] for d in summary]
    recall1_macro = sum(recall1_val) / len(recall1_val)
    print(f"Recall@1 marco: {recall1_macro}")
    recall3_val = [d["Recall@3"] for d in summary]
    recall3_macro = sum(recall3_val) / len(recall3_val)
    print(f"Recall@3 marco: {recall3_macro}")
    recall5_val = [d["Recall@5"] for d in summary]
    recall5_macro = sum(recall5_val) / len(recall5_val)
    print(f"Recall@5 marco: {recall5_macro}")
    recall10_val = [d["Recall@10"] for d in summary]
    recall10_macro = sum(recall10_val) / len(recall10_val)
    print(f"Recall@10 marco: {recall10_macro}")
    df_results = pd.DataFrame(summary)
    markdown_table = df_results.to_markdown(index=False)
    print(markdown_table)

if __name__ == "__main__":
    main()
