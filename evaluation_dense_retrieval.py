"""
read the data to beir format and evaluate it by beir evaluation
"""
from pathlib import Path
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.evaluation import EvaluateRetrieval

BASE_DIR = Path(".").resolve()

MODEL_NAME = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 64
embedding_model = models.SentenceBERT(MODEL_NAME)
dres_model = DRES(embedding_model, BATCH_SIZE)
retriever = EvaluateRetrieval(dres_model, score_function="cos_sim")
K_VALUES = [1, 3, 5, 10]
SPLIT = "train"

def read_to_beir(DircName:str):
    corpus, queries, qrels = GenericDataLoader(
        data_folder=str(BASE_DIR/"dataset"/DircName)
    ).load(split=SPLIT)
    return corpus, queries, qrels

"""
DircName is the directory name for the whole benchmark
"""
def evaluate(DircName:str):
    corpus, queries, qrels = read_to_beir(DircName)
    results = retriever.retrieve(corpus, queries)
    print("start evaluation")
    ndcg, _, recall, precision = retriever.evaluate(qrels, results, K_VALUES)
    return {"ndcg": ndcg, "recall": recall}