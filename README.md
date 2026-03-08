# Multi-Turn RAG System (SemEval Task 8: MTRAG)

## Overview

This repository implements a multi-turn Retrieval-Augmented Generation (RAG) system for conversational question answering.

The system is designed to retrieve relevant evidence from a document corpus and generate grounded responses in multi-turn conversations. 
It was developed for the Multi-Turn RAG benchmark in SemEval.

Our pipeline consists of:

- history-aware query rewriting
- dense retrieval using embedding models
- cross-encoder reranking
- LoRA-adapted generator for answer generation

## System Architecture

Our system follows a four-stage pipeline:

1. **Query Rewriting**: Reformulates the user query using conversation history.

2. **Dense Retrieval**: Retrieves candidate passages using embedding similarity.

3. **Reranking**: A cross-encoder reranks retrieved passages to improve evidence quality.

4. **Answer Generation**: A LoRA-adapted LLM generates the final response grounded in retrieved passages.

## Models

- Embedding model: Qwen3 embedding
- Reranker: cross-encoder/ms-marco-MiniLM-L12-v2
- Generator: Qwen3 LLM with LoRA fine-tuning

## Dataset

We use the official dataset provided by the SemEval MTRAG benchmark.

The dataset contains multi-turn conversational questions paired with evidence passages and reference answers.

Data is split into:

- training
- development
- test set (released later by the organizers)  

The test set is not included in this repository because it is distributed privately by the task organizers.

## Evaluation

Retrieval quality is evaluated using:

- nDCG@k
- Recall@k

Answer quality is evaluated using:

- RL_agg
- LLM-based evaluation metrics

For more details about the evaluation protocol and metrics, please refer to the MTRAG benchmark paper.

### Evaluation of Dense Indexing(BGE)   

|index|dataset|k|nDCG|Recall|
|---|---|---|---|---|
|0|clapnq|1|0\.42308|0\.15585|
|1|clapnq|3|0\.36844|0\.33322|
|2|clapnq|5|0\.40192|0\.43843|
|3|clapnq|10|0\.45971|0\.57165|
|4|cloud|1|0\.23936|0\.12668|
|5|cloud|3|0\.24299|0\.24628|
|6|cloud|5|0\.27315|0\.31055|
|7|cloud|10|0\.30844|0\.38999|
|8|fiqa|1|0\.25|0\.10255|
|9|fiqa|3|0\.21926|0\.19507|
|10|fiqa|5|0\.24345|0\.2621|
|11|fiqa|10|0\.27749|0\.34284|
|12|govt|1|0\.28358|0\.12761|
|13|govt|3|0\.27202|0\.26672|
|14|govt|5|0\.30335|0\.33867|
|15|govt|10|0\.35649|0\.46446|

### Evaluation of BM25

| dataset   |   k1 |   b |   nDCG@1 |   Recall@1 |   nDCG@3 |   Recall@3 |   nDCG@5 |   Recall@5 |   nDCG@10 |   Recall@10 |
|:----------|-----:|----:|---------:|-----------:|---------:|-----------:|---------:|-----------:|----------:|------------:|
| clapnq    |  1.5 | 0.4 |   0.2143 |     0.0571 |   0.2021 |     0.1702 |   0.2617 |     0.3409 |    0.3    |      0.4302 |
| cloud     |  1.5 | 0.2 |   0.2105 |     0.0921 |   0.1974 |     0.1974 |   0.2141 |     0.2303 |    0.2415 |      0.2895 |
| fiqa      |  1.2 | 0.4 |   0.1389 |     0.044  |   0.1295 |     0.125  |   0.1619 |     0.1991 |    0.1812 |      0.2454 |
| govt      |  1.5 | 0.8 |   0.2683 |     0.1138 |   0.2569 |     0.2378 |   0.2713 |     0.2886 |    0.31   |      0.3825 |

Qwen2.5 7B with history selected evaluation:  
{'ndcg': {'NDCG@1': 0.39423, 'NDCG@3': 0.36685, 'NDCG@5': 0.38838, 'NDCG@10': 0.44656}, 'recall': {'Recall@1': 0.15128, 'Recall@3': 0.33915, 'Recall@5': 0.41443, 'Recall@10': 0.55185}}  

Qwen2.5 7B without selection:  
{'ndcg': {'NDCG@1': 0.375, 'NDCG@3': 0.34785, 'NDCG@5': 0.38127, 'NDCG@10': 0.42871}, 'recall': {'Recall@1': 0.14447, 'Recall@3': 0.31872, 'Recall@5': 0.41836, 'Recall@10': 0.5283}}


mixtral 8x7b  
{'ndcg': {'NDCG@1': 0.40865, 'NDCG@3': 0.37001, 'NDCG@5': 0.40704, 'NDCG@10': 0.45717}, 'recall': {'Recall@1': 0.1595, 'Recall@3': 0.34705, 'Recall@5': 0.4507, 'Recall@10': 0.56536}}

Qwen3 14b  
{'ndcg': {'NDCG@1': 0.41827, 'NDCG@3': 0.3613, 'NDCG@5': 0.39691, 'NDCG@10': 0.45795}, 'recall': {'Recall@1': 0.15389, 'Recall@3': 0.32621, 'Recall@5': 0.43143, 'Recall@10': 0.57555}}

Gemma 12b  
{'ndcg': {'NDCG@1': 0.42788, 'NDCG@3': 0.36751, 'NDCG@5': 0.39516, 'NDCG@10': 0.45602}, 'recall': {'Recall@1': 0.17056, 'Recall@3': 0.32541, 'Recall@5': 0.41208, 'Recall@10': 0.55443}}

Using the Qwen3_4B embedding with Qwen3_30B query rewrite the evaluation is:  
{’NDCG@1‘: 0.48077, ’NDCG@3‘: 0.41462, ’NDCG@5‘: 0.45521, ’NDCG@10‘: 0.50715} {’Recall@1‘: 0.19024, ’Recall@3‘: 0.37089, ’Recall@5‘: 0.48541, ’Recall@10‘: 0.61034}  

### subtask B Performance comparison of generation strategies

| Method                                 | RL_agg Score |
|----------------------------------------|--------------|
| Baseline                               | 0.361        |
| Qwen2.5 7B + LoRA                      | 0.423        |
| Qwen 2.5 7B + LoRA + Chat mode         | 0.425        |
| Qwen 3 14B                             | 0.451        |
| Qwen3 14B + LoRA                       | **0.526**    |
| Qwen3 14B + LoRA (Oversampling)        | 0.516        |
| Qwen3 14B + LoRA (Single Ref)          | 0.517        |
| Qwen3 14B + LoRA + Inference Prompt    | 0.518        |
| Qwen3 14B + LoRA + Classifier(LoRA)    | 0.383        |      
| Qwen3 14B + LoRA + Classifier(RoBERTa) | 0.493        |



### The evaluation of the Classifier by using RoBERTa
               precision    recall  f1-score   support

           0     0.8526    0.9779    0.9110       136
           1     0.0000    0.0000    0.0000        23
           2     0.8000    1.0000    0.8889         8
           3     0.0000    0.0000    0.0000         2

    accuracy                         0.8343       169
    macro avg     0.4131    0.4945    0.4500       169
    weighted avg     0.7240    0.8343    0.7752       169

    Confusion matrix (rows=true, cols=pred; order = ['ANSWERABLE', 'PARTIAL', 'UNANSWERABLE', 'CONVERSATIONAL'] ):
        [[133   3   0   0]
         [ 23   0   0   0]
         [  0   0   8   0]
         [  0   0   2   0]]

### Subtask C Performance comparison
| Method                                          | RL_agg Score |
|-------------------------------------------------|--------------|
| Retriever + Generator (Baseline)                | 0.384        |
| Retriever(top-10) + Reranker(top-5) + Generator | 0.421        |
| Retriever(top-50) + Reranker(top-5) + Generator | **0.435**    |

## Acknowledgements

This work was developed at the University of Tübingen as part of the SemEval Multi-Turn RAG shared task.