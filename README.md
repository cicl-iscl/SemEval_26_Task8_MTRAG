# Multi-turn-RAG
09.11.2025 trying to upload files, BM25 retrieval for one benchmark  
14.11.2025 finish the baseline by using BM25  
19.11.2025 get the baseline of the dense indexing  
26.11.2025 first try for question rewrite, but the evaluation is lower than the baseline, will keep fixing some pronoun problem first before the top-k history finding   
To do: do the subquestion and question rewrite

Done:  
1. Upload the dataset into Github 
2. using git LFS to upload indexes.zip,  
please install git LFS before clone (basically the index for BM25 is fixed, so please run the code from BM25_all.py to get the index and please make sure you have indexes directory)
3. Using all the corpus to do the BM25 retrieval, also writing as method to read relative path, cmd code, evaluation etc. on BM25_all.py
4. Using bge to run dense indexing and retrieval baseline
5. write a evaluation method for dense retrieval in evaluation_dense_retrieval.py
6. queries_rewrite.ipynb is the first try for rewrite with top-k history, will revise to see if it can pass the baseline

temp:
bge做dense embedding太大，拿公司电脑跑一下（用colab大概跑一個多小時）看要不要真的跟老師申請bwHPC.  
如果想要跑 queries_rewrite.ipynb的話 要使用了ollama 並且下載了model在本地進行部署

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

