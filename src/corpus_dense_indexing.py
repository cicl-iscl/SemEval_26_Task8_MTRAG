import ollama
import json
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import faiss

PROJECT_ROOT = Path("..").resolve()
DATASET = "clapnq"
CORPUS_PATH = PROJECT_ROOT / "dataset" / DATASET / "corpus.jsonl"

MODEL = "qwen3-embedding:4b"   # 你裝的 Qwen embedding 模型名稱（示例）
OUT_DIR = PROJECT_ROOT / "indexes" / f"{DATASET}-qwen-ollama-faiss"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Load corpus ---
docs = []
with open(CORPUS_PATH, "r") as f:
    for line in f:
        j = json.loads(line)
        docs.append(j["text"])

print("Total docs:", len(docs))

# --- Embeddings ---
all_embs = []
for text in tqdm(docs):
    emb = ollama.embeddings(model=MODEL, prompt=text)["embedding"]
    all_embs.append(emb)

embeddings = np.array(all_embs, dtype="float32")
faiss.normalize_L2(embeddings)
dim = embeddings.shape[1]
print("Embedding shape:", embeddings.shape)

# --- Build FAISS index ---
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

# save
faiss.write_index(index, str(OUT_DIR / "index.faiss"))
np.save(str(OUT_DIR / "emb.npy"), embeddings)
print("Saved to:", OUT_DIR)