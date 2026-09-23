# mini-rag-from-scratch

A minimal **Retrieval-Augmented Generation (RAG)** system built from scratch in a single ~130-line `main.py` — **no FAISS, no vector database, no paid API**. Just `sentence-transformers` for embeddings and pure `numpy` for cosine-similarity retrieval.

It is a teaching repository: every step — **chunking / embedding / retrieval / recall@k evaluation** — is explained step by step in the Chinese code comments, and the built-in corpus is 8 short Chinese documents about RAG itself. The code answers questions about its own pipeline.

## Pipeline

```
docs/*.md ──chunk()──▶ chunks ──embed()──▶ chunk vectors (24 × 384)
                                               │
user query ──embed()──▶ query vector ──numpy cosine──┴──▶ top-k chunks ──▶ recall@k eval
```

| Step | Function | What it does |
|---|---|---|
| 1. Chunking | `chunk()` | sliding window of 120 chars with 30-char overlap, so answers that straddle a boundary survive |
| 2. Embedding | `embed()` | encodes each chunk into a normalized 384-dim vector (unit length ⇒ dot product = cosine similarity) |
| 3. Retrieval | `retrieve()` | encodes the query, computes `vecs @ q` in one matrix multiplication, argsort, take top-k |
| 4. Evaluation | `evaluate()` | recall@k: is the gold document inside the top-k retrieved chunks? |

## Quick start

```bash
pip install -r requirements.txt
python main.py "为什么切块要有重叠"   # single-query retrieval demo
python main.py --eval                # run the built-in recall@k evaluation
```

> The first run downloads the multilingual encoder `paraphrase-multilingual-MiniLM-L12-v2` (~0.5 GB) and caches it locally; afterwards everything works offline. No API keys needed.

## Expected output

Evaluation mode (`python main.py --eval`, output abbreviated, exact scores may vary slightly with model version):

```
loaded 8 docs -> 24 chunks
recall@3 evaluation:
  [hit ] RAG 相比直接问大模型有什么好处？ -> ['chunking', 'embedding', 'what-is-rag']
  [hit ] 为什么切分文本块时要有重叠？ -> ['chunking', 'embedding', 'what-is-rag']
  [hit ] 两句话语义相近，它们的向量有什么特点？ -> ['embedding', 'retrieval', 'similarity']
  ...
  [hit ] RAG 回答出错通常有哪些原因？ -> ['limitations', 'recall-at-k', 'reranking']
recall@3 = 100% (8/8)
```

Single-query mode (`python main.py "余弦相似度怎么算"`):

```
loaded 8 docs -> 24 chunks
[0.812] (similarity) 余弦相似度衡量两个向量方向的接近程度：夹角越小越相似，取值范围是 [-1, 1]...
[0.704] (retrieval) 检索就是"拿问题向量在知识库的向量里找最近的几个"...
[0.687] (embedding) embedding 是把一段文本映射成一个固定长度的向量...
```

## How the recall@k eval works

Each of the 8 test questions is annotated with the document that contains its answer (the gold label). The system retrieves the top-3 chunks for every question; if any retrieved chunk comes from the gold document, the question counts as a hit.

```
recall@k = (# questions whose gold doc appears in top-k) / (# questions)
```

k trades recall for precision: a larger k is easier to hit, which is why the metric is always reported *with* its k. Try `python main.py --eval --top-k 1` to see the harder setting.

## Project layout

```
mini-rag-from-scratch/
├── main.py           # the whole system: chunk → embed → retrieve → evaluate
├── docs/             # built-in toy corpus: 8 Chinese mini-docs about RAG itself
│   ├── what-is-rag.md
│   ├── chunking.md
│   ├── embedding.md
│   ├── similarity.md
│   ├── retrieval.md
│   ├── recall-at-k.md
│   ├── reranking.md
│   └── limitations.md
├── requirements.txt
└── README.md
```

## Extend it

- **Swap the encoder** — one line in `MODEL_NAME`; try a domain-specific model and watch recall@1 move.
- **Add a reranker** — retrieve top-50, rescore pairs with a cross-encoder, keep top-3.
- **Harder eval** — add queries whose gold answers span two documents, or measure recall@1.
- **Real corpora** — point `DOCS_DIR` at your own `.md` files; nothing else changes.

## License

MIT
