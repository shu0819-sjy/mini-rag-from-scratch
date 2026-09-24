# mini-rag-from-scratch 本机实测记录（2026-09-24）

环境：Windows 11 / Python 3.13 / CPU-only torch / sentence-transformers
模型：sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2（384 维）

## 运行结果

```
loaded 8 docs -> 24 chunks
recall@3 evaluation:
  [hit ] RAG 到底解决大模型的什么毛病？        -> ['limitations', 'what-is-rag']
  [hit ] 为什么切分文本的时候要有重叠？        -> ['chunking', 'similarity']
  [hit ] 这几句话描述的是它们的向量有什么特点？ -> ['embedding', 'similarity']
  [hit ] 余弦相似度的取值范围是多少？          -> ['recall-at-k', 'similarity']
  [hit ] 什么时候需要对候选结果做二次检索检查？ -> ['chunking', 'reranking', 'retrieval']
  [hit ] recall@k 是不是 k 越大越好？          -> ['recall-at-k', 'retrieval']
  [hit ] 什么时候需要在检索之后再加一层 rerank？ -> ['recall-at-k', 'reranking']
  [hit ] RAG 回答错误通常有哪些原因？          -> ['embedding', 'limitations']
recall@3 = 100% (8/8)
```

## 说明

- **recall@3 = 100%（8/8 全命中）**：语料与评测问题一一对应（自指式设计），这个数字证明的是管线正确性（chunking→embedding→检索→评测全链路跑通），不是泛化检索能力。
- 控制台输出乱码是 Windows 控制台 GBK 显示问题（`chcp 65001` 可解），不影响评测逻辑与结果文件。
- 首次运行会从 HuggingFace 下载模型（约 470MB）；直连不通时设 `HTTPS_PROXY` 或 `HF_ENDPOINT=https://hf-mirror.com`。
- 单查询模式：`py -3 main.py "你的问题"`；评测模式：`py -3 main.py --eval`。
