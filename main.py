# -*- coding: utf-8 -*-
"""mini-rag-from-scratch：从零手搓一个最小 RAG 系统（纯 numpy 检索，零重依赖）。

RAG 全流程只有四步，对应本文件里的四个函数：
  1) chunking   —— 把长文档切成小块，块与块之间带重叠；
  2) embedding  —— 用 sentence-transformers 把每个块编码成一个语义向量；
  3) retrieval  —— 问题也编码成向量，numpy 余弦相似度取 top-k；
  4) evaluation —— recall@k 评测：正确文档有没有被捞进 top-k。

用法：
  python main.py "为什么切块要有重叠"   # 单条查询演示
  python main.py --eval                 # 跑内置 recall@k 评测
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# 内置示例语料：docs/ 下 8 篇讲 RAG 本身的中文小文档
DOCS_DIR = Path(__file__).resolve().parent / "docs"
# 多语言小编码模型：中英文均可，输出 384 维；首次运行会自动下载（约 0.5GB）
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def load_docs() -> list[tuple[str, str]]:
    """读入语料：返回 [(文档名, 正文), ...]，按文件名排序保证结果可复现。"""
    return [(p.stem, p.read_text(encoding="utf-8").strip())
            for p in sorted(DOCS_DIR.glob("*.md"))]


def chunk(text: str, size: int = 120, overlap: int = 30) -> list[str]:
    """【第 1 步 chunking】固定窗口滑切：每块 size 字，相邻块重叠 overlap 字。

    为什么要切：编码模型输入长度有限（约 256~512 token），且长文本压成一个
    向量会"稀释"语义；切成小块才能做到"一个块只讲一件事"。
    为什么要重叠：答案句子可能正好骑在两块边界上，重叠能保住边界处的信息。
    """
    step = size - overlap                                # 窗口每次前进的步长
    return [text[i:i + size] for i in range(0, len(text), step)
            if text[i:i + size].strip()]                 # 顺手丢掉纯空白块


def embed(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """【第 2 步 embedding】把一批文本编码成归一化向量矩阵，形状 (n, 384)。

    直觉：语义相近的文本 → 向量在空间里也相近。"怎么退货"和"退款流程是什么"
    几乎没有共同关键词，向量却很接近——这正是语义检索强于关键词检索的原因。
    normalize_embeddings=True 把向量归一化成单位长度，之后「点积 = 余弦相似度」。
    """
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)


def retrieve(query: str, vecs: np.ndarray, model: SentenceTransformer,
             top_k: int = 3) -> list[tuple[int, float]]:
    """【第 3 步 retrieval】余弦相似度检索：纯 numpy，一次矩阵乘法算完所有得分。

    余弦相似度只看两个向量的夹角、不看长度，取值 [-1, 1]，越接近 1 越同义。
    向量已归一化，所以 vecs @ q 直接就是每个块与问题的余弦相似度；
    再用 argsort 从大到小取前 top_k 个下标，返回 (块下标, 相似度得分)。
    """
    q = embed([query], model)[0]                         # 问题也是文本，走同一编码器
    scores = vecs @ q                                    # (n,384)·(384,) → n 个相似度
    order = np.argsort(-scores)[:top_k]                  # 得分降序取前 k 个下标
    return [(int(i), float(scores[i])) for i in order]


# 【第 4 步 evaluation】最小评测集：(问题, 标准答案所在文档名)，共 8 条。
EVAL_SET = [
    ("RAG 相比直接问大模型有什么好处？", "what-is-rag"),
    ("为什么切分文本块时要有重叠？", "chunking"),
    ("两句话语义相近，它们的向量有什么特点？", "embedding"),
    ("余弦相似度的取值范围是多少？", "similarity"),
    ("检索时如何从所有块里挑出最相关的几块？", "retrieval"),
    ("recall@k 的 k 是不是越大越好？", "recall-at-k"),
    ("什么时候需要在检索之后加重排 rerank？", "reranking"),
    ("RAG 回答出错通常有哪些原因？", "limitations"),
]


def evaluate(vecs: np.ndarray, srcs: list[str],
             model: SentenceTransformer, k: int = 3) -> float:
    """recall@k：每个问题检索 top-k 块，看其来源文档是否包含金标准文档。

    命中记 1 分；recall@k = 命中问题数 / 总问题数。
    k 越大越容易命中，所以汇报指标时必须带上 k（例如 recall@3）。
    """
    hits = 0
    for query, gold in EVAL_SET:
        got = {srcs[i] for i, _ in retrieve(query, vecs, model, top_k=k)}
        ok = gold in got
        hits += ok
        mark = "[hit ]" if ok else "[miss]"
        print(f"  {mark} {query} -> {sorted(got)}")
    return hits / len(EVAL_SET)


def main() -> None:
    ap = argparse.ArgumentParser(description="mini RAG from scratch")
    ap.add_argument("query", nargs="?", help="演示查询；留空或加 --eval 则跑评测")
    ap.add_argument("--eval", action="store_true", help="跑内置 recall@k 评测")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    docs = load_docs()
    # 先切块并记住每块来自哪个文档（评测时要用"来源"对答案）
    chunks = [(name, c) for name, text in docs for c in chunk(text)]
    srcs = [name for name, _ in chunks]
    model = SentenceTransformer(MODEL_NAME)              # 首次运行会自动下载模型
    print(f"loaded {len(docs)} docs -> {len(chunks)} chunks")
    vecs = embed([c for _, c in chunks], model)

    if args.eval or not args.query:                      # 评测模式
        print(f"recall@{args.top_k} evaluation:")
        score = evaluate(vecs, srcs, model, k=args.top_k)
        print(f"recall@{args.top_k} = {score:.0%} "
              f"({int(score * len(EVAL_SET))}/{len(EVAL_SET)})")
    else:                                                # 检索演示模式
        for i, s in retrieve(args.query, vecs, model, top_k=args.top_k):
            print(f"[{s:.3f}] ({srcs[i]}) {chunks[i][1]}")


if __name__ == "__main__":
    main()
