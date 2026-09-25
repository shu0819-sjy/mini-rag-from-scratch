# -*- coding: utf-8 -*-
"""Teaching tests for the four-step mini-RAG pipeline.

No model download, no network: sentence-transformers is stubbed at import time,
and embedding/retrieval use synthetic unit vectors so cosine math stays visible.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

# CI light job installs numpy + pytest only; stub ST before importing main.
sys.modules.setdefault(
    "sentence_transformers",
    SimpleNamespace(SentenceTransformer=MagicMock(name="SentenceTransformer")),
)

import main  # noqa: E402  — after ST stub


# ---------- helpers: synthetic embeddings (teaching-visible) ----------


def _unit(vec: list[float]) -> np.ndarray:
    a = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(a))
    return a / n if n else a


class FakeModel:
    """Maps known strings to fixed vectors; unknown text → orthogonal noise."""

    def __init__(self, table: dict[str, np.ndarray]):
        self.table = table
        self.dim = next(iter(table.values())).shape[0]

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        out = []
        for i, t in enumerate(texts):
            if t in self.table:
                v = self.table[t]
            else:
                # Deterministic pseudo-orthogonal filler so tests stay repeatable.
                rng = np.random.default_rng(abs(hash(t)) % (2**32))
                v = rng.standard_normal(self.dim).astype(np.float32)
            if normalize_embeddings:
                v = _unit(v.tolist())
            out.append(v)
        return np.stack(out, axis=0)


# ---------- 1) chunking ----------


def test_chunk_basic_windows_and_overlap():
    text = "abcdefghijklmnopqrstuvwxyz"  # 26 chars
    parts = main.chunk(text, size=10, overlap=4)
    # step = 6 → starts at 0,6,12,18,24
    assert parts[0] == "abcdefghij"
    assert parts[1] == "ghijklmnop"
    assert parts[0][-4:] == parts[1][:4]  # overlap preserved
    assert all(parts)  # no empty chunks


def test_chunk_drops_blank_slices():
    assert main.chunk("   ", size=2, overlap=0) == []
    assert main.chunk("ab", size=10, overlap=0) == ["ab"]


def test_chunk_step_equals_size_when_no_overlap():
    parts = main.chunk("1234567890", size=4, overlap=0)
    assert parts == ["1234", "5678", "90"]


# ---------- 2) embedding (via fake model) ----------


def test_embed_returns_normalized_matrix():
    model = FakeModel({"hello": _unit([1.0, 0.0, 0.0]), "world": _unit([0.0, 1.0, 0.0])})
    vecs = main.embed(["hello", "world"], model)  # type: ignore[arg-type]
    assert vecs.shape == (2, 3)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)
    assert np.allclose(vecs[0], [1.0, 0.0, 0.0])


# ---------- 3) retrieval ----------


def test_retrieve_ranks_by_cosine_with_synthetic_vectors():
    # Corpus: three docs along axes; query near doc0.
    corpus = np.stack(
        [
            _unit([1.0, 0.1, 0.0]),
            _unit([0.0, 1.0, 0.0]),
            _unit([0.0, 0.0, 1.0]),
        ]
    )
    model = FakeModel({"q": _unit([1.0, 0.0, 0.0])})
    hits = main.retrieve("q", corpus, model, top_k=2)  # type: ignore[arg-type]
    assert [i for i, _ in hits] == [0, 1]
    assert hits[0][1] > hits[1][1]


def test_retrieve_top_k_bounds():
    corpus = np.eye(3, dtype=np.float32)
    model = FakeModel({"q": _unit([1.0, 0.0, 0.0])})
    hits = main.retrieve("q", corpus, model, top_k=10)  # type: ignore[arg-type]
    assert len(hits) == 3


# ---------- 4) evaluation (recall@k on fixed synthetic corpus) ----------


def test_evaluate_recall_at_k_perfect_and_miss():
    """Fixed mapping: each EVAL_SET gold doc owns one distinctive chunk vector."""
    # Build one chunk per gold doc name; query vector ≈ that doc's vector.
    gold_names = [g for _, g in main.EVAL_SET]
    dim = len(gold_names)
    srcs = list(gold_names)
    vecs = np.eye(dim, dtype=np.float32)

    table = {}
    for i, (query, gold) in enumerate(main.EVAL_SET):
        # Query points almost exactly at its gold document axis.
        table[query] = _unit(np.eye(dim, dtype=np.float32)[srcs.index(gold)].tolist())

    model = FakeModel(table)
    score = main.evaluate(vecs, srcs, model, k=1)  # type: ignore[arg-type]
    assert score == 1.0


def test_evaluate_recall_miss_when_k_too_small_and_wrong_neighbor():
    """If every query is forced to the wrong vector, recall@1 must be 0."""
    srcs = [g for _, g in main.EVAL_SET]
    dim = len(srcs)
    vecs = np.eye(dim, dtype=np.float32)
    # All queries map to the *first* axis regardless of gold.
    wrong = _unit(np.eye(dim, dtype=np.float32)[0].tolist())
    table = {query: wrong for query, _ in main.EVAL_SET}
    model = FakeModel(table)
    score = main.evaluate(vecs, srcs, model, k=1)  # type: ignore[arg-type]
    # Only the first gold name can hit; others miss.
    expected_hits = sum(1 for _, gold in main.EVAL_SET if gold == srcs[0])
    assert score == pytest.approx(expected_hits / len(main.EVAL_SET))
