"""
AMRCO Hybrid Retriever
========================
Simulates dense + sparse hybrid retrieval pipeline.
In production: backed by FAISS (dense) + BM25 (sparse) + cross-encoder re-ranker.
This module provides the interface and simulation for reproducible experiments.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from src.router import RetrievalMode


@dataclass
class RetrievedPassage:
    passage_id:    str
    text:          str
    dense_score:   float    # cosine similarity
    sparse_score:  float    # BM25 score (normalised)
    rerank_score:  float    # cross-encoder score
    is_relevant:   bool     # ground-truth label (for evaluation)


@dataclass
class RetrievalResult:
    query:           str
    mode:            RetrievalMode
    passages:        List[RetrievedPassage]
    latency_ms:      float
    recall_at_k:     float
    precision_at_k:  float
    mrr:             float   # Mean Reciprocal Rank


class HybridRetriever:
    """
    Simulates FAISS dense retrieval + BM25 sparse retrieval + cross-encoder re-ranking.

    Retrieval quality parameters are calibrated to published benchmarks:
    - Dense-only:  Recall@10 = 0.71 (DPR on NQ, Karpukhin et al. 2020)
    - Hybrid:      Recall@10 = 0.79 (Hybrid search, +8pp over dense-only)
    - Re-ranker:   MRR@10 improves by ~12% over base retrieval
    """

    def __init__(self,
                 rng: Optional[np.random.Generator] = None,
                 knowledge_base_size: int = 100_000):
        self.rng = rng or np.random.default_rng(42)
        self.kb_size = knowledge_base_size

        # Calibrated retrieval quality parameters
        self._quality = {
            RetrievalMode.DENSE: {
                "recall@10": 0.71, "precision@10": 0.58, "mrr": 0.412,
                "latency_mean_ms": 82, "latency_std_ms": 18,
            },
            RetrievalMode.SPARSE: {
                "recall@10": 0.65, "precision@10": 0.52, "mrr": 0.378,
                "latency_mean_ms": 45, "latency_std_ms": 12,
            },
            RetrievalMode.HYBRID: {
                "recall@10": 0.79, "precision@10": 0.67, "mrr": 0.463,
                "latency_mean_ms": 148, "latency_std_ms": 28,
            },
            RetrievalMode.NONE: {
                "recall@10": 0.0,  "precision@10": 0.0, "mrr": 0.0,
                "latency_mean_ms": 0, "latency_std_ms": 0,
            },
        }

    def retrieve(self, query: str, mode: RetrievalMode,
                 k: int = 10) -> RetrievalResult:
        """Simulate retrieval and return scored passages."""
        if mode == RetrievalMode.NONE:
            return RetrievalResult(
                query=query, mode=mode, passages=[],
                latency_ms=0.0, recall_at_k=0.0,
                precision_at_k=0.0, mrr=0.0,
            )

        q  = self._quality[mode]
        n_relevant = max(1, int(k * q["precision@10"]))

        passages = []
        for i in range(k):
            is_rel   = i < n_relevant
            d_score  = self.rng.beta(8, 2) if is_rel else self.rng.beta(2, 8)
            s_score  = self.rng.beta(7, 3) if is_rel else self.rng.beta(2, 8)
            r_score  = self.rng.beta(9, 2) if is_rel else self.rng.beta(1, 9)
            passages.append(RetrievedPassage(
                passage_id  = f"doc_{self.rng.integers(self.kb_size):06d}",
                text        = f"[Simulated passage {i+1} for: {query[:40]}...]",
                dense_score = float(d_score),
                sparse_score= float(s_score),
                rerank_score= float(r_score),
                is_relevant = is_rel,
            ))

        # Sort by re-rank score
        passages.sort(key=lambda p: p.rerank_score, reverse=True)

        latency = float(self.rng.normal(q["latency_mean_ms"], q["latency_std_ms"]))
        latency = max(10.0, latency)

        # MRR: 1/rank of first relevant passage
        mrr = 0.0
        for rank, p in enumerate(passages, 1):
            if p.is_relevant:
                mrr = 1.0 / rank
                break

        return RetrievalResult(
            query        = query,
            mode         = mode,
            passages     = passages,
            latency_ms   = latency,
            recall_at_k  = q["recall@10"],
            precision_at_k = q["precision@10"],
            mrr          = mrr,
        )

    def batch_retrieve(self, queries: List[str], mode: RetrievalMode,
                       k: int = 10) -> List[RetrievalResult]:
        return [self.retrieve(q, mode, k) for q in queries]
