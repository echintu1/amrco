"""
Unit tests for AMRCO framework.
Run with: python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest

from src.router import (AMRCORouter, RouterConfig, Query, ModelTier,
                         RetrievalMode, SLATier, MODEL_SPECS)
from src.cost_model import query_cost, cost_per_million_tokens, savings_vs_t3_only
from src.retriever import HybridRetriever
from src.validator import PostGenerationValidator


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def router():
    return AMRCORouter()

@pytest.fixture
def simple_query():
    return Query(text="What is the capital of France?",
                 token_length=12, entity_count=2,
                 reasoning_depth=0.1, domain_specificity=0.1,
                 sla_tier=SLATier.STANDARD)

@pytest.fixture
def complex_query():
    return Query(text="Analyze the ISDA master agreement implications for SOFR transition risk",
                 token_length=256, entity_count=8,
                 reasoning_depth=0.92, domain_specificity=0.95,
                 requires_freshness=True, sla_tier=SLATier.CRITICAL)


# ── Router tests ──────────────────────────────────────────────────────────────

class TestRouter:
    def test_simple_query_routes_to_t1_or_t2(self, router, simple_query):
        """
        Simple query has low complexity score -> T1 candidate.
        Llama 3 8B base accuracy (0.683) < STANDARD threshold (0.70),
        so router correctly upgrades to T2. This is expected behavior.
        """
        decision = router.route(simple_query)
        assert decision.tier in (ModelTier.T1, ModelTier.T2), \
            f"Simple query should route to T1 or T2, got {decision.tier}"
        assert decision.complexity_score < router.config.theta_high

    def test_complex_query_routes_to_t3(self, router, complex_query):
        decision = router.route(complex_query)
        assert decision.tier == ModelTier.T3, \
            f"Complex query should route to T3, got {decision.tier}"

    def test_complexity_score_bounds(self, router, simple_query, complex_query):
        cs_simple  = router.complexity_score(simple_query)
        cs_complex = router.complexity_score(complex_query)
        assert 0.0 <= cs_simple  <= 1.0, "CS must be in [0,1]"
        assert 0.0 <= cs_complex <= 1.0, "CS must be in [0,1]"
        assert cs_simple < cs_complex,   "Complex query must have higher CS"

    def test_critical_sla_upgrades_tier(self, router):
        """CRITICAL SLA (acc>=0.92) should escalate to T3 — no T1/T2 model meets 0.92."""
        q = Query(text="Simple question", token_length=10, entity_count=1,
                  reasoning_depth=0.05, domain_specificity=0.05,
                  sla_tier=SLATier.CRITICAL)
        decision = router.route(q)
        # Both T1 (0.683) and T2 (0.820) are below 0.92 threshold -> must be T3
        assert decision.tier == ModelTier.T3, \
            f"CRITICAL SLA (0.92 acc) should force T3, got {decision.tier}"

    def test_freshness_requirement_disables_t1(self, router):
        q = Query(text="Current interest rates", token_length=20, entity_count=2,
                  reasoning_depth=0.2, domain_specificity=0.3,
                  requires_freshness=True, sla_tier=SLATier.STANDARD)
        decision = router.route(q)
        assert decision.tier != ModelTier.T1, \
            "Freshness requirement should disqualify T1"

    def test_cache_hit(self, router, simple_query):
        d1 = router.route(simple_query)
        d2 = router.route(simple_query)
        assert "[CACHE HIT]" in d2.reasoning

    def test_t3_uses_hybrid_retrieval(self, router, complex_query):
        d = router.route(complex_query)
        assert d.retrieval_mode == RetrievalMode.HYBRID

    def test_routing_decision_has_cost(self, router, simple_query):
        d = router.route(simple_query)
        assert d.estimated_cost_usd > 0

    def test_confidence_threshold_t3_highest(self, router):
        """T3 should always have highest confidence threshold."""
        q_t3 = Query(text="complex", token_length=500, entity_count=15,
                     reasoning_depth=0.95, domain_specificity=0.95,
                     sla_tier=SLATier.CRITICAL)
        d = router.route(q_t3)
        assert d.confidence_threshold >= 0.85


# ── Cost model tests ──────────────────────────────────────────────────────────

class TestCostModel:
    def test_cost_increases_with_tier(self):
        """T3 model cost should be substantially higher than T1."""
        c_t1 = cost_per_million_tokens("llama3-8b")
        c_t3 = cost_per_million_tokens("gpt-4o")
        assert c_t3 > c_t1 * 10, "T3 should cost at least 10x T1"

    def test_savings_positive(self):
        router = AMRCORouter()
        queries = [Query(text=f"q{i}", token_length=100+i*5, entity_count=2,
                        reasoning_depth=0.3, domain_specificity=0.3,
                        sla_tier=SLATier.STANDARD) for i in range(100)]
        decisions = [router.route(q) for q in queries]
        sv = savings_vs_t3_only(decisions, t3_model="gpt-4o")
        assert sv["savings_pct"] > 0, "AMRCO should always save vs all-T3"
        assert sv["savings_pct"] < 100, "Savings must be < 100%"

    def test_query_cost_breakdown_sums(self):
        from src.router import RoutingDecision, ModelTier, RetrievalMode
        d = RoutingDecision(
            model_key="llama3-8b", tier=ModelTier.T1,
            retrieval_mode=RetrievalMode.NONE,
            confidence_threshold=0.70, estimated_cost_usd=0.001,
            estimated_latency_ms=210, complexity_score=0.2, reasoning="test"
        )
        cb = query_cost(d, 256, 128)
        assert abs(cb.total_usd - (cb.inference_usd + cb.retrieval_usd + cb.validation_usd)) < 1e-9


# ── Retriever tests ───────────────────────────────────────────────────────────

class TestRetriever:
    def test_hybrid_better_recall_than_dense(self):
        r = HybridRetriever(rng=np.random.default_rng(42))
        res_dense  = r.retrieve("test", RetrievalMode.DENSE,  k=10)
        res_hybrid = r.retrieve("test", RetrievalMode.HYBRID, k=10)
        assert res_hybrid.recall_at_k >= res_dense.recall_at_k

    def test_none_mode_returns_empty(self):
        r = HybridRetriever()
        res = r.retrieve("test", RetrievalMode.NONE)
        assert len(res.passages) == 0
        assert res.latency_ms == 0.0

    def test_passages_sorted_by_rerank(self):
        r = HybridRetriever(rng=np.random.default_rng(42))
        res = r.retrieve("test", RetrievalMode.HYBRID, k=10)
        scores = [p.rerank_score for p in res.passages]
        assert scores == sorted(scores, reverse=True), "Passages must be sorted by rerank score"


# ── Validator tests ───────────────────────────────────────────────────────────

class TestValidator:
    def test_t3_lower_hallucination_than_t1(self):
        from src.router import RoutingDecision, ModelTier, RetrievalMode
        rng = np.random.default_rng(42)
        v   = PostGenerationValidator(rng=rng)
        n   = 500

        def make_d(mk):
            return RoutingDecision(mk, MODEL_SPECS[mk].tier,
                                   RetrievalMode.NONE, 0.8, 0.01, 500, 0.5, "t")

        t1_flags = [v.validate(make_d("llama3-8b")).hallucination_flag  for _ in range(n)]
        t3_flags = [v.validate(make_d("claude-3-5-sonnet")).hallucination_flag for _ in range(n)]
        assert np.mean(t1_flags) > np.mean(t3_flags), \
            "T1 should have higher hallucination rate than T3"

    def test_hybrid_rag_reduces_hallucination(self):
        from src.router import RoutingDecision, ModelTier, RetrievalMode
        retriever = HybridRetriever(rng=np.random.default_rng(42))
        rng = np.random.default_rng(42)
        v   = PostGenerationValidator(rng=rng)
        n   = 500

        d = RoutingDecision("gpt-4o", ModelTier.T3, RetrievalMode.HYBRID,
                            0.9, 0.01, 1200, 0.8, "t")
        no_rag_flags = [v.validate(RoutingDecision(
            "gpt-4o", ModelTier.T3, RetrievalMode.NONE, 0.9, 0.01, 1200, 0.8, "t"
        )).hallucination_flag for _ in range(n)]
        rag_flags = [v.validate(d, retriever.retrieve("q", RetrievalMode.HYBRID)
                                ).hallucination_flag for _ in range(n)]
        assert np.mean(no_rag_flags) > np.mean(rag_flags), \
            "Hybrid RAG should reduce hallucination vs no RAG"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
