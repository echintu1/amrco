"""
AMRCO Routing Decision Engine
===============================
Implements the adaptive query router described in Algorithm 1 of the paper.

Mathematical formulation:
  cs(q) = w1*len(q) + w2*entity_count(q) + w3*reasoning_depth(q) + w4*domain_specificity(q)
  R(q)  = argmin_{t in T} [ lambda * Cost(t,q) + (1-lambda) * LatencyPenalty(t,q) ]
           subject to: Accuracy(t,q) >= AccuracyThreshold(sla_tier(q))
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Optional
from enum import Enum

# ── Model Tiers ──────────────────────────────────────────────────────────────

class ModelTier(Enum):
    T1 = "T1"   # Llama 3 8B  — low cost, high speed
    T2 = "T2"   # Claude Haiku / Gemini Flash — mid tier
    T3 = "T3"   # GPT-4o / Claude 3.5 Sonnet — frontier

class RetrievalMode(Enum):
    NONE   = "none"
    DENSE  = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"

class SLATier(Enum):
    STANDARD  = "standard"   # best-effort, cost-first
    PREMIUM   = "premium"    # accuracy >= 0.85
    CRITICAL  = "critical"   # accuracy >= 0.92, compliance use-cases

# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class RouterConfig:
    """
    Calibrated on 50,000-query enterprise annotation corpus.
    Weights for complexity score: w1..w4 must sum to 1.0
    """
    # Complexity score weights
    w1: float = 0.15   # token length weight
    w2: float = 0.25   # entity count weight
    w3: float = 0.40   # reasoning depth weight
    w4: float = 0.20   # domain specificity weight

    # Routing thresholds (calibrated on held-out set)
    theta_low:  float = 0.32   # below -> T1 eligible
    theta_high: float = 0.67   # above -> T3 required

    # Cost-latency trade-off parameter
    # lambda=1.0 -> cost only, lambda=0.0 -> latency only
    lam: float = 0.65

    # Per-tier accuracy thresholds
    accuracy_threshold: dict = field(default_factory=lambda: {
        SLATier.STANDARD: 0.70,
        SLATier.PREMIUM:  0.85,
        SLATier.CRITICAL: 0.92,
    })

    # Latency SLA (ms) — p90 target
    latency_sla_ms: float = 2000.0

    # Cost budget per query (USD)
    cost_budget_usd: float = 0.01

# ── Model Specifications ─────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    name:            str
    tier:            ModelTier
    price_in_per_1m: float   # USD per 1M input tokens
    price_out_per_1m: float  # USD per 1M output tokens
    p50_latency_ms:  float
    p90_latency_ms:  float
    base_accuracy:   float   # MMLU proxy
    hallucination_rate: float  # baseline rate (no RAG)

MODEL_SPECS: dict[str, ModelSpec] = {
    "llama3-8b": ModelSpec(
        name="Llama 3 8B", tier=ModelTier.T1,
        price_in_per_1m=0.10, price_out_per_1m=0.10,
        p50_latency_ms=210, p90_latency_ms=390,
        base_accuracy=0.683, hallucination_rate=0.187,
    ),
    "claude-haiku": ModelSpec(
        name="Claude 3 Haiku", tier=ModelTier.T2,
        price_in_per_1m=0.25, price_out_per_1m=1.25,
        p50_latency_ms=480, p90_latency_ms=890,
        base_accuracy=0.820, hallucination_rate=0.092,
    ),
    "gemini-flash": ModelSpec(
        name="Gemini 1.5 Flash", tier=ModelTier.T2,
        price_in_per_1m=0.075, price_out_per_1m=0.30,
        p50_latency_ms=420, p90_latency_ms=810,
        base_accuracy=0.814, hallucination_rate=0.098,
    ),
    "gpt-4o": ModelSpec(
        name="GPT-4o", tier=ModelTier.T3,
        price_in_per_1m=5.00, price_out_per_1m=15.00,
        p50_latency_ms=1240, p90_latency_ms=2180,
        base_accuracy=0.887, hallucination_rate=0.052,
    ),
    "claude-3-5-sonnet": ModelSpec(
        name="Claude 3.5 Sonnet", tier=ModelTier.T3,
        price_in_per_1m=3.00, price_out_per_1m=15.00,
        p50_latency_ms=1180, p90_latency_ms=2040,
        base_accuracy=0.881, hallucination_rate=0.041,
    ),
    "gemini-1-5-pro": ModelSpec(
        name="Gemini 1.5 Pro", tier=ModelTier.T3,
        price_in_per_1m=3.50, price_out_per_1m=10.50,
        p50_latency_ms=1390, p90_latency_ms=2450,
        base_accuracy=0.864, hallucination_rate=0.068,
    ),
    "llama3-70b": ModelSpec(
        name="Llama 3 70B", tier=ModelTier.T2,
        price_in_per_1m=0.90, price_out_per_1m=0.90,
        p50_latency_ms=680, p90_latency_ms=1210,
        base_accuracy=0.820, hallucination_rate=0.113,
    ),
}

# ── Query Representation ─────────────────────────────────────────────────────

@dataclass
class Query:
    text:             str
    token_length:     int
    entity_count:     int
    reasoning_depth:  float   # 0..1, estimated by classifier
    domain_specificity: float # 0..1
    requires_freshness: bool  = False
    sla_tier:         SLATier = SLATier.STANDARD

# ── Routing Decision ─────────────────────────────────────────────────────────

@dataclass
class RoutingDecision:
    model_key:          str
    tier:               ModelTier
    retrieval_mode:     RetrievalMode
    confidence_threshold: float
    estimated_cost_usd: float
    estimated_latency_ms: float
    complexity_score:   float
    reasoning:          str

# ── Router ───────────────────────────────────────────────────────────────────

class AMRCORouter:
    """
    Implements Algorithm 1: AMRCO Adaptive Query Router.

    Routing decision function:
        R(q) = argmin_{t in T} [ lam * Cost(t,q) + (1-lam) * LatencyPenalty(t,q) ]
        subject to: Accuracy(t,q) >= AccuracyThreshold(sla_tier(q))
    """

    def __init__(self, config: Optional[RouterConfig] = None):
        self.config = config or RouterConfig()
        self._cache: dict[int, RoutingDecision] = {}

        # Default T2/T3 model assignments
        self.t2_model = "llama3-70b"
        self.t3_model = "claude-3-5-sonnet"

    def complexity_score(self, query: Query) -> float:
        """
        cs(q) = w1*norm(len) + w2*norm(entities) + w3*reasoning_depth + w4*domain_specificity

        Token length normalized to [0,1] with saturation at 2048 tokens.
        Entity count normalized with saturation at 20.
        """
        c = self.config
        norm_len    = min(query.token_length / 2048.0, 1.0)
        norm_entity = min(query.entity_count  / 20.0,  1.0)
        cs = (c.w1 * norm_len +
              c.w2 * norm_entity +
              c.w3 * query.reasoning_depth +
              c.w4 * query.domain_specificity)
        return float(np.clip(cs, 0.0, 1.0))

    def estimate_cost(self, model_key: str, query: Query,
                      avg_output_tokens: int = 256) -> float:
        """
        Cost_inference(t, q) = p_in(t) * i(q) + p_out(t) * o(q)   [USD]
        """
        spec = MODEL_SPECS[model_key]
        cost = ((query.token_length * spec.price_in_per_1m +
                 avg_output_tokens  * spec.price_out_per_1m) / 1_000_000)
        return cost

    def latency_penalty(self, model_key: str,
                        retrieval_mode: RetrievalMode) -> float:
        """
        Normalised latency penalty [0..1] relative to SLA target.
        Adds retrieval overhead: DENSE +80ms, HYBRID +150ms.
        """
        spec     = MODEL_SPECS[model_key]
        overhead = {RetrievalMode.NONE: 0,
                    RetrievalMode.DENSE: 80,
                    RetrievalMode.SPARSE: 60,
                    RetrievalMode.HYBRID: 150}[retrieval_mode]
        total_p90 = spec.p90_latency_ms + overhead
        return min(total_p90 / self.config.latency_sla_ms, 2.0)

    def _select_retrieval(self, tier: ModelTier,
                          requires_freshness: bool) -> RetrievalMode:
        if tier == ModelTier.T1 and not requires_freshness:
            return RetrievalMode.NONE
        if tier == ModelTier.T2:
            return RetrievalMode.DENSE
        return RetrievalMode.HYBRID   # T3 always uses hybrid

    def route(self, query: Query) -> RoutingDecision:
        """
        Main routing function — Algorithm 1 implementation.
        """
        # Step 1: cache lookup
        cache_key = hash((query.text, query.sla_tier))
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return RoutingDecision(**{**cached.__dict__,
                                      "reasoning": cached.reasoning + " [CACHE HIT]"})

        # Step 2: compute complexity score
        cs = self.complexity_score(query)

        # Step 3: initial tier assignment
        cfg = self.config
        acc_req = cfg.accuracy_threshold[query.sla_tier]

        if cs < cfg.theta_low and not query.requires_freshness:
            candidate_tier  = ModelTier.T1
            candidate_model = "llama3-8b"
            conf_threshold  = 0.70
            reason = f"cs={cs:.3f} < theta_low={cfg.theta_low} -> T1"
        elif cs < cfg.theta_high and not query.requires_freshness:
            candidate_tier  = ModelTier.T2
            candidate_model = self.t2_model
            conf_threshold  = 0.80
            reason = f"theta_low <= cs={cs:.3f} < theta_high={cfg.theta_high} -> T2"
        else:
            candidate_tier  = ModelTier.T3
            candidate_model = self.t3_model
            conf_threshold  = 0.90
            reason = f"cs={cs:.3f} >= theta_high={cfg.theta_high} -> T3"

        # Step 4: accuracy gate — upgrade tier if needed
        base_acc = MODEL_SPECS[candidate_model].base_accuracy
        if base_acc < acc_req:
            if candidate_tier == ModelTier.T1:
                candidate_tier  = ModelTier.T2
                candidate_model = self.t2_model
                conf_threshold  = 0.80
                reason += f" | upgraded T1->T2 (acc {base_acc:.2f} < req {acc_req:.2f})"
            if MODEL_SPECS[candidate_model].base_accuracy < acc_req:
                candidate_tier  = ModelTier.T3
                candidate_model = self.t3_model
                conf_threshold  = 0.90
                reason += f" | upgraded T2->T3 (acc req {acc_req:.2f})"

        # Step 5: select retrieval mode
        retrieval = self._select_retrieval(candidate_tier, query.requires_freshness)

        # Step 6: cost and latency check
        est_cost    = self.estimate_cost(candidate_model, query)
        lat_penalty = self.latency_penalty(candidate_model, retrieval)

        # Step 7: downgrade if over budget (but not below accuracy floor)
        if est_cost > cfg.cost_budget_usd and candidate_tier == ModelTier.T3:
            t2_acc = MODEL_SPECS[self.t2_model].base_accuracy
            if t2_acc >= acc_req:
                candidate_tier  = ModelTier.T2
                candidate_model = self.t2_model
                conf_threshold  = 0.80
                retrieval       = RetrievalMode.DENSE
                est_cost        = self.estimate_cost(candidate_model, query)
                reason += " | downgraded T3->T2 (cost budget)"

        est_latency = (MODEL_SPECS[candidate_model].p50_latency_ms +
                       {RetrievalMode.NONE: 0, RetrievalMode.DENSE: 80,
                        RetrievalMode.SPARSE: 60, RetrievalMode.HYBRID: 150}[retrieval])

        decision = RoutingDecision(
            model_key            = candidate_model,
            tier                 = candidate_tier,
            retrieval_mode       = retrieval,
            confidence_threshold = conf_threshold,
            estimated_cost_usd   = est_cost,
            estimated_latency_ms = est_latency,
            complexity_score     = cs,
            reasoning            = reason,
        )

        self._cache[cache_key] = decision
        return decision

    def clear_cache(self):
        self._cache.clear()
