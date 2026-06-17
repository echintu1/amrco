"""
AMRCO End-to-End Pipeline
==========================
Wires together Router -> Retriever -> Model Dispatch -> Validator.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from src.router import AMRCORouter, Query, RoutingDecision, RouterConfig, SLATier
from src.retriever import HybridRetriever, RetrievalResult
from src.validator import PostGenerationValidator, ValidationResult
from src.cost_model import query_cost, CostBreakdown


@dataclass
class PipelineResult:
    query:           Query
    routing:         RoutingDecision
    retrieval:       Optional[RetrievalResult]
    validation:      ValidationResult
    cost:            CostBreakdown
    final_tier:      str    # may differ from routing if escalated
    was_escalated:   bool
    was_cached:      bool


class AMRCOPipeline:
    """
    Full AMRCO pipeline: L1 -> L2 -> L3 -> L4 -> L5 -> L6 -> L7
    """

    def __init__(self,
                 router_config: Optional[RouterConfig] = None,
                 domain_factor: float = 1.0,
                 seed: int = 42):
        rng = np.random.default_rng(seed)
        self.router    = AMRCORouter(config=router_config)
        self.retriever = HybridRetriever(rng=rng)
        self.validator = PostGenerationValidator(rng=rng, domain_factor=domain_factor)
        self._cache_hits = 0
        self._total      = 0

    def process(self, query: Query) -> PipelineResult:
        self._total += 1

        # L2: Route
        decision = self.router.route(query)
        was_cached = "[CACHE HIT]" in decision.reasoning

        if was_cached:
            self._cache_hits += 1

        # L3: Retrieve
        retrieval = self.retriever.retrieve(
            query.text, decision.retrieval_mode
        )

        # L6: Validate
        validation = self.validator.validate(decision, retrieval)

        # Escalation handling
        was_escalated = False
        final_decision = decision
        if validation.escalation_required:
            was_escalated = True
            # Re-route at higher tier
            from src.router import ModelTier
            escalated_query = Query(
                text             = query.text,
                token_length     = query.token_length,
                entity_count     = query.entity_count,
                reasoning_depth  = min(query.reasoning_depth + 0.3, 1.0),
                domain_specificity = query.domain_specificity,
                requires_freshness = query.requires_freshness,
                sla_tier         = SLATier.CRITICAL,
            )
            final_decision = self.router.route(escalated_query)
            retrieval  = self.retriever.retrieve(query.text, final_decision.retrieval_mode)
            validation = self.validator.validate(final_decision, retrieval)

        cost = query_cost(final_decision, query.token_length)

        return PipelineResult(
            query        = query,
            routing      = final_decision,
            retrieval    = retrieval,
            validation   = validation,
            cost         = cost,
            final_tier   = final_decision.tier.value,
            was_escalated = was_escalated,
            was_cached   = was_cached,
        )

    def process_batch(self, queries: List[Query]) -> List[PipelineResult]:
        return [self.process(q) for q in queries]

    @property
    def cache_hit_rate(self) -> float:
        return self._cache_hits / max(self._total, 1)
