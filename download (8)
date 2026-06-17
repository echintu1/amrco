"""
AMRCO Post-Generation Validator
=================================
Confidence scoring and hallucination detection layer (L6).

In production: uses an ensemble of:
  1. Self-consistency sampling (3 independent generations, majority vote)
  2. NLI-based entailment check against retrieved passages
  3. Named entity cross-reference against knowledge base
  4. Calibrated confidence score from model log-probs

This module provides the simulation and statistical model for experiments.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional
from src.router import MODEL_SPECS, ModelTier, RoutingDecision
from src.retriever import RetrievalResult, RetrievalMode


@dataclass
class ValidationResult:
    confidence_score:    float    # 0..1
    hallucination_flag:  bool     # True if hallucination detected
    escalation_required: bool     # True if should escalate to higher tier
    human_review_flag:   bool     # True if needs human review
    nli_score:           float    # entailment score vs. retrieved context
    consistency_score:   float    # self-consistency across 3 samples


# Hallucination rate reduction from RAG (empirically calibrated)
RAG_HALLUCINATION_REDUCTION = {
    RetrievalMode.NONE:   0.00,
    RetrievalMode.DENSE:  0.48,   # 48% relative reduction
    RetrievalMode.SPARSE: 0.35,
    RetrievalMode.HYBRID: 0.58,   # best reduction
}


class PostGenerationValidator:
    """
    Simulates the post-generation validation pipeline.

    Hallucination model:
        P(hallucination | model, retrieval, domain) =
            base_rate(model) * (1 - RAG_reduction(retrieval)) * domain_factor

    domain_factor > 1 for high-specificity domains (finance, healthcare)
    """

    def __init__(self, rng: Optional[np.random.Generator] = None,
                 domain_factor: float = 1.0):
        self.rng = rng or np.random.default_rng(42)
        self.domain_factor = domain_factor   # 1.2 for financial domain

    def validate(self, decision: RoutingDecision,
                 retrieval: Optional[RetrievalResult] = None) -> ValidationResult:
        """
        Compute confidence score and hallucination flag for a generated response.
        """
        spec     = MODEL_SPECS[decision.model_key]
        base_hr  = spec.hallucination_rate

        # Apply RAG reduction
        rag_mode = retrieval.mode if retrieval else RetrievalMode.NONE
        rag_red  = RAG_HALLUCINATION_REDUCTION[rag_mode]
        adj_hr   = base_hr * (1 - rag_red) * self.domain_factor
        adj_hr   = float(np.clip(adj_hr, 0.01, 0.99))

        # Simulate hallucination detection
        is_hallucination = bool(self.rng.random() < adj_hr)

        # NLI score: higher when retrieval exists and passage is relevant
        if retrieval and retrieval.passages:
            nli_base = retrieval.precision_at_k
            nli_score = float(np.clip(
                self.rng.normal(nli_base, 0.08), 0.0, 1.0
            ))
        else:
            nli_score = float(np.clip(self.rng.normal(0.40, 0.12), 0.0, 1.0))

        # Self-consistency: T3 models more consistent
        tier_consistency = {
            ModelTier.T1: 0.70, ModelTier.T2: 0.80, ModelTier.T3: 0.90
        }[decision.tier]
        consistency_score = float(np.clip(
            self.rng.normal(tier_consistency, 0.06), 0.0, 1.0
        ))

        # Combined confidence score (weighted ensemble)
        confidence = (0.40 * nli_score +
                      0.35 * consistency_score +
                      0.25 * (1 - adj_hr))
        confidence = float(np.clip(
            self.rng.normal(confidence, 0.04), 0.0, 1.0
        ))

        escalation = (confidence < decision.confidence_threshold and
                      decision.tier != ModelTier.T3)
        human_review = confidence < 0.60 or is_hallucination

        return ValidationResult(
            confidence_score   = confidence,
            hallucination_flag = is_hallucination,
            escalation_required = escalation,
            human_review_flag  = human_review,
            nli_score          = nli_score,
            consistency_score  = consistency_score,
        )

    def batch_validate(self, decisions, retrievals=None):
        if retrievals is None:
            retrievals = [None] * len(decisions)
        return [self.validate(d, r) for d, r in zip(decisions, retrievals)]
