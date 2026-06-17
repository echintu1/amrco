"""
AMRCO Cost Model
=================
Implements the mathematical cost model from Section III-C of the paper.

C_total = Σ_{q in W} [ Cost_inference(R(q), q)
                      + Cost_retrieval(r*(q), q)
                      + Cost_validation(theta*(q), q) ]

Cost_inference(t, q) = p_in(t) * i(q) + p_out(t) * o(q)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Sequence
from src.router import MODEL_SPECS, ModelTier, RetrievalMode, RoutingDecision

# ── Retrieval cost constants ──────────────────────────────────────────────────
# USD per 1,000 vector DB queries (approximate Azure AI Search / Pinecone pricing)
RETRIEVAL_COST_PER_1K = {
    RetrievalMode.NONE:   0.000,
    RetrievalMode.DENSE:  0.003,
    RetrievalMode.SPARSE: 0.001,
    RetrievalMode.HYBRID: 0.004,
}

# Validation cost: human-in-loop review at $0.05/query flagged for review
# Automated fact-check API: $0.002/query
VALIDATION_COST_AUTO   = 0.002
VALIDATION_COST_HUMAN  = 0.050
HUMAN_REVIEW_THRESHOLD = 0.80   # queries below this confidence get human review


@dataclass
class CostBreakdown:
    inference_usd:   float
    retrieval_usd:   float
    validation_usd:  float
    total_usd:       float
    model_key:       str
    tier:            str
    retrieval_mode:  str


def query_cost(decision: RoutingDecision,
               input_tokens: int = 512,
               output_tokens: int = 256) -> CostBreakdown:
    """Compute full per-query cost breakdown."""
    spec = MODEL_SPECS[decision.model_key]

    # Inference cost
    inf_cost = ((input_tokens  * spec.price_in_per_1m +
                 output_tokens * spec.price_out_per_1m) / 1_000_000)

    # Retrieval cost
    ret_cost = RETRIEVAL_COST_PER_1K[decision.retrieval_mode] / 1000.0

    # Validation cost
    if decision.confidence_threshold > HUMAN_REVIEW_THRESHOLD:
        # high-confidence threshold means we review fewer queries
        review_prob = 0.05
    else:
        review_prob = 0.15
    val_cost = VALIDATION_COST_AUTO + review_prob * VALIDATION_COST_HUMAN

    total = inf_cost + ret_cost + val_cost

    return CostBreakdown(
        inference_usd  = inf_cost,
        retrieval_usd  = ret_cost,
        validation_usd = val_cost,
        total_usd      = total,
        model_key      = decision.model_key,
        tier           = decision.tier.value,
        retrieval_mode = decision.retrieval_mode.value,
    )


def workload_cost_summary(decisions: Sequence[RoutingDecision],
                           input_tokens: int = 512,
                           output_tokens: int = 256) -> pd.DataFrame:
    """
    Compute cost summary for a workload of routing decisions.
    Returns per-tier breakdown and total.
    """
    rows = []
    for d in decisions:
        cb = query_cost(d, input_tokens, output_tokens)
        rows.append({
            "model":          cb.model_key,
            "tier":           cb.tier,
            "retrieval_mode": cb.retrieval_mode,
            "inference_usd":  cb.inference_usd,
            "retrieval_usd":  cb.retrieval_usd,
            "validation_usd": cb.validation_usd,
            "total_usd":      cb.total_usd,
        })
    df = pd.DataFrame(rows)

    summary = df.groupby("tier").agg(
        query_count    = ("total_usd", "count"),
        total_cost_usd = ("total_usd", "sum"),
        avg_cost_usd   = ("total_usd", "mean"),
        inference_pct  = ("inference_usd", lambda x: x.sum() / df["total_usd"].sum() * 100),
    ).reset_index()

    return summary


def savings_vs_t3_only(decisions: Sequence[RoutingDecision],
                        t3_model: str = "gpt-4o",
                        input_tokens: int = 512,
                        output_tokens: int = 256) -> dict:
    """
    Compute cost savings of AMRCO routing vs. all-T3 baseline.

    Savings = 1 - [ Σ Cost_inference(R(q), q) / Σ Cost_inference(T3, q) ]
    """
    spec_t3 = MODEL_SPECS[t3_model]
    t3_cost_per_query = ((input_tokens  * spec_t3.price_in_per_1m +
                          output_tokens * spec_t3.price_out_per_1m) / 1_000_000)

    amrco_total = sum(query_cost(d, input_tokens, output_tokens).inference_usd
                      for d in decisions)
    t3_total    = t3_cost_per_query * len(decisions)

    return {
        "n_queries":         len(decisions),
        "amrco_total_usd":   amrco_total,
        "t3_baseline_usd":   t3_total,
        "savings_usd":       t3_total - amrco_total,
        "savings_pct":       (1 - amrco_total / t3_total) * 100,
        "avg_amrco_per_query": amrco_total / len(decisions),
        "avg_t3_per_query":    t3_cost_per_query,
    }


def cost_per_million_tokens(model_key: str,
                             input_ratio: float = 0.75) -> float:
    """
    Weighted average cost per 1M tokens at given input:output ratio.
    Default 3:1 input:output = input_ratio=0.75.
    """
    spec = MODEL_SPECS[model_key]
    return (input_ratio * spec.price_in_per_1m +
            (1 - input_ratio) * spec.price_out_per_1m)
