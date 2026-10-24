"""Value-of-Information (VoI) budget allocator package.

Adheres to MESSMATE_SPEC.md §6.3 and Master Blueprint §3.7.
"""

from app.services.allocator.engine import (
    V_FLOOR,
    CandidateLabel,
    GeneratedQuestion,
    ItemCandidate,
    compute_goodman_item_uncertainty,
    compute_goodman_variance,
    compute_label_mixture_moments,
    evaluate_budget_pressure_ask,
    get_budget_remaining,
    reserve_question_slot,
)

__all__ = [
    "V_FLOOR",
    "CandidateLabel",
    "GeneratedQuestion",
    "ItemCandidate",
    "evaluate_budget_pressure_ask",
    "reserve_question_slot",
    "get_budget_remaining",
    "compute_goodman_variance",
    "compute_label_mixture_moments",
    "compute_goodman_item_uncertainty",
]
