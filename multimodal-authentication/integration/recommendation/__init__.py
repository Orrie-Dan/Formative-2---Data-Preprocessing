"""Product-recommendation domain (Member 3).

Kept separate from ``integration.authentication`` on purpose: authentication
decides *who* the customer is, recommendation decides *what to offer* them.

    base.py    : re-exports the ProductRecommender contract
    mock.py    : deterministic stand-in used until the real model lands
    real.py    : where Member 3 loads models/recommendation/*.pkl
    service.py : thin wrapper the pipeline / app call
    display.py : presentation-only component that renders a recommendation
"""

from integration.recommendation.display import (
    AuthStatus,
    RecommendationDisplay,
    render_streamlit,
    render_text,
)
from integration.recommendation.service import RecommendationService

__all__ = [
    "AuthStatus",
    "RecommendationDisplay",
    "RecommendationService",
    "render_streamlit",
    "render_text",
]
