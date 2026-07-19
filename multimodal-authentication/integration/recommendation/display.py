"""Recommendation display component (presentation only).

This module is responsible for *showing* a recommendation, and nothing else. It
deliberately contains **no prediction logic**: it never calls a model, never
touches the recommender, and never decides what to offer. It only takes an
already-computed result and renders it.

Separation of concerns
-----------------------
* :class:`RecommendationDisplay` — an immutable, framework-free *view-model*
  holding exactly the four things the UI shows:
  Recommended Product, Confidence Score, Authentication Status, Timestamp.
* ``render_text`` / ``render_streamlit`` — pure *renderers* that take a
  view-model and draw it (to a string, or to a Streamlit page). Swapping the
  renderer never touches the data, and building the data never imports a UI.

Mock now, real later
--------------------
``RecommendationDisplay.mock()`` returns placeholder values so the component can
be dropped into the UI today. When Member 3's model is wired up, the *same*
component renders real output via :meth:`RecommendationDisplay.from_prediction`,
which accepts a :class:`~integration.core.types.ProductRecommendation` (the real
model's output) plus the authentication outcome. No renderer changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from integration.core.types import ProductRecommendation

logger = logging.getLogger(__name__)

_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# Placeholder values used by :meth:`RecommendationDisplay.mock` until a real
# model is injected. Kept here so the "fake" data lives in exactly one place.
_MOCK_PRODUCT = "Premium Rewards Credit Card"
_MOCK_CONFIDENCE = 0.87


class AuthStatus(str, Enum):
    """Authentication outcome, as shown to the user."""

    AUTHENTICATED = "Authenticated"
    DENIED = "Access Denied"
    PENDING = "Pending"

    @classmethod
    def from_authenticated(cls, authenticated: bool) -> "AuthStatus":
        """Map a boolean auth result to a display status."""
        return cls.AUTHENTICATED if authenticated else cls.DENIED


@dataclass(frozen=True)
class RecommendationDisplay:
    """Immutable view-model: the exact data the display renders.

    Construct it with :meth:`mock` today, or :meth:`from_prediction` once a real
    :class:`ProductRecommendation` is available — the renderers do not care which.
    """

    recommended_product: str
    confidence_score: float  # in [0, 1]
    authentication_status: str
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.confidence_score) <= 1.0:
            raise ValueError(
                f"confidence_score must be in [0, 1], got {self.confidence_score!r}"
            )

    # --- derived, display-friendly views -------------------------------- #

    @property
    def confidence_percent(self) -> str:
        """Confidence as a rounded percentage string, e.g. ``"87%"``."""
        return f"{self.confidence_score * 100:.0f}%"

    @property
    def formatted_timestamp(self) -> str:
        """Timestamp formatted for display (``YYYY-MM-DD HH:MM:SS``)."""
        return self.timestamp.strftime(_TIMESTAMP_FORMAT)

    @property
    def is_authenticated(self) -> bool:
        return self.authentication_status == AuthStatus.AUTHENTICATED.value

    # --- constructors (the injection seam) ------------------------------ #

    @classmethod
    def mock(
        cls,
        *,
        authenticated: bool = True,
        timestamp: datetime | None = None,
    ) -> "RecommendationDisplay":
        """Build a display filled with placeholder values (no model involved)."""
        return cls(
            recommended_product=_MOCK_PRODUCT,
            confidence_score=_MOCK_CONFIDENCE,
            authentication_status=AuthStatus.from_authenticated(authenticated).value,
            timestamp=timestamp or datetime.now(),
        )

    @classmethod
    def from_prediction(
        cls,
        recommendation: ProductRecommendation,
        *,
        authenticated: bool,
        timestamp: datetime | None = None,
    ) -> "RecommendationDisplay":
        """Build a display from a real model output.

        This is the seam that lets real prediction results be *injected* later:
        pass the recommender's :class:`ProductRecommendation` and the auth
        outcome; the component renders it exactly like the mock.
        """
        return cls(
            recommended_product=recommendation.category,
            confidence_score=recommendation.confidence,
            authentication_status=AuthStatus.from_authenticated(authenticated).value,
            timestamp=timestamp or datetime.now(),
        )


# --------------------------------------------------------------------------- #
# Renderers — kept separate from the data so either can change independently.
# --------------------------------------------------------------------------- #
def render_text(display: RecommendationDisplay, *, min_width: int = 44) -> str:
    """Render the display as a plain-text block (for the CLI, logs, or tests).

    The box auto-sizes to its content so long values are never truncated.
    """
    title = "PRODUCT RECOMMENDATION"
    fields = (
        ("Recommended Product", display.recommended_product),
        ("Confidence Score", display.confidence_percent),
        ("Authentication Status", display.authentication_status),
        ("Timestamp", display.formatted_timestamp),
    )
    contents = [f" {label}: {value}" for label, value in fields]

    inner = max(min_width - 2, len(title) + 2, *(len(c) + 1 for c in contents))
    rule = "+" + "-" * inner + "+"

    lines = [rule, "|" + title.center(inner) + "|", rule]
    lines += ["|" + content.ljust(inner) + "|" for content in contents]
    lines.append(rule)
    return "\n".join(lines)


def render_streamlit(display: RecommendationDisplay, st_module: Any | None = None) -> None:
    """Render the display in a Streamlit app using metric cards.

    ``st_module`` is injectable so this can be unit-tested with a fake Streamlit
    (and so the module never imports streamlit at import time).
    """
    st = st_module if st_module is not None else _import_streamlit()

    st.subheader("Product Recommendation")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Recommended Product", display.recommended_product)
    col2.metric("Confidence Score", display.confidence_percent)
    col3.metric("Authentication Status", display.authentication_status)
    col4.metric("Timestamp", display.formatted_timestamp)


def _import_streamlit() -> Any:
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - only when streamlit absent
        raise RuntimeError(
            "streamlit is required for render_streamlit; install it or pass st_module."
        ) from exc
    return st


def _demo() -> None:
    """Print the text renderer with mock data (``python -m ...display``)."""
    print(render_text(RecommendationDisplay.mock()))


__all__ = [
    "AuthStatus",
    "RecommendationDisplay",
    "render_streamlit",
    "render_text",
]


if __name__ == "__main__":
    _demo()
