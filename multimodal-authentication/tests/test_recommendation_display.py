"""Tests for the recommendation display component.

Run from ``multimodal-authentication/``:

    python -m unittest discover -s tests -v
"""

import unittest
from datetime import datetime

from integration.core.types import ProductRecommendation
from integration.recommendation.display import (
    AuthStatus,
    RecommendationDisplay,
    render_streamlit,
    render_text,
)

FIXED_TS = datetime(2026, 7, 17, 12, 10, 5)


class MockData(unittest.TestCase):
    def test_mock_has_all_four_fields(self):
        display = RecommendationDisplay.mock(timestamp=FIXED_TS)
        self.assertTrue(display.recommended_product)
        self.assertGreaterEqual(display.confidence_score, 0.0)
        self.assertLessEqual(display.confidence_score, 1.0)
        self.assertEqual(display.authentication_status, AuthStatus.AUTHENTICATED.value)
        self.assertEqual(display.timestamp, FIXED_TS)

    def test_mock_denied_status(self):
        display = RecommendationDisplay.mock(authenticated=False)
        self.assertEqual(display.authentication_status, AuthStatus.DENIED.value)
        self.assertFalse(display.is_authenticated)


class Injection(unittest.TestCase):
    """Real prediction outputs can be injected via from_prediction."""

    def test_from_prediction_maps_fields(self):
        rec = ProductRecommendation(
            category="travel", confidence=0.734, model_name="real-recommender"
        )
        display = RecommendationDisplay.from_prediction(
            rec, authenticated=True, timestamp=FIXED_TS
        )
        self.assertEqual(display.recommended_product, "travel")
        self.assertAlmostEqual(display.confidence_score, 0.734)
        self.assertEqual(display.authentication_status, AuthStatus.AUTHENTICATED.value)
        self.assertTrue(display.is_authenticated)

    def test_from_prediction_denied(self):
        rec = ProductRecommendation(category="finance", confidence=0.9)
        display = RecommendationDisplay.from_prediction(rec, authenticated=False)
        self.assertEqual(display.authentication_status, AuthStatus.DENIED.value)


class DerivedViews(unittest.TestCase):
    def test_confidence_percent_formatting(self):
        display = RecommendationDisplay(
            recommended_product="x", confidence_score=0.874, authentication_status="Authenticated"
        )
        self.assertEqual(display.confidence_percent, "87%")

    def test_formatted_timestamp(self):
        display = RecommendationDisplay.mock(timestamp=FIXED_TS)
        self.assertEqual(display.formatted_timestamp, "2026-07-17 12:10:05")

    def test_invalid_confidence_raises(self):
        with self.assertRaises(ValueError):
            RecommendationDisplay(
                recommended_product="x",
                confidence_score=1.5,
                authentication_status="Authenticated",
            )


class TextRenderer(unittest.TestCase):
    def test_text_contains_all_four_fields(self):
        display = RecommendationDisplay.mock(timestamp=FIXED_TS)
        text = render_text(display)
        self.assertIn("Recommended Product", text)
        self.assertIn(display.recommended_product, text)
        self.assertIn("Confidence Score", text)
        self.assertIn(display.confidence_percent, text)
        self.assertIn("Authentication Status", text)
        self.assertIn(display.authentication_status, text)
        self.assertIn("Timestamp", text)
        self.assertIn(display.formatted_timestamp, text)


class _FakeColumn:
    def __init__(self, sink):
        self._sink = sink

    def metric(self, label, value):
        self._sink.append((label, value))


class _FakeStreamlit:
    def __init__(self):
        self.metrics: list[tuple[str, str]] = []
        self.subheaders: list[str] = []

    def subheader(self, text):
        self.subheaders.append(text)

    def columns(self, n):
        return [_FakeColumn(self.metrics) for _ in range(n)]


class StreamlitRenderer(unittest.TestCase):
    def test_render_streamlit_emits_four_metrics(self):
        fake = _FakeStreamlit()
        display = RecommendationDisplay.mock(timestamp=FIXED_TS)
        render_streamlit(display, st_module=fake)

        self.assertIn("Product Recommendation", fake.subheaders)
        labels = [label for label, _ in fake.metrics]
        self.assertEqual(
            labels,
            [
                "Recommended Product",
                "Confidence Score",
                "Authentication Status",
                "Timestamp",
            ],
        )
        values = dict(fake.metrics)
        self.assertEqual(values["Recommended Product"], display.recommended_product)
        self.assertEqual(values["Confidence Score"], display.confidence_percent)
        self.assertEqual(values["Timestamp"], display.formatted_timestamp)


if __name__ == "__main__":
    unittest.main()
