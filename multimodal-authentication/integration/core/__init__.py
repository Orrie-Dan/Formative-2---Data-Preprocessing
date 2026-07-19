"""Core shared building blocks.

``types``        : the plain data objects passed between components.
``interfaces``   : the ``RiskScorer`` contract (ML-model abstractions live in
                   :mod:`integration.abstractions`).
``registry``     : a factory that returns a mock or real implementation.
``model_loader`` : reusable loaders that locate/verify (and later deserialize)
                   the face / voice / recommendation model artifacts.
"""

from integration.core.interfaces import RiskScorer
from integration.core.model_loader import (
    FaceModelLoader,
    ModelLoader,
    PlaceholderModel,
    RecommendationModelLoader,
    VoiceModelLoader,
    build_face_loader,
    build_recommendation_loader,
    build_voice_loader,
    is_placeholder,
)
from integration.core.types import (
    AuthDecision,
    Decision,
    FaceResult,
    ProductRecommendation,
    RiskAssessment,
    VoiceResult,
)

__all__ = [
    "AuthDecision",
    "Decision",
    "FaceModelLoader",
    "FaceResult",
    "ModelLoader",
    "PlaceholderModel",
    "ProductRecommendation",
    "RecommendationModelLoader",
    "RiskAssessment",
    "RiskScorer",
    "VoiceModelLoader",
    "VoiceResult",
    "build_face_loader",
    "build_recommendation_loader",
    "build_voice_loader",
    "is_placeholder",
]
