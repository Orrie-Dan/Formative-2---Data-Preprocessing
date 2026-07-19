"""Shared pytest fixtures and test doubles for the authentication workflow.

This is the reusable core of the workflow testing framework. It provides:

* **Test doubles** (no ML): authenticators/recommenders that pass, fail,
  validate their input file, raise a chosen error, or record their calls.
* **Fixtures**: a ``build_workflow`` factory plus a couple of ready-made
  workflows, and a temp-file helper for the missing/valid-file scenarios.

Every double honours the same abstraction the real models do
(:class:`FaceAuthenticator` / :class:`VoiceAuthenticator` /
:class:`ProductRecommender`), so tests exercise the *workflow*, never a model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from integration.abstractions import (
    FaceAuthenticator,
    ProductRecommender,
    VoiceAuthenticator,
)
from integration.authentication.face.mock import MockFaceAuthenticator
from integration.authentication.voice.mock import MockVoiceAuthenticator
from integration.authentication.workflow import AuthenticationWorkflow
from integration.core.types import FaceResult, ProductRecommendation, VoiceResult
from integration.validation import validate_audio_path, validate_image_path

CUSTOMER_ID = "a001"


# --------------------------------------------------------------------------- #
# Test doubles (framework building blocks)
# --------------------------------------------------------------------------- #
class ValidatingFaceAuthenticator(FaceAuthenticator):
    """Validates the image path (missing/format), then passes. No ML."""

    name = "validating-face"

    def verify(self, image_path, claimed_id) -> FaceResult:
        validate_image_path(image_path)  # raises SampleError on missing/bad file
        return FaceResult(
            matched=True, confidence=1.0, person_id=str(claimed_id), model_name=self.name
        )


class ValidatingVoiceAuthenticator(VoiceAuthenticator):
    """Validates the audio path (missing/format), then passes. No ML."""

    name = "validating-voice"

    def verify(self, audio_path, claimed_id) -> VoiceResult:
        validate_audio_path(audio_path)  # raises SampleError on missing/bad file
        return VoiceResult(
            matched=True, confidence=1.0, person_id=str(claimed_id), model_name=self.name
        )


class RaisingFaceAuthenticator(FaceAuthenticator):
    """Raises a chosen exception on verify (for missing-model / unexpected)."""

    name = "raising-face"

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def verify(self, image_path, claimed_id) -> FaceResult:
        raise self._exc


class RaisingVoiceAuthenticator(VoiceAuthenticator):
    """Raises a chosen exception on verify (for missing-model / unexpected)."""

    name = "raising-voice"

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def verify(self, audio_path, claimed_id) -> VoiceResult:
        raise self._exc


class RecordingRecommender(ProductRecommender):
    """Records the features it was called with and returns a fixed result."""

    name = "recording-recommender"

    def __init__(self, category: str = "finance", confidence: float = 0.77) -> None:
        self.calls: list[dict] = []
        self._category = category
        self._confidence = confidence

    def predict(self, features) -> ProductRecommendation:
        self.calls.append(dict(features))
        return ProductRecommendation(
            category=self._category, confidence=self._confidence, model_name=self.name
        )


class RaisingRecommender(ProductRecommender):
    """Raises a chosen exception on predict (recommendation failure branch)."""

    name = "raising-recommender"

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def predict(self, features) -> ProductRecommendation:
        raise self._exc


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def customer_id() -> str:
    return CUSTOMER_ID


@pytest.fixture
def build_workflow():
    """Factory to build a workflow from any mix of components (mocks by default)."""

    def _build(
        *,
        face: FaceAuthenticator | None = None,
        voice: VoiceAuthenticator | None = None,
        recommender: ProductRecommender | None = None,
    ) -> AuthenticationWorkflow:
        return AuthenticationWorkflow(
            face_authenticator=face,
            voice_authenticator=voice,
            recommender=recommender,
        )

    return _build


@pytest.fixture
def recording_recommender() -> RecordingRecommender:
    return RecordingRecommender()


@pytest.fixture
def passing_workflow(recording_recommender) -> AuthenticationWorkflow:
    """A workflow where both biometrics pass; recommender records its calls."""
    return AuthenticationWorkflow(
        face_authenticator=MockFaceAuthenticator(always_match=True),
        voice_authenticator=MockVoiceAuthenticator(always_match=True),
        recommender=recording_recommender,
    )


@pytest.fixture
def valid_image(tmp_path: Path) -> Path:
    """A readable image file (valid extension + a PNG header)."""
    path = tmp_path / "face.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    return path


@pytest.fixture
def valid_audio(tmp_path: Path) -> Path:
    """A readable audio file (valid extension)."""
    path = tmp_path / "voice.wav"
    path.write_bytes(b"RIFF....WAVEfmt ")
    return path


@pytest.fixture
def missing_image(tmp_path: Path) -> Path:
    """A path to an image file that does not exist."""
    return tmp_path / "ghost_face.png"


@pytest.fixture
def missing_audio(tmp_path: Path) -> Path:
    """A path to an audio file that does not exist."""
    return tmp_path / "ghost_voice.wav"
