"""Pytest framework for the authentication workflow.

Covers every branch of :meth:`AuthenticationWorkflow.run` and the required
scenarios, using mock/test-double components only — **no ML is exercised**:

    * Successful authentication      (face pass -> voice pass -> recommend)
    * Unknown face                   (face fail -> denied, short-circuit)
    * Unknown voice                  (voice fail -> denied, short-circuit)
    * Missing image                  (image file absent)
    * Missing audio                  (audio file absent)
    * Missing models                 (verifier/recommender reports no model)
    * Unexpected exceptions          (verifier crashes; service wraps it)
    * Recommendation execution       (recommender invoked with features)

Test doubles and fixtures live in ``tests/conftest.py``.
"""

from __future__ import annotations

import pytest

from integration.abstractions import FaceAuthenticator
from integration.authentication.face.mock import MockFaceAuthenticator
from integration.authentication.risk import HeuristicRiskScorer
from integration.authentication.service import AuthenticationService
from integration.authentication.state_machine import AuthState
from integration.authentication.voice.mock import MockVoiceAuthenticator
from integration.authentication.workflow import (
    ACCESS_DENIED,
    ACCESS_GRANTED,
    Stage,
)
from integration.config import Thresholds
from integration.errors import (
    AuthenticationError,
    ModelNotFoundError,
    SampleNotFoundError,
)

from tests.conftest import (
    RaisingFaceAuthenticator,
    RaisingRecommender,
    RaisingVoiceAuthenticator,
    RecordingRecommender,
    ValidatingFaceAuthenticator,
    ValidatingVoiceAuthenticator,
)


# --------------------------------------------------------------------------- #
# 1. Successful authentication
# --------------------------------------------------------------------------- #
def test_successful_authentication(passing_workflow, recording_recommender, customer_id):
    result = passing_workflow.run(customer_id=customer_id)

    assert result.granted is True
    assert result.message == ACCESS_GRANTED
    assert result.stage == Stage.COMPLETE
    assert result.final_state == AuthState.COMPLETED
    assert result.face is not None and result.face.matched
    assert result.voice is not None and result.voice.matched
    assert result.recommendation is not None
    # The recommender ran exactly once (recommendation execution).
    assert len(recording_recommender.calls) == 1


def test_successful_authentication_state_path(passing_workflow, customer_id):
    result = passing_workflow.run(customer_id=customer_id)
    assert result.history == [
        AuthState.IDLE,
        AuthState.FACE_VERIFICATION,
        AuthState.VOICE_VERIFICATION,
        AuthState.RECOMMENDATION,
        AuthState.COMPLETED,
    ]


# --------------------------------------------------------------------------- #
# 2. Unknown face  (face-fail branch, short-circuits before voice)
# --------------------------------------------------------------------------- #
def test_unknown_face_denies_and_short_circuits(build_workflow, customer_id):
    recorder = RecordingRecommender()
    workflow = build_workflow(
        face=MockFaceAuthenticator(always_match=False),
        voice=MockVoiceAuthenticator(always_match=True),
        recommender=recorder,
    )
    result = workflow.run(customer_id=customer_id)

    assert result.granted is False
    assert result.message == ACCESS_DENIED
    assert result.stage == Stage.FACE
    assert result.final_state == AuthState.ACCESS_DENIED
    # Voice + recommendation must never run once face fails.
    assert result.voice is None
    assert result.recommendation is None
    assert recorder.calls == []
    assert AuthState.VOICE_VERIFICATION not in result.history


# --------------------------------------------------------------------------- #
# 3. Unknown voice  (voice-fail branch, after face passes)
# --------------------------------------------------------------------------- #
def test_unknown_voice_denies_after_face(build_workflow, customer_id):
    recorder = RecordingRecommender()
    workflow = build_workflow(
        face=MockFaceAuthenticator(always_match=True),
        voice=MockVoiceAuthenticator(always_match=False),
        recommender=recorder,
    )
    result = workflow.run(customer_id=customer_id)

    assert result.granted is False
    assert result.stage == Stage.VOICE
    assert result.final_state == AuthState.ACCESS_DENIED
    assert result.face is not None and result.face.matched
    assert result.recommendation is None
    assert recorder.calls == []  # recommendation never runs on denial
    assert AuthState.RECOMMENDATION not in result.history


# --------------------------------------------------------------------------- #
# 4. Missing image
# --------------------------------------------------------------------------- #
def test_missing_image_raises_sample_not_found(build_workflow, missing_image, customer_id):
    workflow = build_workflow(
        face=ValidatingFaceAuthenticator(),
        voice=MockVoiceAuthenticator(always_match=True),
    )
    with pytest.raises(SampleNotFoundError):
        workflow.run(customer_id=customer_id, image_path=missing_image)


# --------------------------------------------------------------------------- #
# 5. Missing audio  (face passes first, then audio is absent)
# --------------------------------------------------------------------------- #
def test_missing_audio_raises_sample_not_found(
    build_workflow, valid_image, missing_audio, customer_id
):
    workflow = build_workflow(
        face=ValidatingFaceAuthenticator(),
        voice=ValidatingVoiceAuthenticator(),
    )
    with pytest.raises(SampleNotFoundError):
        workflow.run(
            customer_id=customer_id, image_path=valid_image, audio_path=missing_audio
        )


# --------------------------------------------------------------------------- #
# 6. Missing models  (each component can report "no model available")
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("stage", ["face", "voice", "recommender"])
def test_missing_model_surfaces_model_not_found(build_workflow, customer_id, stage):
    err = ModelNotFoundError("model missing")
    face = MockFaceAuthenticator(always_match=True)
    voice = MockVoiceAuthenticator(always_match=True)
    recommender = None

    if stage == "face":
        face = RaisingFaceAuthenticator(err)
    elif stage == "voice":
        voice = RaisingVoiceAuthenticator(err)
    else:
        recommender = RaisingRecommender(err)

    workflow = build_workflow(face=face, voice=voice, recommender=recommender)
    with pytest.raises(ModelNotFoundError):
        workflow.run(customer_id=customer_id)


# --------------------------------------------------------------------------- #
# 7. Unexpected exceptions
# --------------------------------------------------------------------------- #
def test_unexpected_exception_propagates_from_workflow(build_workflow, customer_id):
    # The workflow surfaces unexpected errors to its caller (the CLI/service
    # boundary is where they are turned into friendly messages).
    workflow = build_workflow(face=RaisingFaceAuthenticator(RuntimeError("gpu on fire")))
    with pytest.raises(RuntimeError):
        workflow.run(customer_id=customer_id)


def test_service_wraps_unexpected_exception_gracefully(customer_id):
    # The authentication *service* converts a crashing verifier into a handled
    # AuthenticationError (never leaking the raw exception).
    class Boom(FaceAuthenticator):
        name = "boom"

        def verify(self, image_path, claimed_id):
            raise RuntimeError("driver crashed")

    service = AuthenticationService(
        risk_scorer=HeuristicRiskScorer(),
        face_authenticator=Boom(),
        voice_authenticator=MockVoiceAuthenticator(always_match=True),
        thresholds=Thresholds(),
    )
    with pytest.raises(AuthenticationError):
        service.authenticate({"customer_id": customer_id}, customer_id, image_path="x.jpg")


# --------------------------------------------------------------------------- #
# 8. Recommendation execution
# --------------------------------------------------------------------------- #
def test_recommendation_executes_with_features(build_workflow, customer_id):
    recorder = RecordingRecommender(category="travel", confidence=0.91)
    workflow = build_workflow(
        face=MockFaceAuthenticator(always_match=True),
        voice=MockVoiceAuthenticator(always_match=True),
        recommender=recorder,
    )
    features = {"customer_id": customer_id, "fraud_rate": 0.1}
    result = workflow.run(customer_id=customer_id, features=features)

    assert result.recommendation is not None
    assert result.recommendation.category == "travel"
    assert result.recommendation.confidence == pytest.approx(0.91)
    assert "Run Product Recommendation" in result.trace
    # The recommender received the features (recommendation execution).
    assert recorder.calls == [features]


def test_recommendation_failure_propagates(build_workflow, customer_id):
    workflow = build_workflow(
        face=MockFaceAuthenticator(always_match=True),
        voice=MockVoiceAuthenticator(always_match=True),
        recommender=RaisingRecommender(RuntimeError("no model")),
    )
    with pytest.raises(RuntimeError):
        workflow.run(customer_id=customer_id)


# --------------------------------------------------------------------------- #
# Branch coverage — every (face, voice) outcome reaches the right terminal state
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("face_ok", "voice_ok", "expected_state", "expected_stage", "granted"),
    [
        (True, True, AuthState.COMPLETED, Stage.COMPLETE, True),
        (False, True, AuthState.ACCESS_DENIED, Stage.FACE, False),
        (True, False, AuthState.ACCESS_DENIED, Stage.VOICE, False),
    ],
)
def test_every_branch_reaches_expected_terminal_state(
    build_workflow, customer_id, face_ok, voice_ok, expected_state, expected_stage, granted
):
    workflow = build_workflow(
        face=MockFaceAuthenticator(always_match=face_ok),
        voice=MockVoiceAuthenticator(always_match=voice_ok),
    )
    result = workflow.run(customer_id=customer_id)
    assert result.final_state == expected_state
    assert result.stage == expected_stage
    assert result.granted is granted
