"""Authentication workflow, driven by an explicit state machine.

Implements exactly this flow (no ML models — mock verifiers that pass):

    IDLE
      -> FACE_VERIFICATION   --fail--> ACCESS_DENIED (terminal)
      -> VOICE_VERIFICATION  --fail--> ACCESS_DENIED (terminal)
      -> RECOMMENDATION
      -> COMPLETED (terminal)

The *control flow* (which state follows which, and when) lives entirely in
:class:`~integration.authentication.state_machine.AuthStateMachine`. This module
supplies the *behaviour* for each state (acquire a sample, call a verifier, run
the recommender) and feeds the machine events based on the results. Keeping the
two concerns apart means the sequence can be extended (e.g. add a modality) by
editing the transition table, not this file.

Design goals
------------
* **Explicit states** : every transition is declared once in the state machine;
  ``run`` just performs the action for the current state and fires an event.
* **Swappable** : verification is delegated to the ``FaceAuthenticator`` /
  ``VoiceAuthenticator`` contracts and recommendation to ``ProductRecommender``.
  The defaults are mocks that always pass; replace them with real models by
  passing real implementations (or via ``from_config`` + env vars) — the
  workflow code does not change.
* **Testable**  : ``run`` returns a :class:`WorkflowResult` carrying the full
  state ``history`` and a human-readable ``trace``, and never raises on a failed
  check, so both the happy path and the "Access Denied" short-circuits are easy
  to assert.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from integration.abstractions import (
    FaceAuthenticator,
    ProductRecommender,
    VoiceAuthenticator,
)
from integration.authentication.state_machine import (
    AuthEvent,
    AuthState,
    AuthStateMachine,
)
from integration.config import IntegrationConfig
from integration.core.types import FaceResult, ProductRecommendation, VoiceResult

logger = logging.getLogger(__name__)

ACCESS_DENIED = "Access Denied"
ACCESS_GRANTED = "Access Granted"


class Stage(str, Enum):
    """Where the workflow ended up (kept for backwards-compatible results).

    This is a coarse view of the final :class:`AuthState`; new code should
    prefer :attr:`WorkflowResult.final_state` / :attr:`WorkflowResult.history`.
    """

    FACE = "face"
    VOICE = "voice"
    RECOMMENDATION = "recommendation"
    COMPLETE = "complete"


# Which stage a failed verification state maps to (for WorkflowResult.stage).
_STATE_TO_FAIL_STAGE: dict[AuthState, Stage] = {
    AuthState.FACE_VERIFICATION: Stage.FACE,
    AuthState.VOICE_VERIFICATION: Stage.VOICE,
}


def _verification_event(passed: bool) -> AuthEvent:
    """Map a verifier's pass/fail into the generic verification event."""
    return AuthEvent.VERIFICATION_PASSED if passed else AuthEvent.VERIFICATION_FAILED


@dataclass(frozen=True)
class WorkflowResult:
    """Outcome of one run of the authentication workflow."""

    granted: bool
    message: str  # ACCESS_GRANTED or ACCESS_DENIED
    stage: Stage  # the stage reached (failed stage if denied)
    face: FaceResult | None = None
    voice: VoiceResult | None = None
    recommendation: ProductRecommendation | None = None
    trace: list[str] = field(default_factory=list)
    final_state: AuthState = AuthState.IDLE  # terminal state the machine reached
    history: list[AuthState] = field(default_factory=list)  # full state path


class AuthenticationWorkflow:
    """Runs face -> voice -> recommendation as a short-circuiting sequence."""

    def __init__(
        self,
        face_authenticator: FaceAuthenticator | None = None,
        voice_authenticator: VoiceAuthenticator | None = None,
        recommender: ProductRecommender | None = None,
    ) -> None:
        # Defaults are mocks that ALWAYS pass, so the workflow runs end-to-end
        # today. Swap in real models here (or via `from_config`) later.
        from integration.authentication.face.mock import MockFaceAuthenticator
        from integration.authentication.voice.mock import MockVoiceAuthenticator
        from integration.recommendation.mock import MockProductRecommender

        self.face_authenticator = face_authenticator or MockFaceAuthenticator(always_match=True)
        self.voice_authenticator = voice_authenticator or MockVoiceAuthenticator(always_match=True)
        self.recommender = recommender or MockProductRecommender()

    @classmethod
    def from_config(cls, config: IntegrationConfig | None = None) -> "AuthenticationWorkflow":
        """Build the workflow from config (route to real models via env vars)."""
        from integration.core.registry import (
            build_face_authenticator,
            build_recommender,
            build_voice_authenticator,
        )

        config = config or IntegrationConfig()
        return cls(
            face_authenticator=build_face_authenticator(config),
            voice_authenticator=build_voice_authenticator(config),
            recommender=build_recommender(config),
        )

    # --- individual steps (each is small, named, and independently testable) ---

    def submit_face(self, image_path: str | Path | None) -> str | Path:
        """Acquire the face sample. Mock: just echo the provided path."""
        sample = image_path if image_path is not None else "<mock-face-sample>"
        logger.info("Image loaded (source=%s)", sample)
        return sample

    def verify_face(self, sample: str | Path, customer_id: str) -> FaceResult:
        logger.info("Face verification started (customer=%s)", customer_id)
        result = self.face_authenticator.verify(sample, customer_id)
        if result.matched:
            logger.info(
                "Face verification passed (customer=%s, confidence=%.2f)",
                customer_id,
                result.confidence,
            )
        else:
            logger.warning(
                "Face verification failed (customer=%s, confidence=%.2f)",
                customer_id,
                result.confidence,
            )
        return result

    def submit_voice(self, audio_path: str | Path | None) -> str | Path:
        """Acquire the voice sample. Mock: just echo the provided path."""
        sample = audio_path if audio_path is not None else "<mock-voice-sample>"
        logger.info("Audio loaded (source=%s)", sample)
        return sample

    def verify_voice(self, sample: str | Path, customer_id: str) -> VoiceResult:
        logger.info("Voice verification started (customer=%s)", customer_id)
        result = self.voice_authenticator.verify(sample, customer_id)
        if result.matched:
            logger.info(
                "Voice verification passed (customer=%s, confidence=%.2f)",
                customer_id,
                result.confidence,
            )
        else:
            logger.warning(
                "Voice verification failed (customer=%s, confidence=%.2f)",
                customer_id,
                result.confidence,
            )
        return result

    def recommend(self, features: Mapping[str, Any]) -> ProductRecommendation:
        recommendation = self.recommender.predict(features)
        logger.info(
            "Recommendation generated (category=%s, confidence=%.2f)",
            recommendation.category,
            recommendation.confidence,
        )
        return recommendation

    # --- the sequence ---

    def run(
        self,
        customer_id: str,
        image_path: str | Path | None = None,
        audio_path: str | Path | None = None,
        features: Mapping[str, Any] | None = None,
    ) -> WorkflowResult:
        """Drive the state machine IDLE -> FACE -> VOICE -> RECOMMENDATION.

        The machine decides *what state comes next*; this method supplies the
        behaviour for each state and fires the resulting event. On a failed
        check the machine moves to ``ACCESS_DENIED`` and the loop stops with
        ``message == ACCESS_DENIED``. Never raises on a failed check.
        """
        trace: list[str] = ["Start"]
        features = dict(features or {"customer_id": customer_id})

        machine = AuthStateMachine(
            on_transition=lambda src, event, dst, ctx: logger.debug(
                "State transition: %s --%s--> %s", src.value, event.value, dst.value
            )
        )

        face: FaceResult | None = None
        voice: VoiceResult | None = None
        recommendation: ProductRecommendation | None = None

        # IDLE -> FACE_VERIFICATION
        machine.fire(AuthEvent.START)

        # Run until a terminal state (ACCESS_DENIED or COMPLETED) is reached.
        while not machine.is_terminal:
            if machine.state is AuthState.FACE_VERIFICATION:
                face_sample = self.submit_face(image_path)
                trace.append("Submit Face")
                face = self.verify_face(face_sample, customer_id)
                trace.append(f"Verify Face -> {'pass' if face.matched else 'fail'}")
                machine.fire(_verification_event(face.matched), {"face": face})

            elif machine.state is AuthState.VOICE_VERIFICATION:
                voice_sample = self.submit_voice(audio_path)
                trace.append("Submit Voice")
                voice = self.verify_voice(voice_sample, customer_id)
                trace.append(f"Verify Voice -> {'pass' if voice.matched else 'fail'}")
                machine.fire(_verification_event(voice.matched), {"voice": voice})

            elif machine.state is AuthState.RECOMMENDATION:
                trace.append("Run Product Recommendation")
                recommendation = self.recommend(features)
                trace.append(f"Display Recommendation -> {recommendation.category}")
                machine.fire(AuthEvent.RECOMMENDATION_READY, {"recommendation": recommendation})

        return self._result_for(
            machine, trace=trace, face=face, voice=voice, recommendation=recommendation
        )

    @staticmethod
    def _result_for(
        machine: AuthStateMachine,
        *,
        trace: list[str],
        face: FaceResult | None,
        voice: VoiceResult | None,
        recommendation: ProductRecommendation | None,
    ) -> WorkflowResult:
        """Package the terminal machine state into a :class:`WorkflowResult`."""
        granted = machine.state is AuthState.COMPLETED
        if granted:
            message, stage = ACCESS_GRANTED, Stage.COMPLETE
        else:
            message = ACCESS_DENIED
            trace.append(ACCESS_DENIED)
            # The state *before* ACCESS_DENIED is the one that failed.
            failed_state = machine.history[-2]
            stage = _STATE_TO_FAIL_STAGE.get(failed_state, Stage.FACE)

        return WorkflowResult(
            granted=granted,
            message=message,
            stage=stage,
            face=face,
            voice=voice,
            recommendation=recommendation,
            trace=trace,
            final_state=machine.state,
            history=list(machine.history),
        )


def run_demo(customer_id: str = "a001") -> WorkflowResult:
    """Console demo of the workflow using the always-pass mocks."""
    workflow = AuthenticationWorkflow()
    result = workflow.run(customer_id=customer_id)
    print(" -> ".join(result.trace))
    print(f"\nResult: {result.message}")
    if result.recommendation is not None:
        print(f"Recommended product: {result.recommendation.category}")
    return result


if __name__ == "__main__":
    run_demo()
