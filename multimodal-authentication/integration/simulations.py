"""Unauthorized-access simulations (no ML — mock verification only).

Demonstrates that the system *rejects* bad authentication attempts safely. Each
scenario runs a real authentication attempt through :class:`AuthenticationWorkflow`
using mock verifiers / crafted sample files, and returns a structured
:class:`SimulationResult` capturing the four things every rejection must show:

    1. Authentication attempt   -> ``attempt``
    2. Reason for rejection     -> ``reason``
    3. Access denied message    -> ``denied_message``
    4. Graceful application exit -> ``exited_gracefully`` / ``exit_code``

Scenarios
---------
* **Unknown face**   — a face that the (mock) model does not recognize.
* **Invalid voice**  — a voice that does not match the enrolled speaker.
* **Missing image**  — the face image file does not exist.
* **Missing audio**  — the voice audio file does not exist.
* **Corrupted files**— a file exists with a valid extension but unreadable bytes.

A simulation *never* raises: every failure is caught and turned into a denied,
gracefully-exited result — exactly how the real CLI/app behave.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

from integration.abstractions import FaceAuthenticator
from integration.authentication.face.mock import MockFaceAuthenticator
from integration.authentication.voice.mock import MockVoiceAuthenticator
from integration.authentication.workflow import ACCESS_DENIED, AuthenticationWorkflow
from integration.core.types import FaceResult
from integration.errors import CorruptedSampleError, IntegrationError
from integration.validation import validate_audio_path, validate_image_path

logger = logging.getLogger(__name__)

#: User-facing message shown when access is refused.
DENIED_MESSAGE = "Access denied. The session has ended securely."

#: Process exit code a CLI would return for a denied attempt (0 == success).
DENIED_EXIT_CODE = 1

# Leading bytes of common image formats; used by the corruption check (no ML,
# just a cheap integrity probe on the file header).
_IMAGE_MAGIC = (b"\x89PNG", b"\xff\xd8\xff", b"GIF87a", b"GIF89a", b"BM", b"RIFF")

# A valid tiny PNG header so scenarios that need a *readable* image have one.
_VALID_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


@dataclass(frozen=True)
class SimulationResult:
    """Outcome of one unauthorized-access simulation."""

    scenario: str
    attempt: str  # what was tried (the authentication attempt)
    granted: bool  # always False for these simulations
    reason: str  # why it was rejected
    denied_message: str  # the access-denied message shown to the user
    exited_gracefully: bool  # True == handled cleanly, no traceback leaked
    exit_code: int  # process exit code (DENIED_EXIT_CODE for a denial)
    trace: list[str] = field(default_factory=list)


class _CorruptAwareFaceAuthenticator(FaceAuthenticator):
    """Mock face verifier that rejects corrupted image bytes.

    This is *not* ML: it only checks the file header looks like a real image and
    raises :class:`CorruptedSampleError` otherwise, so the corrupted-file
    scenario has something deterministic to fail on.
    """

    name = "corrupt-aware-face"

    def verify(self, image_path, claimed_id) -> FaceResult:
        try:
            data = Path(image_path).read_bytes()
        except OSError as exc:
            raise CorruptedSampleError(
                "The image file could not be read.", detail=str(exc)
            ) from exc
        if not any(data.startswith(magic) for magic in _IMAGE_MAGIC):
            raise CorruptedSampleError(
                "The image file appears to be corrupted.", detail=str(image_path)
            )
        return FaceResult(
            matched=True, confidence=1.0, person_id=str(claimed_id), model_name=self.name
        )


# --------------------------------------------------------------------------- #
# Core attempt runner (shared by every scenario)
# --------------------------------------------------------------------------- #
def _deny(scenario: str, attempt: str, reason: str, trace: list[str]) -> SimulationResult:
    return SimulationResult(
        scenario=scenario,
        attempt=attempt,
        granted=False,
        reason=reason,
        denied_message=DENIED_MESSAGE,
        exited_gracefully=True,
        exit_code=DENIED_EXIT_CODE,
        trace=[*trace, ACCESS_DENIED],
    )


def _run_attempt(
    *,
    scenario: str,
    attempt: str,
    workflow: AuthenticationWorkflow,
    customer_id: str,
    image_path: str | Path | None = None,
    audio_path: str | Path | None = None,
    validate_media: bool = False,
) -> SimulationResult:
    """Run one attempt (face -> voice) and translate any failure into a denial.

    Mirrors the CLI's short-circuiting flow and its "never leak a traceback"
    boundary, but returns structured data instead of printing.
    """
    trace: list[str] = ["Authentication started"]
    try:
        # --- Face ---------------------------------------------------------
        trace.append("Face verification attempted")
        face_sample: str | Path = image_path if image_path is not None else "<mock-face-sample>"
        if validate_media and image_path is not None:
            face_sample = validate_image_path(image_path)  # may raise SampleError
        face = workflow.verify_face(face_sample, customer_id)
        if not face.matched:
            return _deny(scenario, attempt, "Face not recognized (unknown identity).", trace)
        trace.append("Face verified")

        # --- Voice --------------------------------------------------------
        trace.append("Voice verification attempted")
        voice_sample: str | Path = audio_path if audio_path is not None else "<mock-voice-sample>"
        if validate_media and audio_path is not None:
            voice_sample = validate_audio_path(audio_path)  # may raise SampleError
        voice = workflow.verify_voice(voice_sample, customer_id)
        if not voice.matched:
            return _deny(scenario, attempt, "Voice did not match the enrolled speaker.", trace)
        trace.append("Voice verified")

        # Not expected in these unauthorized simulations.
        return SimulationResult(
            scenario=scenario,
            attempt=attempt,
            granted=True,
            reason="",
            denied_message="",
            exited_gracefully=True,
            exit_code=0,
            trace=[*trace, "Access granted"],
        )

    except IntegrationError as exc:
        # Known, handled problem: show the friendly message, exit gracefully.
        logger.warning("%s: handled rejection: %s", scenario, exc)
        trace.append(f"Handled {type(exc).__name__}")
        return _deny(scenario, attempt, exc.user_message, trace)
    except Exception:  # noqa: BLE001 - last-resort guard; never leak a traceback
        logger.exception("%s: unexpected error (handled gracefully)", scenario)
        trace.append("Handled unexpected error")
        return _deny(scenario, attempt, "An unexpected problem occurred.", trace)


# --------------------------------------------------------------------------- #
# The five scenarios
# --------------------------------------------------------------------------- #
def simulate_unknown_face(customer_id: str = "a001") -> SimulationResult:
    """A face the model does not recognize -> denied at the face stage."""
    workflow = AuthenticationWorkflow(
        face_authenticator=MockFaceAuthenticator(always_match=False),
        voice_authenticator=MockVoiceAuthenticator(always_match=True),
    )
    return _run_attempt(
        scenario="Unknown face",
        attempt=f"Present an unrecognized face for customer '{customer_id}'.",
        workflow=workflow,
        customer_id=customer_id,
    )


def simulate_invalid_voice(customer_id: str = "a001") -> SimulationResult:
    """Face passes, but the voice does not match -> denied at the voice stage."""
    workflow = AuthenticationWorkflow(
        face_authenticator=MockFaceAuthenticator(always_match=True),
        voice_authenticator=MockVoiceAuthenticator(always_match=False),
    )
    return _run_attempt(
        scenario="Invalid voice",
        attempt=f"Present a mismatched voice sample for customer '{customer_id}'.",
        workflow=workflow,
        customer_id=customer_id,
    )


def simulate_missing_image(customer_id: str = "a001") -> SimulationResult:
    """The face image file does not exist -> denied before any verification."""
    workflow = AuthenticationWorkflow(
        face_authenticator=MockFaceAuthenticator(always_match=True),
        voice_authenticator=MockVoiceAuthenticator(always_match=True),
    )
    with TemporaryDirectory() as tmp:
        missing = Path(tmp) / "ghost_face.png"  # never created
        return _run_attempt(
            scenario="Missing image",
            attempt=f"Submit a non-existent face image: {missing.name}",
            workflow=workflow,
            customer_id=customer_id,
            image_path=missing,
            validate_media=True,
        )


def simulate_missing_audio(customer_id: str = "a001") -> SimulationResult:
    """Face is fine, but the voice audio file does not exist -> denied."""
    workflow = AuthenticationWorkflow(
        face_authenticator=MockFaceAuthenticator(always_match=True),
        voice_authenticator=MockVoiceAuthenticator(always_match=True),
    )
    with TemporaryDirectory() as tmp:
        image = Path(tmp) / "face.png"
        image.write_bytes(_VALID_PNG_BYTES)  # a readable image so face passes
        missing_audio = Path(tmp) / "ghost_voice.wav"  # never created
        return _run_attempt(
            scenario="Missing audio",
            attempt=f"Submit a non-existent voice audio: {missing_audio.name}",
            workflow=workflow,
            customer_id=customer_id,
            image_path=image,
            audio_path=missing_audio,
            validate_media=True,
        )


def simulate_corrupted_files(customer_id: str = "a001") -> SimulationResult:
    """A file with a valid extension but unreadable bytes -> denied."""
    workflow = AuthenticationWorkflow(
        face_authenticator=_CorruptAwareFaceAuthenticator(),
        voice_authenticator=MockVoiceAuthenticator(always_match=True),
    )
    with TemporaryDirectory() as tmp:
        corrupt = Path(tmp) / "corrupt.png"
        corrupt.write_bytes(b"\x00\x01\x02 not a real image \xff")
        return _run_attempt(
            scenario="Corrupted files",
            attempt=f"Submit a corrupted image file: {corrupt.name}",
            workflow=workflow,
            customer_id=customer_id,
            image_path=corrupt,
            validate_media=True,  # passes extension/existence; verify detects corruption
        )


#: Ordered registry so the runner and tests share one source of truth.
SCENARIOS = (
    simulate_unknown_face,
    simulate_invalid_voice,
    simulate_missing_image,
    simulate_missing_audio,
    simulate_corrupted_files,
)


def run_all(customer_id: str = "a001") -> list[SimulationResult]:
    """Run every unauthorized-access scenario and collect the results."""
    return [scenario(customer_id) for scenario in SCENARIOS]


__all__ = [
    "DENIED_EXIT_CODE",
    "DENIED_MESSAGE",
    "SCENARIOS",
    "SimulationResult",
    "run_all",
    "simulate_corrupted_files",
    "simulate_invalid_voice",
    "simulate_missing_audio",
    "simulate_missing_image",
    "simulate_unknown_face",
]
