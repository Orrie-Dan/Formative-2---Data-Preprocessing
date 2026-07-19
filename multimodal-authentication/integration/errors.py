"""Domain exception hierarchy for the integration layer.

Every *expected* failure mode (missing file, unsupported format, missing model,
data problem, authentication error) is represented by a subclass of
:class:`IntegrationError`. Each carries a ``user_message`` — a short, friendly
sentence safe to show an end user — separate from the technical ``detail`` that
belongs in the logs.

Entry points (CLI, Streamlit app) catch :class:`IntegrationError` and show
``user_message`` instead of a traceback; anything that is *not* an
``IntegrationError`` is treated as an unexpected bug (logged with a full
traceback, shown to the user as a generic message).
"""

from __future__ import annotations


class IntegrationError(Exception):
    """Base class for all known, handled errors in this system."""

    default_message = "Something went wrong. Please try again."

    def __init__(self, user_message: str | None = None, *, detail: str | None = None):
        self.user_message = user_message or self.default_message
        self.detail = detail
        full = self.user_message if detail is None else f"{self.user_message} ({detail})"
        super().__init__(full)


# --- Input / sample problems ------------------------------------------------ #
class SampleError(IntegrationError):
    """Base for problems with a submitted image/audio sample."""

    default_message = "There was a problem with the submitted sample."


class SampleNotFoundError(SampleError):
    """A required image or audio file does not exist (missing image/audio)."""

    default_message = "The requested file could not be found."


class InvalidPathError(SampleError):
    """A path is empty, malformed, or does not point to a file."""

    default_message = "The file path provided is not valid."


class UnsupportedFormatError(SampleError):
    """A file exists but its extension is not a supported format."""

    default_message = "That file format is not supported."


class CorruptedSampleError(SampleError):
    """A file exists with a supported extension but its contents are unreadable."""

    default_message = "The file appears to be corrupted and could not be read."


# --- Model problems --------------------------------------------------------- #
class ModelError(IntegrationError):
    """Base for problems loading or running a model."""

    default_message = "A model could not be used right now."


class ModelNotFoundError(ModelError):
    """A model artifact is missing, or a real model is not available yet."""

    default_message = "The required model is not available."


class ModelLoadError(ModelError):
    """A model artifact exists but failed to load."""

    default_message = "The model could not be loaded."


# --- Data problems ---------------------------------------------------------- #
class DataError(IntegrationError):
    """Required data files are missing or unreadable."""

    default_message = "Required data is missing or could not be read."


# --- Runtime / domain problems --------------------------------------------- #
class AuthenticationError(IntegrationError):
    """Authentication could not be completed due to an error (not a denial)."""

    default_message = "Authentication could not be completed."


class RecommendationError(IntegrationError):
    """The recommendation step failed."""

    default_message = "A product recommendation could not be generated."
