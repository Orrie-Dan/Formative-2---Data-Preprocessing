"""Reusable model-loader utilities for the three ML components.

Each real model (face, voice, recommendation) needs the *same* boilerplate:

    1. work out where its artifact lives,
    2. check the file is actually there (and fail cleanly if not),
    3. deserialize it (with joblib or pickle).

Rather than repeat that in every ``real.py``, this module centralizes it behind
a small :class:`ModelLoader` base class with one loader per component.

Current behaviour (by design)
-----------------------------
Out of the box ``load()`` uses the ``"placeholder"`` strategy: it verifies the
model file exists and returns a lightweight :class:`PlaceholderModel` instead of
deserializing anything. This lets the loaders be wired in and tested *before*
any real artifact or heavy dependency exists — exactly the same philosophy as
the mock authenticators.

Future behaviour (already supported)
------------------------------------
Passing ``strategy="joblib"`` / ``"pickle"`` / ``"auto"`` deserializes the real
artifact. The plumbing is here today; a member only flips the strategy once
their trained model is dropped in the matching ``models/<component>/`` folder.

Exception handling
------------------
* A missing file always raises :class:`~integration.errors.ModelNotFoundError`
  with a friendly, component-specific message (never a bare ``FileNotFoundError``).
* Any failure while deserializing an existing file is converted to
  :class:`~integration.errors.ModelLoadError`.
Both are :class:`~integration.errors.IntegrationError` subclasses, so the CLI /
app boundaries already show them as a friendly message instead of a traceback.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from integration.config import IntegrationConfig
from integration.errors import ModelLoadError, ModelNotFoundError

logger = logging.getLogger(__name__)

#: How ``ModelLoader.load`` should turn a file on disk into an object.
LoadStrategy = Literal["placeholder", "joblib", "pickle", "auto"]

#: Extensions we deserialize with joblib in ``"auto"`` mode (else pickle).
_JOBLIB_SUFFIXES = frozenset({".joblib", ".pkl", ".pickle"})


@dataclass(frozen=True)
class PlaceholderModel:
    """Stand-in returned by the default ``"placeholder"`` load strategy.

    It proves the artifact exists and where it is, without deserializing it.
    Callers can check :func:`is_placeholder` to tell it apart from a real model.
    """

    component: str
    path: Path

    def __repr__(self) -> str:  # pragma: no cover - debug convenience
        return f"PlaceholderModel(component={self.component!r}, path={self.path})"


def is_placeholder(obj: Any) -> bool:
    """Whether ``obj`` is a :class:`PlaceholderModel` (not a real loaded model)."""
    return isinstance(obj, PlaceholderModel)


class ModelLoader:
    """Locate, verify and (optionally) deserialize a single model artifact.

    Subclasses set :attr:`component`, :attr:`default_filename` and
    :attr:`friendly_missing`; everything else is shared.
    """

    #: Human-readable component name, used in logs and error details.
    component: str = "model"
    #: Default artifact filename, relative to ``models_dir``.
    default_filename: str = "model.pkl"
    #: Friendly, user-safe message when the artifact is missing.
    friendly_missing: str = "The required model is not available."

    def __init__(self, models_dir: str | Path, *, filename: str | None = None) -> None:
        self.models_dir = Path(models_dir)
        self.filename = filename or self.default_filename

    @property
    def model_path(self) -> Path:
        """Full path to the expected artifact."""
        return self.models_dir / self.filename

    # --- existence checks ------------------------------------------------ #

    def exists(self) -> bool:
        """Whether the artifact file is present (no exception, just a bool)."""
        return self.model_path.is_file()

    def verify_exists(self) -> Path:
        """Return the artifact path, or raise :class:`ModelNotFoundError`.

        This is the "clean exception handling for missing model files" the rest
        of the system relies on: a precise, friendly domain error rather than a
        raw ``FileNotFoundError``.
        """
        path = self.model_path
        if not path.is_file():
            logger.error("%s model file not found at %s", self.component, path)
            raise ModelNotFoundError(
                self.friendly_missing,
                detail=f"expected {self.component} model file at {path}",
            )
        return path

    # --- loading --------------------------------------------------------- #

    def load(self, *, strategy: LoadStrategy = "placeholder") -> Any:
        """Verify the artifact exists, then load it using ``strategy``.

        * ``"placeholder"`` (default) — return a :class:`PlaceholderModel`.
        * ``"joblib"`` / ``"pickle"`` — deserialize with that library.
        * ``"auto"`` — joblib for ``.pkl``/``.joblib``/``.pickle``, else pickle,
          falling back to pickle if joblib cannot read the file.

        Raises:
            ModelNotFoundError: the artifact file does not exist.
            ModelLoadError: the file exists but could not be deserialized.
            ValueError: an unknown ``strategy`` was requested.
        """
        if strategy not in ("placeholder", "joblib", "pickle", "auto"):
            raise ValueError(f"Unknown load strategy: {strategy!r}")

        path = self.verify_exists()

        if strategy == "placeholder":
            logger.info(
                "%s model file found at %s (placeholder load; not deserialized)",
                self.component,
                path,
            )
            return PlaceholderModel(component=self.component, path=path)

        try:
            if strategy == "joblib":
                model = self._load_joblib(path)
            elif strategy == "pickle":
                model = self._load_pickle(path)
            else:  # "auto"
                model = self._load_auto(path)
        except ModelLoadError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface any load failure uniformly
            logger.exception("Failed to load %s model from %s", self.component, path)
            raise ModelLoadError(
                f"The {self.component} model could not be loaded.",
                detail=str(exc),
            ) from exc

        logger.info("%s model loaded from %s (strategy=%s)", self.component, path, strategy)
        return model

    # --- deserializers (lazy imports; only pulled in when actually used) -- #

    @staticmethod
    def _load_joblib(path: Path) -> Any:
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover - joblib ships in requirements
            raise ModelLoadError(
                "joblib is required to load this model.",
                detail=str(exc),
            ) from exc
        return joblib.load(path)

    @staticmethod
    def _load_pickle(path: Path) -> Any:
        with open(path, "rb") as handle:
            return pickle.load(handle)

    def _load_auto(self, path: Path) -> Any:
        if path.suffix.lower() in _JOBLIB_SUFFIXES:
            try:
                return self._load_joblib(path)
            except Exception:  # noqa: BLE001 - fall back to plain pickle
                logger.debug("joblib load failed for %s; falling back to pickle", path)
                return self._load_pickle(path)
        return self._load_pickle(path)

    def __repr__(self) -> str:  # pragma: no cover - debug convenience
        return f"{type(self).__name__}(model_path={self.model_path})"


class FaceModelLoader(ModelLoader):
    """Loader for the facial-recognition model (Member 1)."""

    component = "facial-recognition"
    default_filename = "face_model.pkl"
    friendly_missing = "The facial-recognition model is not available."


class VoiceModelLoader(ModelLoader):
    """Loader for the voice-verification model (Member 2)."""

    component = "voice-verification"
    default_filename = "voice_model.pkl"
    friendly_missing = "The voice-verification model is not available."


class RecommendationModelLoader(ModelLoader):
    """Loader for the product-recommendation model (Member 3)."""

    component = "product-recommendation"
    default_filename = "product_model.pkl"
    friendly_missing = "The product-recommendation model is not available."


# --------------------------------------------------------------------------- #
# Config-driven factories (mirror integration.core.registry conventions)
# --------------------------------------------------------------------------- #
def build_face_loader(
    config: IntegrationConfig | None = None, *, filename: str | None = None
) -> FaceModelLoader:
    """Face loader pointed at the configured ``models/face`` directory."""
    config = config or IntegrationConfig()
    return FaceModelLoader(config.face_model_dir, filename=filename)


def build_voice_loader(
    config: IntegrationConfig | None = None, *, filename: str | None = None
) -> VoiceModelLoader:
    """Voice loader pointed at the configured ``models/voice`` directory."""
    config = config or IntegrationConfig()
    return VoiceModelLoader(config.voice_model_dir, filename=filename)


def build_recommendation_loader(
    config: IntegrationConfig | None = None, *, filename: str | None = None
) -> RecommendationModelLoader:
    """Recommendation loader pointed at ``models/recommendation``."""
    config = config or IntegrationConfig()
    return RecommendationModelLoader(config.recommendation_model_dir, filename=filename)


__all__ = [
    "FaceModelLoader",
    "LoadStrategy",
    "ModelLoader",
    "PlaceholderModel",
    "RecommendationModelLoader",
    "VoiceModelLoader",
    "build_face_loader",
    "build_recommendation_loader",
    "build_voice_loader",
    "is_placeholder",
]
