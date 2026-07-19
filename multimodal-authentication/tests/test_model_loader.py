"""Tests for the reusable model-loader utilities.

Run from ``multimodal-authentication/``:

    python -m unittest discover -s tests -v
"""

import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from integration.config import IntegrationConfig, Paths
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
from integration.errors import ModelLoadError, ModelNotFoundError

LOADER_CASES = (
    (FaceModelLoader, "facial-recognition", "face_model.pkl"),
    (VoiceModelLoader, "voice-verification", "voice_model.pkl"),
    (RecommendationModelLoader, "product-recommendation", "product_model.pkl"),
)


class MissingFile(unittest.TestCase):
    def test_each_loader_raises_model_not_found(self):
        with TemporaryDirectory() as d:
            for loader_cls, _component, _filename in LOADER_CASES:
                loader = loader_cls(d)
                self.assertFalse(loader.exists())
                with self.assertRaises(ModelNotFoundError):
                    loader.verify_exists()
                with self.assertRaises(ModelNotFoundError):
                    loader.load()

    def test_missing_error_is_friendly_and_has_detail(self):
        with TemporaryDirectory() as d:
            try:
                FaceModelLoader(d).load()
            except ModelNotFoundError as exc:
                self.assertTrue(exc.user_message)
                self.assertNotIn("Traceback", exc.user_message)
                self.assertIn("face_model.pkl", exc.detail)

    def test_default_filenames_and_paths(self):
        with TemporaryDirectory() as d:
            for loader_cls, component, filename in LOADER_CASES:
                loader = loader_cls(d)
                self.assertEqual(loader.component, component)
                self.assertEqual(loader.model_path, Path(d) / filename)


class PlaceholderStrategy(unittest.TestCase):
    def test_existing_file_returns_placeholder(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "face_model.pkl"
            path.write_bytes(b"not-a-real-model")
            loader = FaceModelLoader(d)
            self.assertTrue(loader.exists())

            result = loader.load()  # default strategy == "placeholder"
            self.assertIsInstance(result, PlaceholderModel)
            self.assertTrue(is_placeholder(result))
            self.assertEqual(result.component, "facial-recognition")
            self.assertEqual(result.path, path)

    def test_placeholder_does_not_deserialize(self):
        # A file that is NOT valid pickle must still load as a placeholder.
        with TemporaryDirectory() as d:
            path = Path(d) / "voice_model.pkl"
            path.write_bytes(b"\x00\x01garbage")
            result = VoiceModelLoader(d).load(strategy="placeholder")
            self.assertTrue(is_placeholder(result))


class RealDeserialization(unittest.TestCase):
    def test_pickle_strategy_round_trips(self):
        payload = {"weights": [1, 2, 3], "name": "demo"}
        with TemporaryDirectory() as d:
            path = Path(d) / "product_model.pkl"
            with open(path, "wb") as fh:
                pickle.dump(payload, fh)
            loaded = RecommendationModelLoader(d).load(strategy="pickle")
            self.assertEqual(loaded, payload)
            self.assertFalse(is_placeholder(loaded))

    def test_auto_strategy_reads_pickle_pkl(self):
        payload = ["a", "b", "c"]
        with TemporaryDirectory() as d:
            path = Path(d) / "face_model.pkl"
            with open(path, "wb") as fh:
                pickle.dump(payload, fh)
            loaded = FaceModelLoader(d).load(strategy="auto")
            self.assertEqual(loaded, payload)

    def test_joblib_strategy_round_trips(self):
        joblib = _import_or_skip("joblib")
        payload = {"model": "tree", "depth": 5}
        with TemporaryDirectory() as d:
            path = Path(d) / "product_model.pkl"
            joblib.dump(payload, path)
            loaded = RecommendationModelLoader(d).load(strategy="joblib")
            self.assertEqual(loaded, payload)

    def test_corrupt_file_raises_model_load_error(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "voice_model.pkl"
            path.write_bytes(b"\x00\x01 not a pickle")
            with self.assertRaises(ModelLoadError):
                VoiceModelLoader(d).load(strategy="pickle")

    def test_unknown_strategy_raises_value_error(self):
        with TemporaryDirectory() as d:
            (Path(d) / "face_model.pkl").write_bytes(b"x")
            with self.assertRaises(ValueError):
                FaceModelLoader(d).load(strategy="bogus")  # type: ignore[arg-type]


class CustomFilenameAndConfig(unittest.TestCase):
    def test_custom_filename_override(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "encoder.pkl"
            path.write_bytes(b"x")
            loader = ModelLoader(d, filename="encoder.pkl")
            self.assertEqual(loader.model_path, path)
            self.assertTrue(is_placeholder(loader.load()))

    def test_build_from_config_uses_configured_dirs(self):
        with TemporaryDirectory() as d:
            base = Path(d)
            config = IntegrationConfig(
                paths=Paths(
                    base_dir=base,
                    face_model_dir=base / "f",
                    voice_model_dir=base / "v",
                    recommendation_model_dir=base / "r",
                )
            )
            self.assertEqual(build_face_loader(config).models_dir, base / "f")
            self.assertEqual(build_voice_loader(config).models_dir, base / "v")
            self.assertEqual(
                build_recommendation_loader(config).models_dir, base / "r"
            )


def _import_or_skip(module_name: str):
    try:
        return __import__(module_name)
    except ImportError:  # pragma: no cover - only when optional dep is absent
        raise unittest.SkipTest(f"{module_name} not installed")


if __name__ == "__main__":
    unittest.main()
