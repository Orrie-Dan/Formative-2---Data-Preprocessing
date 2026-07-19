"""Tests for the unauthorized-access simulations.

Run from ``multimodal-authentication/``:

    python -m unittest discover -s tests -v
"""

import unittest

import simulate_unauthorized
from integration.simulations import (
    DENIED_EXIT_CODE,
    DENIED_MESSAGE,
    SimulationResult,
    run_all,
    simulate_corrupted_files,
    simulate_invalid_voice,
    simulate_missing_audio,
    simulate_missing_image,
    simulate_unknown_face,
)
from integration.authentication.workflow import ACCESS_DENIED


class EachScenarioDeniesGracefully(unittest.TestCase):
    def _assert_denied(self, result: SimulationResult):
        self.assertIsInstance(result, SimulationResult)
        self.assertFalse(result.granted)  # access denied
        self.assertTrue(result.reason)  # a reason for rejection is given
        self.assertEqual(result.denied_message, DENIED_MESSAGE)  # denied message
        self.assertTrue(result.exited_gracefully)  # graceful exit
        self.assertEqual(result.exit_code, DENIED_EXIT_CODE)
        self.assertIn(ACCESS_DENIED, result.trace)  # attempt is traced to denial

    def test_unknown_face(self):
        result = simulate_unknown_face()
        self._assert_denied(result)
        self.assertIn("recognized", result.reason.lower())
        # Denied at face: voice should never be attempted.
        self.assertNotIn("Voice verification attempted", result.trace)

    def test_invalid_voice(self):
        result = simulate_invalid_voice()
        self._assert_denied(result)
        self.assertIn("voice", result.reason.lower())
        # Face passed first, then voice failed.
        self.assertIn("Face verified", result.trace)

    def test_missing_image(self):
        result = simulate_missing_image()
        self._assert_denied(result)
        self.assertIn("image", result.reason.lower())

    def test_missing_audio(self):
        result = simulate_missing_audio()
        self._assert_denied(result)
        self.assertIn("audio", result.reason.lower())
        self.assertIn("Face verified", result.trace)  # got past face first

    def test_corrupted_files(self):
        result = simulate_corrupted_files()
        self._assert_denied(result)
        self.assertIn("corrupt", result.reason.lower())


class RunAll(unittest.TestCase):
    def test_run_all_covers_five_scenarios_all_denied(self):
        results = run_all()
        self.assertEqual(len(results), 5)
        names = {r.scenario for r in results}
        self.assertEqual(
            names,
            {"Unknown face", "Invalid voice", "Missing image", "Missing audio", "Corrupted files"},
        )
        self.assertTrue(all(not r.granted for r in results))
        self.assertTrue(all(r.exited_gracefully for r in results))


class Runner(unittest.TestCase):
    def test_main_returns_zero_when_all_rejected(self):
        # All scenarios are expected to deny safely -> overall success (0).
        self.assertEqual(simulate_unauthorized.main(["--customer", "a001"]), 0)


if __name__ == "__main__":
    unittest.main()
