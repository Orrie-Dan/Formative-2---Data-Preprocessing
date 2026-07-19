"""Tests for the authentication state machine.

Run from ``multimodal-authentication/``:

    python -m unittest discover -s tests -v
"""

import unittest

from integration.authentication.state_machine import (
    AuthEvent,
    AuthState,
    AuthStateMachine,
    InvalidTransitionError,
)
from integration.authentication.workflow import AuthState as WorkflowAuthState


class HappyPath(unittest.TestCase):
    def test_full_success_path_reaches_completed(self):
        sm = AuthStateMachine()
        self.assertEqual(sm.state, AuthState.IDLE)

        self.assertEqual(sm.fire(AuthEvent.START), AuthState.FACE_VERIFICATION)
        self.assertEqual(
            sm.fire(AuthEvent.VERIFICATION_PASSED), AuthState.VOICE_VERIFICATION
        )
        self.assertEqual(
            sm.fire(AuthEvent.VERIFICATION_PASSED), AuthState.RECOMMENDATION
        )
        self.assertEqual(sm.fire(AuthEvent.RECOMMENDATION_READY), AuthState.COMPLETED)

        self.assertTrue(sm.is_terminal)
        self.assertEqual(
            sm.history,
            [
                AuthState.IDLE,
                AuthState.FACE_VERIFICATION,
                AuthState.VOICE_VERIFICATION,
                AuthState.RECOMMENDATION,
                AuthState.COMPLETED,
            ],
        )


class DenialPaths(unittest.TestCase):
    def test_face_failure_denies_immediately(self):
        sm = AuthStateMachine()
        sm.fire(AuthEvent.START)
        self.assertEqual(
            sm.fire(AuthEvent.VERIFICATION_FAILED), AuthState.ACCESS_DENIED
        )
        self.assertTrue(sm.is_terminal)
        self.assertNotIn(AuthState.VOICE_VERIFICATION, sm.history)

    def test_voice_failure_denies_after_face(self):
        sm = AuthStateMachine()
        sm.fire(AuthEvent.START)
        sm.fire(AuthEvent.VERIFICATION_PASSED)  # face ok -> voice
        self.assertEqual(
            sm.fire(AuthEvent.VERIFICATION_FAILED), AuthState.ACCESS_DENIED
        )
        self.assertTrue(sm.is_terminal)
        self.assertNotIn(AuthState.RECOMMENDATION, sm.history)


class GuardsAndIntrospection(unittest.TestCase):
    def test_illegal_transition_raises(self):
        sm = AuthStateMachine()
        # Cannot verify before starting.
        with self.assertRaises(InvalidTransitionError):
            sm.fire(AuthEvent.VERIFICATION_PASSED)
        # State is unchanged after a rejected event.
        self.assertEqual(sm.state, AuthState.IDLE)

    def test_firing_from_terminal_state_raises(self):
        sm = AuthStateMachine()
        sm.fire(AuthEvent.START)
        sm.fire(AuthEvent.VERIFICATION_FAILED)  # -> ACCESS_DENIED
        self.assertTrue(sm.is_terminal)
        self.assertEqual(sm.allowed_events(), [])
        with self.assertRaises(InvalidTransitionError):
            sm.fire(AuthEvent.START)

    def test_can_fire_and_allowed_events(self):
        sm = AuthStateMachine()
        self.assertTrue(sm.can_fire(AuthEvent.START))
        self.assertFalse(sm.can_fire(AuthEvent.VERIFICATION_PASSED))
        self.assertEqual(sm.allowed_events(), [AuthEvent.START])

    def test_on_transition_hook_is_called(self):
        seen = []
        sm = AuthStateMachine(
            on_transition=lambda src, ev, dst, ctx: seen.append((src, ev, dst, ctx))
        )
        sm.fire(AuthEvent.START, {"customer_id": "a001"})
        self.assertEqual(
            seen,
            [(AuthState.IDLE, AuthEvent.START, AuthState.FACE_VERIFICATION, {"customer_id": "a001"})],
        )

    def test_reset_returns_to_idle(self):
        sm = AuthStateMachine()
        sm.fire(AuthEvent.START)
        sm.reset()
        self.assertEqual(sm.state, AuthState.IDLE)
        self.assertEqual(sm.history, [AuthState.IDLE])

    def test_transitions_table_is_exposed(self):
        transitions = AuthStateMachine.transitions()
        self.assertIn(
            (AuthState.IDLE, AuthEvent.START, AuthState.FACE_VERIFICATION), transitions
        )
        self.assertEqual(len(transitions), 6)


class WorkflowIntegration(unittest.TestCase):
    """The workflow re-exports and reaches the expected terminal states."""

    def test_workflow_reexports_state_enum(self):
        self.assertIs(WorkflowAuthState, AuthState)

    def test_happy_path_history_from_workflow(self):
        from integration.authentication.workflow import AuthenticationWorkflow

        result = AuthenticationWorkflow().run(customer_id="a001")
        self.assertEqual(result.final_state, AuthState.COMPLETED)
        self.assertEqual(result.history[0], AuthState.IDLE)
        self.assertEqual(result.history[-1], AuthState.COMPLETED)

    def test_face_failure_history_from_workflow(self):
        from integration.authentication.face.mock import MockFaceAuthenticator
        from integration.authentication.workflow import AuthenticationWorkflow

        result = AuthenticationWorkflow(
            face_authenticator=MockFaceAuthenticator(always_match=False),
        ).run(customer_id="a001")
        self.assertEqual(result.final_state, AuthState.ACCESS_DENIED)
        self.assertNotIn(AuthState.VOICE_VERIFICATION, result.history)


if __name__ == "__main__":
    unittest.main()
