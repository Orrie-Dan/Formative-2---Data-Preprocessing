"""Authentication workflow modelled as an explicit finite state machine.

This module owns the *control flow* of authentication — which state comes next
and under what condition — and nothing else. It contains **no machine learning
and no I/O**: it only knows about states and the events that move between them.
The workflow (``workflow.py``) feeds it events derived from the face/voice
verifiers and the recommender, so the "what happens next" logic lives in exactly
one place.

Assignment workflow (the transitions implemented below)::

    IDLE
      --START-->                FACE_VERIFICATION
    FACE_VERIFICATION
      --VERIFICATION_PASSED-->  VOICE_VERIFICATION
      --VERIFICATION_FAILED-->  ACCESS_DENIED        (terminal)
    VOICE_VERIFICATION
      --VERIFICATION_PASSED-->  RECOMMENDATION
      --VERIFICATION_FAILED-->  ACCESS_DENIED        (terminal)
    RECOMMENDATION
      --RECOMMENDATION_READY--> COMPLETED            (terminal)

Design goals
------------
* **Explicit** : every legal transition is declared once in ``_TRANSITIONS``;
  anything not in that table is rejected. There is no hidden control flow.
* **Extensible** : adding a modality (e.g. a fingerprint check) is a new
  :class:`AuthState` plus a couple of rows in the transition table — no existing
  branch is edited. A generic ``VERIFICATION_PASSED`` / ``VERIFICATION_FAILED``
  event pair is reused by every verification state.
* **Observable** : an optional ``on_transition`` hook fires on every move, and
  :attr:`AuthStateMachine.history` records the full path for tracing/tests.
* **Safe**    : illegal transitions raise :class:`InvalidTransitionError` rather
  than silently doing nothing, so bugs surface immediately.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import Enum
from typing import Any, Optional


class AuthState(str, Enum):
    """Every state the authentication workflow can occupy."""

    IDLE = "IDLE"
    FACE_VERIFICATION = "FACE_VERIFICATION"
    VOICE_VERIFICATION = "VOICE_VERIFICATION"
    RECOMMENDATION = "RECOMMENDATION"
    ACCESS_DENIED = "ACCESS_DENIED"
    COMPLETED = "COMPLETED"


class AuthEvent(str, Enum):
    """Signals that drive transitions between :class:`AuthState` values.

    The verification events are intentionally *generic* (not "face passed" /
    "voice passed") so any number of verification states can reuse them; the
    transition table decides what a pass/fail means from the current state.
    """

    START = "START"
    VERIFICATION_PASSED = "VERIFICATION_PASSED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    RECOMMENDATION_READY = "RECOMMENDATION_READY"


class StateMachineError(Exception):
    """Base class for state-machine misuse (a programming error, not a denial)."""


class InvalidTransitionError(StateMachineError):
    """Raised when an event is fired that is not legal from the current state."""

    def __init__(self, state: AuthState, event: AuthEvent) -> None:
        self.state = state
        self.event = event
        super().__init__(
            f"Cannot fire event {event.value!r} from state {state.value!r}."
        )


# Hook signature: (source_state, event, target_state, context) -> None
TransitionHook = Callable[[AuthState, AuthEvent, AuthState, Mapping[str, Any]], None]


class AuthStateMachine:
    """A deterministic finite state machine for the authentication workflow.

    Usage::

        sm = AuthStateMachine()
        sm.fire(AuthEvent.START)               # -> FACE_VERIFICATION
        sm.fire(AuthEvent.VERIFICATION_PASSED) # -> VOICE_VERIFICATION
        sm.fire(AuthEvent.VERIFICATION_PASSED) # -> RECOMMENDATION
        sm.fire(AuthEvent.RECOMMENDATION_READY)# -> COMPLETED (terminal)

    The machine holds only control-flow state; the caller owns the domain data
    (face/voice results, recommendation) and passes it through as ``context`` for
    observability if desired.
    """

    #: The single source of truth for the workflow. ``(state, event) -> state``.
    #: To extend the workflow, add states to :class:`AuthState` and rows here.
    _TRANSITIONS: dict[tuple[AuthState, AuthEvent], AuthState] = {
        (AuthState.IDLE, AuthEvent.START): AuthState.FACE_VERIFICATION,
        (AuthState.FACE_VERIFICATION, AuthEvent.VERIFICATION_PASSED): AuthState.VOICE_VERIFICATION,
        (AuthState.FACE_VERIFICATION, AuthEvent.VERIFICATION_FAILED): AuthState.ACCESS_DENIED,
        (AuthState.VOICE_VERIFICATION, AuthEvent.VERIFICATION_PASSED): AuthState.RECOMMENDATION,
        (AuthState.VOICE_VERIFICATION, AuthEvent.VERIFICATION_FAILED): AuthState.ACCESS_DENIED,
        (AuthState.RECOMMENDATION, AuthEvent.RECOMMENDATION_READY): AuthState.COMPLETED,
    }

    #: States from which no further transition is possible.
    TERMINAL_STATES: frozenset[AuthState] = frozenset(
        {AuthState.ACCESS_DENIED, AuthState.COMPLETED}
    )

    def __init__(
        self,
        *,
        initial: AuthState = AuthState.IDLE,
        on_transition: Optional[TransitionHook] = None,
    ) -> None:
        self._state = initial
        self._on_transition = on_transition
        self.history: list[AuthState] = [initial]

    # --- introspection --------------------------------------------------- #

    @property
    def state(self) -> AuthState:
        """The current state."""
        return self._state

    @property
    def is_terminal(self) -> bool:
        """Whether the machine has reached a state it cannot leave."""
        return self._state in self.TERMINAL_STATES

    def can_fire(self, event: AuthEvent) -> bool:
        """Whether ``event`` is legal from the current state."""
        return (self._state, event) in self._TRANSITIONS

    def allowed_events(self) -> list[AuthEvent]:
        """Events that are legal from the current state (empty if terminal)."""
        return [event for (state, event) in self._TRANSITIONS if state == self._state]

    # --- transitions ----------------------------------------------------- #

    def fire(
        self, event: AuthEvent, context: Mapping[str, Any] | None = None
    ) -> AuthState:
        """Apply ``event`` and move to the next state.

        Records the move in :attr:`history` and invokes ``on_transition`` (if
        set). Raises :class:`InvalidTransitionError` if the event is not legal
        from the current state, so illegal flows fail loudly rather than being
        silently ignored.
        """
        key = (self._state, event)
        try:
            target = self._TRANSITIONS[key]
        except KeyError:
            raise InvalidTransitionError(self._state, event) from None

        source = self._state
        self._state = target
        self.history.append(target)
        if self._on_transition is not None:
            self._on_transition(source, event, target, dict(context or {}))
        return target

    def reset(self, initial: AuthState = AuthState.IDLE) -> None:
        """Return the machine to a starting state (for reuse across sessions)."""
        self._state = initial
        self.history = [initial]

    # --- documentation / visualization ---------------------------------- #

    @classmethod
    def transitions(cls) -> list[tuple[AuthState, AuthEvent, AuthState]]:
        """All ``(source, event, target)`` transitions (for docs/diagrams)."""
        return [(state, event, target) for (state, event), target in cls._TRANSITIONS.items()]

    def __repr__(self) -> str:  # pragma: no cover - debug convenience
        return f"AuthStateMachine(state={self._state.value})"


__all__ = [
    "AuthEvent",
    "AuthState",
    "AuthStateMachine",
    "InvalidTransitionError",
    "StateMachineError",
    "TransitionHook",
]
