"""Authentication domain.

Everything needed to decide *whether a customer is who they claim to be*:

    risk.py          : tabular risk heuristic (social + transaction signals)
    face/            : facial-recognition verifier (Member 1 plugs in here)
    voice/           : voice verifier (Member 2 plugs in here)
    service.py       : fuses the three signals into a single AuthDecision
    state_machine.py : explicit FSM defining the authentication control flow
    workflow.py      : drives the state machine (face -> voice -> recommendation)

Kept deliberately separate from ``integration.recommendation``.
"""

from integration.authentication.service import AuthenticationService
from integration.authentication.state_machine import (
    AuthEvent,
    AuthState,
    AuthStateMachine,
    InvalidTransitionError,
)
from integration.authentication.workflow import (
    AuthenticationWorkflow,
    WorkflowResult,
)

__all__ = [
    "AuthEvent",
    "AuthState",
    "AuthStateMachine",
    "AuthenticationService",
    "AuthenticationWorkflow",
    "InvalidTransitionError",
    "WorkflowResult",
]
