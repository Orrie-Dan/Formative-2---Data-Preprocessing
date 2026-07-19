"""Console runner for the unauthorized-access simulations.

Demonstrates that the system safely rejects bad authentication attempts. No
machine learning is involved — it drives ``integration.simulations`` (mock
verifiers + crafted sample files) and prints, for each scenario:

    * the authentication attempt,
    * the reason for rejection,
    * the access-denied message,
    * a graceful application exit.

Run from the ``multimodal-authentication/`` directory:

    python simulate_unauthorized.py
    python simulate_unauthorized.py --customer a042
"""

from __future__ import annotations

import argparse
import logging

from integration.config import IntegrationConfig
from integration.logging_config import configure_logging
from integration.simulations import SimulationResult, run_all

logger = logging.getLogger("integration.simulate")

WIDTH = 64


def rule(char: str = "-") -> str:
    return char * WIDTH


def banner(title: str) -> None:
    print("")
    print(rule("="))
    print(title.center(WIDTH))
    print(rule("="))


def render(index: int, total: int, result: SimulationResult) -> None:
    """Print one simulation as attempt -> reason -> denied -> graceful exit."""
    print("")
    print(rule())
    print(f"SCENARIO {index}/{total}  |  {result.scenario.upper()}")
    print(rule())
    print(f"  Authentication attempt : {result.attempt}")
    print(f"  Flow                   : {' -> '.join(result.trace)}")
    print(f"  Reason for rejection   : {result.reason}")
    print(f"  Access decision        : {'GRANTED' if result.granted else 'DENIED'}")
    print(f"  Access denied message  : {result.denied_message}")
    exit_state = "graceful" if result.exited_gracefully else "ABRUPT"
    print(f"  Application exit        : {exit_state} (exit code {result.exit_code})")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate unauthorized-access attempts (mock, no ML).",
    )
    parser.add_argument(
        "--customer",
        default=None,
        help="Customer ID to use for the attempts (defaults to config default).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = IntegrationConfig.from_env()
    configure_logging(config.logging)
    customer_id = args.customer or config.app.default_customer_id

    logger.info("Unauthorized-access simulations started")
    banner("UNAUTHORIZED ACCESS SIMULATIONS")
    print("  Mock verification only - demonstrating safe rejection + exit.")

    results = run_all(customer_id)
    for i, result in enumerate(results, start=1):
        render(i, len(results), result)

    # Summary: every scenario is expected to deny access and exit gracefully.
    denied = sum(1 for r in results if not r.granted)
    graceful = sum(1 for r in results if r.exited_gracefully)
    all_ok = denied == len(results) and graceful == len(results)

    banner("SUMMARY")
    print(f"  Scenarios run       : {len(results)}")
    print(f"  Access denied       : {denied}/{len(results)}")
    print(f"  Graceful exits      : {graceful}/{len(results)}")
    print(f"  Result              : {'ALL REJECTED SAFELY' if all_ok else 'UNEXPECTED OUTCOME'}")
    print("")

    logger.info("Unauthorized-access simulations finished (all_ok=%s)", all_ok)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
