"""Command line interface for running the simulation or the optional dashboard."""
from __future__ import annotations

import argparse
import threading

import config
from integrated_system import IntegratedSystem
import utils


def run_simulation(
    instruction: str,
    size: int,
    goal: str,
    stop_event: threading.Event | None = None,
    pause_event: threading.Event | None = None,
) -> None:
    """Run a full conversation cycle without any web interface."""
    config.WIZARD_DEFAULT_GOAL = goal
    system = IntegratedSystem()
    system.run(instruction, size, stop_event=stop_event, pause_event=pause_event)


def run_dashboard(dev: bool) -> None:
    """Start the Flask dashboard."""
    from api import app, socketio  # imported lazily to avoid heavy deps if unused

    utils.ensure_logs_dir()
    msg = "Starting AI Agent Monitor backend..." if dev else "Starting AI Agent Monitor..."
    print(msg)
    socketio.run(app, debug=dev, host="0.0.0.0", port=5000)


def main() -> None:
    """Entry point parsed from command line arguments."""
    parser = argparse.ArgumentParser(description="Run the AIgentChat simulation")
    parser.add_argument("--web", action="store_true", help="launch the web dashboard")
    parser.add_argument("--dev", action="store_true", help="run the dashboard in development mode")
    parser.add_argument("--size", type=int, default=config.POPULATION_SIZE, help="number of agents to generate")
    parser.add_argument("--goal", default=config.WIZARD_DEFAULT_GOAL, help="wizard goal for the conversations")
    parser.add_argument(
        "--instruction",
        default="Generate population",
        help="instruction text used when creating the population",
    )
    args = parser.parse_args()

    if args.web or args.dev:
        run_dashboard(dev=args.dev)
    else:
        run_simulation(args.instruction, args.size, args.goal)


if __name__ == "__main__":
    main()