"""Command line interface for running the simulation or the optional dashboard."""
from __future__ import annotations

import argparse
import threading
import os
import sys

import config
from integrated_system import IntegratedSystem
import utils


def validate_environment():
    """Validate that the environment is properly configured."""
    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY") and not config.OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not found in environment variables!")
        print("Please set it using:")
        print("  Linux/Mac: export OPENAI_API_KEY='your-key-here'")
        print("  Windows:   $env:OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    
    # Validate configuration
    try:
        config.validate_configuration()
    except ValueError as e:
        print(f"ERROR: Configuration validation failed: {e}")
        sys.exit(1)
    
    # Ensure logs directory exists
    utils.ensure_logs_dir()
    
    # Check template files exist
    template_files = [
        config.POPULATION_INSTRUCTION_TEMPLATE_PATH,
        config.WIZARD_PROMPT_TEMPLATE_PATH,
        config.JUDGE_PROMPT_TEMPLATE_PATH,
        config.SELF_IMPROVE_PROMPT_TEMPLATE_PATH,
    ]
    
    for template_path in template_files:
        if not os.path.exists(template_path):
            print(f"ERROR: Template file not found: {template_path}")
            sys.exit(1)


def run_simulation(
    instruction: str,
    size: int,
    goal: str,
    stop_event: threading.Event | None = None,
    pause_event: threading.Event | None = None,
) -> None:
    """Run a full conversation cycle without any web interface."""
    # Validate before running
    validate_environment()
    
    # Update configuration with command-line arguments
    config.WIZARD_DEFAULT_GOAL = goal
    config.POPULATION_SIZE = size
    
    print(f"\n{'='*80}")
    print("AI Agent Chat Simulation")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  - LLM Model: {config.LLM_MODEL}")
    print(f"  - Population Size: {size}")
    print(f"  - Max Turns per Conversation: {config.MAX_TURNS}")
    print(f"  - Parallel Conversations: {config.PARALLEL_CONVERSATIONS}")
    print(f"  - Live Conversation Display: {config.SHOW_LIVE_CONVERSATIONS}")
    print(f"  - Self Improvement Schedule: {config.SELF_IMPROVE_AFTER}")
    print(f"  - Judge Improvement Interval: {config.JUDGE_IMPROVEMENT_INTERVAL}")
    print(f"  - Multi-Judge Enabled: {config.ENABLE_MULTI_JUDGE}")
    print(f"{'='*80}\n")
    
    # Create and run the system
    system = IntegratedSystem()
    system.run(instruction, size, stop_event=stop_event, pause_event=pause_event)


def run_dashboard(dev: bool) -> None:
    """Start the Flask dashboard."""
    # Validate before running
    validate_environment()
    
    from api import app, socketio  # imported lazily to avoid heavy deps if unused

    utils.ensure_logs_dir()
    msg = "Starting AI Agent Monitor backend..." if dev else "Starting AI Agent Monitor..."
    print(msg)
    socketio.run(app, debug=dev, host="0.0.0.0", port=5000)


def main() -> None:
    """Entry point parsed from command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the AIgentChat simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run simulation with default settings
  python main.py
  
  # Run simulation with custom population size
  python main.py --size 10
  
  # Run simulation with custom goal
  python main.py --goal "Research user experiences with hearing aids"
  
  # Run simulation with custom instruction for population generation
  python main.py --instruction "Create diverse personas with varying levels of hearing loss"
  
  # Launch web dashboard
  python main.py --web
  
  # Launch web dashboard in development mode
  python main.py --dev
"""
    )
    
    parser.add_argument("--web", action="store_true", help="launch the web dashboard")
    parser.add_argument("--dev", action="store_true", help="run the dashboard in development mode")
    parser.add_argument(
        "--size", 
        type=int, 
        default=config.POPULATION_SIZE, 
        help=f"number of agents to generate (default: {config.POPULATION_SIZE})"
    )
    parser.add_argument(
        "--goal", 
        default=config.WIZARD_DEFAULT_GOAL, 
        help="wizard goal for the conversations"
    )
    parser.add_argument(
        "--instruction",
        default="Generate a diverse population of individuals with hearing loss experiences",
        help="instruction text used when creating the population",
    )
    
    args = parser.parse_args()

    if args.web or args.dev:
        run_dashboard(dev=args.dev)
    else:
        try:
            run_simulation(args.instruction, args.size, args.goal)
        except KeyboardInterrupt:
            print("\n\nSimulation interrupted by user. Exiting...")
            sys.exit(0)
        except Exception as e:
            print(f"\n\nERROR: Simulation failed with error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
