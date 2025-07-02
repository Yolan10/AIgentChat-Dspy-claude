#!/usr/bin/env python3
"""Startup script to ensure proper initialization before running the API."""

import os
import sys

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Import after changing directory
import config
from api import app, socketio, init_user_db, utils

def startup_checks():
    """Perform all necessary startup checks and initialization."""
    print("=== STARTUP CHECKS ===")
    
    # 1. Ensure logs directory exists
    print("1. Checking logs directory...")
    utils.ensure_logs_dir()
    print(f"   ✓ Logs directory exists at: {config.LOGS_DIRECTORY}")
    
    # 2. Check templates
    print("2. Checking templates...")
    template_files = [
        config.POPULATION_INSTRUCTION_TEMPLATE_PATH,
        config.WIZARD_PROMPT_TEMPLATE_PATH,
        config.JUDGE_PROMPT_TEMPLATE_PATH,
        config.SELF_IMPROVE_PROMPT_TEMPLATE_PATH,
    ]
    
    missing_templates = []
    for template in template_files:
        if os.path.exists(template):
            print(f"   ✓ {template}")
        else:
            print(f"   ✗ {template} - MISSING!")
            missing_templates.append(template)
    
    if missing_templates:
        print("\n3. Creating missing templates...")
        # Import and run template creator
        try:
            from create_templates import create_templates
            create_templates()
            print("   ✓ Templates created successfully")
        except Exception as e:
            print(f"   ✗ Failed to create templates: {e}")
    
    # 3. Initialize database
    print("\n3. Initializing database...")
    try:
        init_user_db()
        print("   ✓ Database initialized successfully")
    except Exception as e:
        print(f"   ✗ Database initialization failed: {e}")
        sys.exit(1)
    
    # 4. Check environment
    print("\n4. Checking environment...")
    if os.environ.get("OPENAI_API_KEY"):
        print("   ✓ OPENAI_API_KEY is set")
    else:
        print("   ⚠ OPENAI_API_KEY not found - some features may not work")
    
    print("\n=== STARTUP COMPLETE ===\n")


if __name__ == "__main__":
    # Run startup checks
    startup_checks()
    
    # Get port from environment
    port = int(os.environ.get("PORT", 5000))
    
    # Use production-ready server
    print(f"Starting server on port {port}...")
    
    # For Render, we use gunicorn, but this can be used for local testing
    if "gunicorn" in sys.argv[0]:
        # We're being run by gunicorn, just do the checks
        pass
    else:
        # Direct execution - use socketio.run
        socketio.run(app, debug=False, host="0.0.0.0", port=port)
