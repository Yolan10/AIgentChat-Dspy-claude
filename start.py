#!/usr/bin/env python3
"""Enhanced startup script to ensure proper initialization before running the API."""

import os
import sys

print("="*60)
print("AIGENTCHAT-DSPY STARTUP")
print("="*60)

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"Working directory: {script_dir}")

# Add current directory to Python path
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import after changing directory
try:
    import config
    print("✓ Config module loaded")
except Exception as e:
    print(f"✗ Failed to load config: {e}")
    sys.exit(1)

def startup_checks():
    """Perform all necessary startup checks and initialization."""
    print("\n[1/6] Checking logs directory...")
    
    # Import utils after config is loaded
    try:
        import utils
        utils.ensure_logs_dir()
        print(f"   ✓ Logs directory exists at: {config.LOGS_DIRECTORY}")
    except Exception as e:
        print(f"   ✗ Failed to create logs directory: {e}")
        sys.exit(1)
    
    # CRITICAL: Initialize database BEFORE any other checks
    print("\n[1.5/6] Initializing database FIRST...")
    try:
        from api import init_user_db
        init_user_db()
        print("   ✓ Database initialized successfully")
    except Exception as e:
        print(f"   ✗ Database initialization failed: {e}")
        # Try to create directory and retry
        try:
            db_dir = os.path.dirname(config.USER_DB_PATH)
            os.makedirs(db_dir, exist_ok=True)
            init_user_db()
            print("   ✓ Database initialized on retry")
        except Exception as e2:
            print(f"   ✗ Critical: Database cannot be initialized: {e2}")
            sys.exit(1)
            
    print("\n[2/6] Checking templates...")
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
        print("\n[3/6] Creating missing templates...")
        try:
            from create_templates import create_templates
            create_templates()
            print("   ✓ Templates created successfully")
        except Exception as e:
            print(f"   ✗ Failed to create templates: {e}")
            # Don't exit, templates might be optional for some operations
    else:
        print("\n[3/6] All templates present")
    
    print("\n[4/6] Testing database initialization...")
    try:
        from api import init_user_db
        init_user_db()
        print("   ✓ Database initialized successfully")
    except Exception as e:
        print(f"   ✗ Database initialization failed: {e}")
        print("   Attempting to create database directory...")
        try:
            db_dir = os.path.dirname(config.USER_DB_PATH)
            os.makedirs(db_dir, exist_ok=True)
            print(f"   ✓ Created directory: {db_dir}")
            # Try again
            init_user_db()
            print("   ✓ Database initialized on retry")
        except Exception as e2:
            print(f"   ✗ Database initialization failed on retry: {e2}")
            sys.exit(1)
    
    print("\n[5/6] Checking environment...")
    if os.environ.get("OPENAI_API_KEY"):
        print("   ✓ OPENAI_API_KEY is set")
    else:
        print("   ⚠ OPENAI_API_KEY not found - some features may not work")
    
    # Check PORT for Render
    port = os.environ.get("PORT", "5000")
    print(f"   ✓ Server will run on port {port}")
    
    # Check frontend build
    print("\n[6/6] Checking frontend build...")
    frontend_dist = "frontend/dist"
    if os.path.exists(frontend_dist) and os.path.exists(os.path.join(frontend_dist, "index.html")):
        print("   ✓ Frontend build found")
    else:
        print("   ⚠ Frontend build not found - building now...")
        try:
            import subprocess
            subprocess.run(["npm", "run", "build"], cwd="frontend", check=True)
            print("   ✓ Frontend built successfully")
        except Exception as e:
            print(f"   ⚠ Frontend build failed: {e}")
    
    print("\n" + "="*60)
    print("STARTUP CHECKS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    try:
        startup_checks()
        print("✓ All startup checks passed")
        print("Ready to start server...\n")
    except KeyboardInterrupt:
        print("\n✗ Startup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Startup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
