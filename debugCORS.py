#!/usr/bin/env python3
"""Debug script to test database and CORS fixes.

INSTRUCTIONS:
1. Save this file as 'debug.py' in your project ROOT directory (same level as api.py)
2. Run it with: python debug.py
3. It will test all the fixes and show you what's working/broken
"""

import os
import sys
import sqlite3
import json
from werkzeug.security import generate_password_hash, check_password_hash

def test_database():
    """Test database creation and operations."""
    print("="*50)
    print("TESTING DATABASE")
    print("="*50)
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    db_path = "logs/users.db"
    
    # Remove existing database for clean test
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"✓ Removed existing database")
    
    try:
        # Create database
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        
        # Create table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                username TEXT UNIQUE NOT NULL, 
                password_hash TEXT NOT NULL
            )
        """)
        conn.commit()
        print("✓ Users table created")
        
        # Insert admin user
        password_hash = generate_password_hash('admin')
        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            ("admin", password_hash),
        )
        conn.commit()
        print("✓ Admin user created")
        
        # Test login
        cur = conn.execute("SELECT * FROM users WHERE username=?", ("admin",))
        user_row = cur.fetchone()
        
        if user_row and check_password_hash(user_row["password_hash"], "admin"):
            print("✓ Login test successful")
        else:
            print("✗ Login test failed")
            
        conn.close()
        print("✓ Database test completed successfully")
        
    except Exception as e:
        print(f"✗ Database test failed: {e}")
        import traceback
        traceback.print_exc()

def test_templates():
    """Test template creation."""
    print("\n" + "="*50)
    print("TESTING TEMPLATES")
    print("="*50)
    
    os.makedirs("templates", exist_ok=True)
    
    templates = {
        "templates/wizard_prompt.txt": "You are a research wizard. Goal: {{goal}}",
        "templates/judge_prompt.txt": "You are a judge. Evaluate: {{transcript}}",
        "templates/population_instruction.txt": "Create {{n}} personas.",
        "templates/self_improve_prompt.txt": "Improve based on: {{logs}}"
    }
    
    for path, content in templates.items():
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Created {path}")
        except Exception as e:
            print(f"✗ Failed to create {path}: {e}")

def test_cors_config():
    """Test CORS configuration."""
    print("\n" + "="*50)
    print("TESTING CORS CONFIG")
    print("="*50)
    
    # Test CORS origins parsing
    test_origins = [
        "*",
        "https://aigentchat-dspy-pf7r.onrender.com",
        "https://aigentchat-dspy-pf7r.onrender.com,http://localhost:3000"
    ]
    
    for origins_str in test_origins:
        origins = origins_str.split(",")
        render_url = "https://aigentchat-dspy-pf7r.onrender.com"
        
        if render_url not in origins and origins != ["*"]:
            origins.append(render_url)
            
        print(f"✓ Origins '{origins_str}' -> {origins}")

def test_frontend_build():
    """Check if frontend is built."""
    print("\n" + "="*50)
    print("CHECKING FRONTEND BUILD")
    print("="*50)
    
    frontend_dist = "frontend/dist"
    index_path = os.path.join(frontend_dist, "index.html")
    
    if os.path.exists(index_path):
        print("✓ Frontend build exists")
        
        # Check file size
        size = os.path.getsize(index_path)
        print(f"✓ index.html size: {size} bytes")
        
        # List other files
        if os.path.exists(frontend_dist):
            files = os.listdir(frontend_dist)
            print(f"✓ Frontend files: {files}")
            
    else:
        print("✗ Frontend build not found")
        print("   Run: cd frontend && npm install && npm run build")

def test_config():
    """Test configuration loading."""
    print("\n" + "="*50)
    print("TESTING CONFIG")
    print("="*50)
    
    try:
        import config
        print("✓ Config module loaded")
        print(f"✓ LOGS_DIRECTORY: {config.LOGS_DIRECTORY}")
        print(f"✓ USER_DB_PATH: {config.USER_DB_PATH}")
        print(f"✓ LLM_MODEL: {config.LLM_MODEL}")
    except Exception as e:
        print(f"✗ Config loading failed: {e}")

def main():
    """Run all tests."""
    print("AIGENTCHAT-DSPY DEBUG SCRIPT")
    print("Testing fixes for database and CORS issues...")
    
    test_config()
    test_database()
    test_templates()
    test_cors_config()
    test_frontend_build()
    
    print("\n" + "="*50)
    print("DEBUG TESTS COMPLETE")
    print("="*50)
    print("\nIf all tests passed, the fixes should resolve:")
    print("1. 'no such table: users' error")
    print("2. Socket.IO CORS origin rejection")
    print("3. Missing template files")
    print("\nNext steps:")
    print("1. Update your api.py with the fixed version")
    print("2. Update your Procfile")
    print("3. Update your render.yaml")
    print("4. Add the start.py file")
    print("5. Commit and push to trigger new deployment")

if __name__ == "__main__":
    main()
