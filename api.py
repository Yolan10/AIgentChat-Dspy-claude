"""Web API backend for the AI agent simulation system."""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from typing import Dict, List, Any
import csv
import re
import sys
import sqlite3

from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    current_user,
    login_required,
)

from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO
from flask_cors import CORS
from tracking_chat_openai import TrackingChatOpenAI as ChatOpenAI
from langchain_core.messages import SystemMessage

import config
import utils
from integrated_system import IntegratedSystem
from token_tracker import token_tracker

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", os.urandom(24).hex())

# Configure CORS properly for production
cors_origins = os.environ.get("CORS_ORIGINS", "*").split(",")
# Add specific Render URL if not already included
render_url = "https://aigentchat-dspy-pf7r.onrender.com"
if render_url not in cors_origins and cors_origins != ["*"]:
    cors_origins.append(render_url)

# Clean up origins list
cors_origins = [origin.strip() for origin in cors_origins if origin.strip()]

CORS(app, 
     origins=cors_origins,
     supports_credentials=True,
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])

# Fix SocketIO CORS configuration - ensure it accepts the render URL
if cors_origins == ["*"]:
    socketio_origins = "*"
else:
    socketio_origins = cors_origins

socketio = SocketIO(app, 
                   cors_allowed_origins=socketio_origins, 
                   async_mode="threading",
                   logger=False,  # Reduce logging noise
                   engineio_logger=False)

login_manager = LoginManager()
login_manager.init_app(app)


def get_db_connection():
    """Get database connection with proper settings."""
    # Ensure logs directory exists first
    utils.ensure_logs_dir()
    
    conn = sqlite3.connect(config.USER_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_user_db():
    """Initialize user database with default admin account."""
    print(f"[DB] Initializing database at: {config.USER_DB_PATH}")
    
    # Ensure the directory exists first
    db_dir = os.path.dirname(config.USER_DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
        print(f"[DB] Created database directory: {db_dir}")
    
    try:
        # Always try to create/connect to database
        with get_db_connection() as conn:
            # Create the users table if it doesn't exist
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    username TEXT UNIQUE NOT NULL, 
                    password_hash TEXT NOT NULL
                )
            """)
            conn.commit()
            print("[DB] Users table created/verified")
            
            # Check if any users exist
            cur = conn.execute("SELECT COUNT(*) FROM users")
            count = cur.fetchone()[0]
            
            if count == 0:
                # Create default admin user
                password_hash = generate_password_hash('admin')
                conn.execute(
                    "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                    ("admin", password_hash),
                )
                conn.commit()
                print("[DB] Created default admin user - username: 'admin', password: 'admin'")
            else:
                print(f"[DB] Database initialized with {count} existing users")
                
    except Exception as e:
        print(f"[DB] Error initializing database: {e}")
        import traceback
        traceback.print_exc()
        raise


class User(UserMixin):
    def __init__(self, user_id: int, username: str, password_hash: str):
        self.id = user_id
        self.username = username
        self.password_hash = password_hash


def get_user_by_id(user_id: int):
    with get_db_connection() as conn:
        cur = conn.execute("SELECT * FROM users WHERE id=?", (user_id,))
        row = cur.fetchone()
    return row


def get_user_by_username(username: str):
    with get_db_connection() as conn:
        cur = conn.execute("SELECT * FROM users WHERE username=?", (username,))
        row = cur.fetchone()
    return row


def create_user(username: str, password: str):
    password_hash = generate_password_hash(password)
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, password_hash),
        )
        conn.commit()


@login_manager.user_loader
def load_user(user_id):
    row = get_user_by_id(user_id)
    if row:
        return User(row["id"], row["username"], row["password_hash"])
    return None

# Initialize the user database when the module is imported. This ensures
# the required table and default admin account exist even if the script is
# run via Gunicorn without executing the ``__main__`` section.
try:
    init_user_db()
except Exception as init_err:
    print(f"[DB] Failed to initialize user database: {init_err}")


# Global state
simulation_state = {
    "running": False,
    "paused": False,
    "current_run": None,
    "progress": {"completed": 0, "total": 0},
    "latest_conversation": None,
    "clients": 0,
}

# Lock protecting access to ``simulation_state``
simulation_state_lock = threading.Lock()

stop_event = threading.Event()
pause_event = threading.Event()
simulation_thread = None
connected_clients = set()


def get_simulation_state() -> Dict[str, Any]:
    """Return a copy of the current simulation state."""
    with simulation_state_lock:
        return dict(simulation_state)


class WebSocketLogger:
    """Logger that emits events to connected web clients."""

    def __init__(self):
        self.conversation_buffer = []

    def log_conversation_turn(self, speaker: str, text: str, agent_id: str = None):
        """Log a conversation turn and emit to clients."""
        turn_data = {
            "speaker": speaker,
            "text": text,
            "timestamp": utils.get_timestamp(),
            "agent_id": agent_id,
        }
        self.conversation_buffer.append(turn_data)
        socketio.start_background_task(socketio.emit, "conversation_turn", turn_data)

    def log_progress(self, completed: int, total: int, run_no: int):
        """Log simulation progress."""
        progress_data = {
            "completed": completed,
            "total": total,
            "run_no": run_no,
            "percentage": (completed / total * 100) if total > 0 else 0,
        }
        with simulation_state_lock:
            simulation_state["progress"] = progress_data
        socketio.start_background_task(socketio.emit, "progress_update", progress_data)

    def log_system_event(self, event_type: str, data: Dict[str, Any]):
        """Log system events."""
        event_data = {
            "type": event_type,
            "timestamp": utils.get_timestamp(),
            "data": data,
        }
        socketio.start_background_task(socketio.emit, "system_event", event_data)


ws_logger = WebSocketLogger()


def run_simulation_with_logging(instruction: str, population_size: int, goal: str):
    """Run simulation with WebSocket logging."""

    try:
        with simulation_state_lock:
            simulation_state["running"] = True
            simulation_state["paused"] = False

        # Update config with new goal
        config.WIZARD_DEFAULT_GOAL = goal

        system = IntegratedSystem()
        run_no = utils.increment_run_number()
        with simulation_state_lock:
            simulation_state["current_run"] = run_no
        token_tracker.set_run(run_no)

        ws_logger.log_system_event(
            "simulation_start",
            {
                "run_no": run_no,
                "instruction": instruction,
                "population_size": population_size,
                "goal": goal,
            },
        )

        # Generate population using the GodAgent directly
        population = system.god.spawn_population(
            instruction,
            population_size,
            run_no=run_no,
            start_index=1,
        )

        for agent in population:
            ws_logger.log_system_event(
                "agent_created",
                {
                    "agent_id": agent.agent_id,
                    "name": agent.name,
                    "personality": agent.personality_description,
                },
            )

        # Run conversations
        summary = []
        for idx, pop in enumerate(population):
            if stop_event.is_set():
                break

            # Handle pause
            while pause_event.is_set() and not stop_event.is_set():
                time.sleep(0.5)

            if stop_event.is_set():
                break

            ws_logger.log_system_event(
                "conversation_start", {"agent_id": pop.agent_id, "agent_name": pop.name}
            )

            # Monkey patch the wizard to log conversations
            original_converse = system.wizard.converse_with

            def logged_converse(agent, show_live=False):
                log = original_converse(agent, show_live=False)

                # Emit conversation turns
                for turn in log.get("turns", []):
                    ws_logger.log_conversation_turn(
                        turn["speaker"], turn["text"], agent.agent_id
                    )

                return log

            system.wizard.converse_with = logged_converse

            # Run the conversation once
            log = system.wizard.converse_with(pop)
            
            # Now run the independent judge
            judge_result = system.primary_judge.assess(log)
            log["judge_result"] = judge_result
            
            # Give feedback to wizard for learning
            system.wizard.add_judge_feedback(pop.agent_id, judge_result)
            
            # Save log with judge results
            filename = f"{system.wizard.wizard_id}_{pop.agent_id}_{utils.get_timestamp().replace(':', '').replace('-', '')}.json"
            utils.save_conversation_log(log, filename)
            
            # Update summary
            spec = pop.get_spec()
            entry = {
                "pop_agent_id": pop.agent_id,
                "name": spec.get("name"),
                "personality_description": spec.get("personality_description"),
                "system_instruction": spec.get("system_instruction"),
                "temperature": spec.get("llm_settings", {}).get("temperature"),
                "max_tokens": spec.get("llm_settings", {}).get("max_tokens"),
                "success": log["judge_result"].get("success"),
                "score": log["judge_result"].get("score"),
                "goal_completion": log["judge_result"].get("goal_completion", 0),
                "coherence": log["judge_result"].get("coherence", 0),
                "tone": log["judge_result"].get("tone", 0),
            }
            
            summary.append(entry)

            # Update progress
            ws_logger.log_progress(idx + 1, len(population), run_no)

            ws_logger.log_system_event(
                "conversation_end",
                {
                    "agent_id": pop.agent_id,
                    "success": entry["success"],
                    "score": entry["score"],
                },
            )

        # Save final summary
        utils.save_conversation_log(summary, f"summary_{run_no}.json")

        ws_logger.log_system_event(
            "simulation_complete",
            {"run_no": run_no, "total_conversations": len(summary)},
        )

    except Exception as e:
        print(f"[ERROR] Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        ws_logger.log_system_event("simulation_error", {"error": str(e)})
    finally:
        with simulation_state_lock:
            simulation_state["running"] = False
            simulation_state["paused"] = False


# API Routes


@app.route("/api/status")
def get_status():
    """Get current simulation status."""
    return jsonify(get_simulation_state())


@app.route("/api/login", methods=["POST"])
def login():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        username = data.get("username", "")
        password = data.get("password", "")
        
        if not username or not password:
            return jsonify({"error": "Username and password required"}), 400
            
        user_row = get_user_by_username(username)
        if user_row and check_password_hash(user_row["password_hash"], password):
            user = User(user_row["id"], user_row["username"], user_row["password_hash"])
            login_user(user)
            return jsonify({"status": "success"})
        return jsonify({"error": "Invalid credentials"}), 401
    except Exception as e:
        print(f"Login error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Server error during login"}), 500


@app.route("/api/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    return jsonify({"status": "success"})


@app.route("/api/check_auth")
def check_auth():
    return jsonify({"authenticated": current_user.is_authenticated})


@app.route("/api/config", methods=["GET"])
def get_config():
    """Get current configuration."""
    config_data = {
        "POPULATION_SIZE": config.POPULATION_SIZE,
        "WIZARD_DEFAULT_GOAL": config.WIZARD_DEFAULT_GOAL,
        "MAX_TURNS": config.MAX_TURNS,
        "LLM_MODEL": config.LLM_MODEL,
        "LLM_TEMPERATURE": config.LLM_TEMPERATURE,
        "LLM_MAX_TOKENS": config.LLM_MAX_TOKENS,
        "LLM_TOP_P": config.LLM_TOP_P,
        "SELF_IMPROVE_AFTER": config.SELF_IMPROVE_AFTER,
        "SHOW_LIVE_CONVERSATIONS": config.SHOW_LIVE_CONVERSATIONS,
        "DSPY_TRAINING_ITER": config.DSPY_TRAINING_ITER,
        "DSPY_LEARNING_RATE": config.DSPY_LEARNING_RATE,
        "HISTORY_BUFFER_LIMIT": config.HISTORY_BUFFER_LIMIT,
        "POP_HISTORY_LIMIT": config.POP_HISTORY_LIMIT,
    }
    return jsonify(config_data)


@app.route("/api/config", methods=["POST"])
@login_required
def update_config():
    """Update configuration."""
    data = request.json

    # Update config values
    for key, value in data.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return jsonify({"status": "success", "message": "Configuration updated"})


@app.route("/api/templates", methods=["GET"])
def get_templates():
    """Get all prompt templates."""
    templates = {}
    template_files = [
        "wizard_prompt.txt",
        "judge_prompt.txt",
        "population_instruction.txt",
        "self_improve_prompt.txt",
    ]

    for filename in template_files:
        path = os.path.join("templates", filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                templates[filename] = f.read()

    return jsonify(templates)


@app.route("/api/templates/<template_name>", methods=["POST"])
@login_required
def update_template(template_name):
    """Update a specific template."""
    data = request.json
    content = data.get("content", "")

    path = os.path.join("templates", template_name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    return jsonify(
        {"status": "success", "message": f"Template {template_name} updated"}
    )


@app.route("/api/run", methods=["POST"])
@login_required
def start_simulation():
    """Start a new simulation."""
    global simulation_thread, stop_event, pause_event

    with simulation_state_lock:
        if simulation_state["running"]:
            return jsonify({"error": "Simulation already running"}), 400

    data = request.json
    instruction = data.get("instruction", "Generate population")
    population_size = data.get("population_size", config.POPULATION_SIZE)
    goal = data.get("goal", config.WIZARD_DEFAULT_GOAL)

    # Reset events
    stop_event.clear()
    pause_event.clear()

    # Start simulation in background thread
    simulation_thread = threading.Thread(
        target=run_simulation_with_logging, args=(instruction, population_size, goal)
    )
    simulation_thread.start()

    return jsonify({"status": "success", "message": "Simulation started"})


@app.route("/api/pause", methods=["POST"])
def pause_simulation():
    """Pause/resume the simulation."""
    with simulation_state_lock:
        running = simulation_state["running"]
    if not running:
        return jsonify({"error": "No simulation running"}), 400

    if pause_event.is_set():
        pause_event.clear()
        with simulation_state_lock:
            simulation_state["paused"] = False
        message = "Simulation resumed"
    else:
        pause_event.set()
        with simulation_state_lock:
            simulation_state["paused"] = True
        message = "Simulation paused"

    return jsonify({"status": "success", "message": message})


@app.route("/api/stop", methods=["POST"])
def stop_simulation():
    """Stop the simulation."""
    global simulation_thread

    with simulation_state_lock:
        running = simulation_state["running"]
    if not running:
        return jsonify({"error": "No simulation running"}), 400

    stop_event.set()
    if simulation_thread:
        simulation_thread.join(timeout=5)

    with simulation_state_lock:
        simulation_state["running"] = False
        simulation_state["paused"] = False

    return jsonify({"status": "success", "message": "Simulation stopped"})


@app.route("/api/logs/summary/<int:run_no>")
def get_summary(run_no):
    """Get summary for a specific run."""
    path = os.path.join(config.LOGS_DIRECTORY, f"summary_{run_no}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Summary not found"}), 404


@app.route("/api/logs/scores")
def get_scores():
    """Get wizard scores data."""
    path = os.path.join(config.LOGS_DIRECTORY, "wizard_scores.csv")
    if not os.path.exists(path):
        return jsonify([])

    scores = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(
                {
                    "timestamp": row["timestamp"],
                    "run": int(row["run"]),
                    "conversation": int(row["conversation"]),
                    "score": float(row["score"]),
                    "improved": bool(int(row["improved"])),
                }
            )

    return jsonify(scores)


@app.route("/api/logs/runs")
def get_runs():
    """Get list of available runs."""
    if not os.path.exists(config.LOGS_DIRECTORY):
        return jsonify([])

    runs = set()
    for filename in os.listdir(config.LOGS_DIRECTORY):
        if filename.startswith("summary_") and filename.endswith(".json"):
            run_match = re.search(r"summary_(\d+)\.json", filename)
            if run_match:
                runs.add(int(run_match.group(1)))

    return jsonify(sorted(list(runs), reverse=True))


@app.route("/api/logs/token_usage")
def get_token_usage():
    """Return recorded token usage."""
    path = os.path.join(config.LOGS_DIRECTORY, "token_usage.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({})


@app.route("/api/logs/summarize/<filename>")
def summarize_conversation_log(filename):
    """Return an LLM-generated summary of the given conversation log."""
    path = os.path.join(config.LOGS_DIRECTORY, filename)
    if not os.path.exists(path):
        # try prefix match if exact file not found
        matches = [
            f for f in os.listdir(config.LOGS_DIRECTORY) if f.startswith(filename)
        ]
        if not matches:
            return jsonify({"error": "Log not found"}), 404
        path = os.path.join(config.LOGS_DIRECTORY, sorted(matches)[0])

    with open(path, "r", encoding="utf-8") as f:
        log = json.load(f)

    transcript = "\n".join(
        [f"{t['speaker']}: {t['text']}" for t in log.get("turns", [])]
    )
    prompt = (
        "Summarize the following conversation between a wizard and an agent in a few sentences:\n\n"
        f"{transcript}\n\nSummary:"
    )
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=0.3,
        max_tokens=200,
        max_retries=config.OPENAI_MAX_RETRIES,
    )
    summary = llm.invoke([SystemMessage(content=prompt)]).content.strip()
    return jsonify({"summary": summary})


def _iter_conversation_logs():
    """Yield parsed conversation log objects."""
    if not os.path.exists(config.LOGS_DIRECTORY):
        return
    for filename in os.listdir(config.LOGS_DIRECTORY):
        if not filename.endswith(".json"):
            continue
        if (
            filename.startswith("summary_")
            or filename.startswith("improve_")
            or "spec_" in filename
        ):
            continue
        path = os.path.join(config.LOGS_DIRECTORY, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                yield json.load(f)
        except Exception:
            continue


def _extract_metrics(log: dict) -> dict:
    """Return key metrics from a conversation log."""
    turns = log.get("turns", [])
    judge = log.get("judge_result", {})
    agent_id = log.get("pop_agent_id")
    run_no = None
    if agent_id:
        try:
            run_no = int(agent_id.split(".")[0])
        except (ValueError, IndexError):
            run_no = None
    name = None
    spec = log.get("pop_agent_spec")
    if isinstance(spec, dict):
        name = spec.get("name")
    return {
        "run": run_no,
        "agent_id": agent_id,
        "name": name,
        "length": len(turns),
        "score": float(judge.get("overall", judge.get("score", 0))),
        "success": bool(judge.get("success")),
        "goal_completion": float(judge.get("goal_completion", 0)),
        "coherence": float(judge.get("coherence", 0)),
        "tone": float(judge.get("tone", 0)),
    }


@app.route("/api/logs/lengths")
def get_length_distribution():
    """Return conversation length and score for each log."""
    data = [_extract_metrics(log) for log in _iter_conversation_logs()]
    return jsonify(data)


@app.route("/api/logs/agent_stats")
def get_agent_stats():
    """Aggregate success counts per population agent."""
    stats: Dict[str, Dict[str, Any]] = {}
    for log in _iter_conversation_logs():
        m = _extract_metrics(log)
        entry = stats.setdefault(
            m["agent_id"],
            {
                "agent_id": m["agent_id"],
                "name": m["name"],
                "successes": 0,
                "failures": 0,
                "run": m["run"],
            },
        )
        if m["success"]:
            entry["successes"] += 1
        else:
            entry["failures"] += 1
    return jsonify(list(stats.values()))

@app.route("/api/judge/performance")
def get_judge_performance():
    """Get performance metrics for all judges."""
    # Need to maintain a system instance or recreate it
    system = IntegratedSystem()
    reports = system.get_judge_performance_reports()
    return jsonify(reports)


@app.route("/api/judge/<judge_id>/metrics")
def get_judge_metrics(judge_id):
    """Get performance metrics for a specific judge."""
    from judge_improver import JudgeCalibrator
    
    calibrator = JudgeCalibrator(f"logs/judge_calibration_{judge_id}.json")
    metrics = calibrator.calculate_metrics()
    
    return jsonify({
        "judge_id": judge_id,
        "consistency": metrics.consistency_score,
        "discrimination": metrics.discrimination_score,
        "calibration": metrics.calibration_score,
        "detail_quality": metrics.detail_score,
        "overall": metrics.overall_score,
        "suggestions": calibrator.get_improvement_suggestions()
    })


@app.route("/api/judge/<judge_id>/feedback", methods=["POST"])
@login_required
def add_judge_feedback(judge_id):
    """Add human feedback for a judge evaluation."""
    data = request.json
    conversation_id = data.get("conversation_id")
    human_score = data.get("human_score")
    
    if not conversation_id or human_score is None:
        return jsonify({"error": "Missing required fields"}), 400
    
    from judge_improver import JudgeCalibrator
    calibrator = JudgeCalibrator(f"logs/judge_calibration_{judge_id}.json")
    
    # Find and update the evaluation
    for entry in calibrator.calibration_data:
        if entry.get("conversation_id") == conversation_id:
            entry["human_score"] = human_score
            calibrator._save_history()
            return jsonify({"status": "success"})
    
    return jsonify({"error": "Conversation not found"}), 404


@app.route("/api/logs/metrics.csv")
def download_metrics_csv():
    """Return aggregated metrics as a CSV download."""
    rows = [_extract_metrics(log) for log in _iter_conversation_logs()]
    fieldnames = [
        "run",
        "agent_id",
        "name",
        "length",
        "score",
        "success",
        "goal_completion",
        "coherence",
        "tone",
    ]
    import io

    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    csv_data = out.getvalue()
    return app.response_class(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=metrics.csv"},
    )


@app.route("/api/logs/conversation/<filename>")
def get_conversation_log(filename):
    """Get a specific conversation log."""
    path = os.path.join(config.LOGS_DIRECTORY, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Log not found"}), 404


@app.route("/api/search_logs")
def search_logs():
    """Search JSON logs for a keyword or agent ID."""
    query = request.args.get("q", "").strip().lower()
    if not query:
        return jsonify([])

    try:
        limit = int(request.args.get("limit", 50))
    except ValueError:
        limit = 50
    try:
        offset = int(request.args.get("offset", 0))
    except ValueError:
        offset = 0

    results = []
    logs_dir = config.LOGS_DIRECTORY
    if not os.path.exists(logs_dir):
        return jsonify([])

    filenames = sorted(os.listdir(logs_dir), reverse=True)
    for filename in filenames:
        if len(results) >= limit + offset:
            break
        if not filename.endswith(".json"):
            continue
        path = os.path.join(logs_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        def search_obj(obj):
            if len(results) >= limit + offset:
                return True
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, (str, int, float)):
                        if query in str(v).lower():
                            snippet = str(obj)
                            results.append({"file": filename, "snippet": snippet})
                            return len(results) >= limit + offset
                    else:
                        if search_obj(v):
                            return True
            elif isinstance(obj, list):
                for item in obj:
                    if search_obj(item):
                        return True
            return False

        search_obj(data)

    return jsonify(results[offset : offset + limit])


# Serve frontend static files
@app.route("/")
def serve_frontend():
    return send_from_directory("frontend/dist", "index.html")


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory("frontend/dist", path)


# WebSocket events
@socketio.on("connect")
def handle_connect():
    """Handle client connection."""
    connected_clients.add(request.sid)
    with simulation_state_lock:
        simulation_state["clients"] = len(connected_clients)
        state_copy = dict(simulation_state)
    socketio.emit("status_update", state_copy, to=request.sid)


@socketio.on("disconnect")
def handle_disconnect():
    """Handle client disconnection."""
    connected_clients.discard(request.sid)
    with simulation_state_lock:
        simulation_state["clients"] = len(connected_clients)
    ws_logger.log_system_event("client_disconnect", {"sid": request.sid})


if __name__ == "__main__":
    print("[STARTUP] Starting API server...")
    
    # Ensure logs directory exists
    utils.ensure_logs_dir()
    print(f"[STARTUP] Logs directory: {config.LOGS_DIRECTORY}")
    
    # Initialize database
    print("[STARTUP] Initializing database...")
    init_user_db()
    
    if len(sys.argv) == 4 and sys.argv[1] == "create_user":
        create_user(sys.argv[2], sys.argv[3])
        print("User created")
    else:
        # Use regular Flask run in production instead of socketio.run
        # This is more compatible with various deployment platforms
        port = int(os.environ.get("PORT", 5000))
        print(f"[STARTUP] Starting server on port {port}")
        socketio.run(app, debug=False, host="0.0.0.0", port=port)
