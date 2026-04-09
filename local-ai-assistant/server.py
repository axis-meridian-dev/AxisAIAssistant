#!/usr/bin/env python3
"""
Local AI Assistant — Web Dashboard Server
Serves a browser-based control panel + chat interface.

Run: python server.py
Then open: http://localhost:5000
"""

import json
import asyncio
import logging
import os
import re
import secrets
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from functools import wraps

from flask import Flask, request, jsonify, send_file, Response, session, abort
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import ollama

from agent import Agent
from config import load_config
from cloud_reasoning import MODELS as CLOUD_MODELS
from log import setup_logging

setup_logging()
logger = logging.getLogger("ai_assistant.server")

app = Flask(__name__, template_folder="templates")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))

# ── Rate Limiting ───────────────────────────────────────────────────────────

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["120 per minute"],
    storage_uri="memory://",
)

# ── Authentication ──────────────────────────────────────────────────────────

AUTH_TOKEN = os.environ.get("DASHBOARD_TOKEN", "")
_auth_initialized = False


def require_auth(f):
    """Protect endpoints with token-based auth (if DASHBOARD_TOKEN is set)."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not AUTH_TOKEN:
            return f(*args, **kwargs)
        if session.get("authenticated"):
            return f(*args, **kwargs)
        token = request.headers.get("X-Auth-Token") or request.args.get("token")
        if token == AUTH_TOKEN:
            session["authenticated"] = True
            return f(*args, **kwargs)
        return jsonify({"error": "Unauthorized"}), 401
    return decorated


# ── CSRF Protection ─────────────────────────────────────────────────────────

def generate_csrf_token():
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_hex(32)
    return session["csrf_token"]


@app.after_request
def set_csrf_cookie(response):
    if "csrf_token" in session:
        response.set_cookie("csrf_token", session["csrf_token"], samesite="Strict", httponly=False)
    return response


@app.before_request
def check_csrf():
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return
    # Exempt the login endpoint and SSE streams
    if request.endpoint in ("index", "login", "pull_model"):
        return
    token = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token")
    if "csrf_token" in session and token != session["csrf_token"]:
        abort(403)


# ── Request Logging ─────────────────────────────────────────────────────────

@app.before_request
def log_request():
    logger.info("%s %s from %s", request.method, request.path, request.remote_addr)


@app.after_request
def log_response(response):
    if response.status_code >= 400:
        logger.warning("%s %s → %s", request.method, request.path, response.status_code)
    return response


# ── Input Validation Helpers ────────────────────────────────────────────────

MAX_MESSAGE_LENGTH = 50000  # ~50k chars
MODEL_NAME_RE = re.compile(r'^[a-zA-Z0-9_.:\-/]+$')


def validate_model_name(name: str) -> bool:
    return bool(name) and len(name) < 200 and MODEL_NAME_RE.match(name)


def safe_error(msg: str, code: int = 500):
    """Return a sanitized error without leaking internals."""
    return jsonify({"error": msg}), code


# ── Global state ────────────────────────────────────────────────────────────

config = load_config()
agent = None
agent_lock = threading.Lock()


def get_agent():
    global agent
    if agent is None:
        agent = Agent(config)
    return agent


# ── API Routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    generate_csrf_token()
    return send_file("templates/dashboard.html")


@app.route("/api/login", methods=["POST"])
def login():
    """Authenticate with a token (if DASHBOARD_TOKEN is set)."""
    if not AUTH_TOKEN:
        return jsonify({"success": True, "message": "No auth required"})
    token = (request.json or {}).get("token", "")
    if secrets.compare_digest(token, AUTH_TOKEN):
        session["authenticated"] = True
        generate_csrf_token()
        return jsonify({"success": True})
    return jsonify({"error": "Invalid token"}), 401


@app.route("/api/chat", methods=["POST"])
@require_auth
@limiter.limit("30 per minute")
def chat():
    """Send a message to the agent and get a response."""
    data = request.json or {}
    message = data.get("message", "")

    if not message.strip():
        return safe_error("Empty message", 400)
    if len(message) > MAX_MESSAGE_LENGTH:
        return safe_error(f"Message too long (max {MAX_MESSAGE_LENGTH} chars)", 400)

    import time as _time
    start = _time.time()

    with agent_lock:
        a = get_agent()
        model_override = data.get("cloud_model")
        if model_override:
            if model_override not in CLOUD_MODELS:
                return safe_error("Unknown cloud model", 400)
            a.cloud_model_override = model_override

        loop = asyncio.new_event_loop()
        try:
            response = loop.run_until_complete(a.process(message))
        except Exception as e:
            logger.exception("Chat processing error")
            response = "Sorry, an error occurred while processing your request."
        finally:
            loop.close()

    elapsed = _time.time() - start

    with agent_lock:
        balances = a.cloud.get_balances()
        monthly_spend = a.cloud.monthly_spend

    return jsonify({
        "response": response,
        "model": config["llm"]["primary_model"],
        "cloud_model": a.cloud_model_override,
        "session_id": a.session_stats.session_id,
        "timestamp": datetime.now().isoformat(),
        "response_time": round(elapsed, 2),
        "balances": balances,
        "monthly_spend": round(monthly_spend, 4),
        "history_truncated": getattr(a, '_history_truncated', False),
        "verbose": getattr(a, '_verbose_log', []),
    })


# ── Session Management ──────────────────────────────────────────────────────

@app.route("/api/sessions", methods=["GET"])
@require_auth
def list_sessions():
    """List all saved chat sessions."""
    a = get_agent()
    sessions = a.session_stats.list_chat_sessions(limit=50)
    return jsonify({
        "sessions": sessions,
        "current_session": a.session_stats.session_id,
    })


@app.route("/api/sessions/new", methods=["POST"])
@require_auth
def new_session():
    """Start a new chat session."""
    with agent_lock:
        a = get_agent()
        a.new_session()
    return jsonify({
        "success": True,
        "session_id": a.session_stats.session_id,
    })


@app.route("/api/sessions/<session_id>/load", methods=["POST"])
@require_auth
def load_session(session_id):
    """Resume a previous session."""
    if not re.match(r'^[a-zA-Z0-9_-]+$', session_id):
        return safe_error("Invalid session ID", 400)
    with agent_lock:
        a = get_agent()
        success = a.load_session(session_id)
    if not success:
        return safe_error("Session not found", 404)
    return jsonify({
        "success": True,
        "session_id": a.session_stats.session_id,
        "messages": a.history,
    })


@app.route("/api/sessions/<session_id>/rename", methods=["POST"])
@require_auth
def rename_session(session_id):
    """Rename a chat session."""
    if not re.match(r'^[a-zA-Z0-9_-]+$', session_id):
        return safe_error("Invalid session ID", 400)
    data = request.json or {}
    title = data.get("title", "").strip()
    if not title:
        return safe_error("Title required", 400)
    if len(title) > 200:
        return safe_error("Title too long", 400)

    a = get_agent()
    if session_id == a.session_stats.session_id:
        a.session_stats.rename_session(title)
    else:
        session_file = a.session_stats.history_dir / f"chat_{session_id}.json"
        if session_file.exists():
            try:
                with open(session_file) as f:
                    sdata = json.load(f)
                sdata["title"] = title
                with open(session_file, "w") as f:
                    json.dump(sdata, f, indent=2)
            except (json.JSONDecodeError, IOError):
                return safe_error("Failed to rename session")
        else:
            return safe_error("Session not found", 404)

    return jsonify({"success": True})


@app.route("/api/sessions/<session_id>", methods=["DELETE"])
@require_auth
def delete_session(session_id):
    """Delete a saved chat session."""
    if not re.match(r'^[a-zA-Z0-9_-]+$', session_id):
        return safe_error("Invalid session ID", 400)
    a = get_agent()
    if a.session_stats.delete_chat_session(session_id):
        return jsonify({"success": True})
    return safe_error("Session not found", 404)


@app.route("/api/sessions/<session_id>/export", methods=["GET"])
@require_auth
def export_session(session_id):
    """Export a chat session as JSON download."""
    if not re.match(r'^[a-zA-Z0-9_-]+$', session_id):
        return safe_error("Invalid session ID", 400)
    a = get_agent()
    session_file = a.session_stats.history_dir / f"chat_{session_id}.json"
    if not session_file.exists():
        return safe_error("Session not found", 404)
    return send_file(session_file, as_attachment=True, download_name=f"chat_{session_id}.json")


# ── Cloud Model Selection & Balances ────────────────────────────────────────

@app.route("/api/cloud/models", methods=["GET"])
@require_auth
def list_cloud_models():
    """List all cloud models with pricing, availability, and balance info."""
    a = get_agent()
    models = []
    for model_id, info in CLOUD_MODELS.items():
        models.append({
            "id": model_id,
            "name": info["name"],
            "provider": info["provider"],
            "tier": info["tier"],
            "speed": info["speed"],
            "input_price": info["input"],
            "output_price": info["output"],
            "context": info["context"],
            "strengths": info["strengths"],
            "affordable": a.cloud.can_afford(model_id),
        })
    return jsonify({
        "models": models,
        "active_model": a.cloud_model_override,
        "default_anthropic": a.cloud.default_anthropic,
        "default_openai": a.cloud.default_openai,
    })


@app.route("/api/cloud/model", methods=["POST"])
@require_auth
def set_cloud_model():
    """Set the active cloud model (or 'auto' for smart routing)."""
    data = request.json or {}
    model_id = data.get("model", "auto")

    with agent_lock:
        a = get_agent()
        if model_id == "auto":
            a.cloud_model_override = None
        elif model_id in CLOUD_MODELS:
            a.cloud_model_override = model_id
        else:
            return safe_error("Unknown model", 400)

    return jsonify({
        "success": True,
        "active_model": a.cloud_model_override,
    })


@app.route("/api/cloud/balances", methods=["GET"])
@require_auth
def get_balances():
    """Get per-provider balances and spend."""
    with agent_lock:
        a = get_agent()
        balances = a.cloud.get_balances()
        provider_spend = dict(a.cloud.provider_spend)
        monthly_spend = a.cloud.monthly_spend
    return jsonify({
        "balances": balances,
        "provider_spend": provider_spend,
        "monthly_spend": round(monthly_spend, 4),
    })


@app.route("/api/cloud/balances", methods=["POST"])
@require_auth
def set_balances():
    """Update starting balances (e.g., after topping up an account)."""
    data = request.json or {}
    with agent_lock:
        a = get_agent()
        if "anthropic" in data:
            val = float(data["anthropic"])
            if val < 0:
                return safe_error("Balance cannot be negative", 400)
            a.cloud.provider_balances["anthropic"] = val
        if "openai" in data:
            val = float(data["openai"])
            if val < 0:
                return safe_error("Balance cannot be negative", 400)
            a.cloud.provider_balances["openai"] = val
        a.cloud._save_spend()
        balances = a.cloud.get_balances()
    return jsonify({"success": True, "balances": balances})


# ── Existing Routes (cleaned up) ────────────────────────────────────────────

@app.route("/api/models", methods=["GET"])
@require_auth
def list_models():
    """List locally available Ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10
        )
        models = []
        for line in result.stdout.strip().split("\n")[1:]:
            parts = line.split()
            if parts:
                name = parts[0]
                size = parts[2] + " " + parts[3] if len(parts) > 3 else "unknown"
                models.append({
                    "name": name,
                    "size": size,
                    "is_primary": name == config["llm"]["primary_model"],
                    "is_fast": name == config["llm"]["fast_model"],
                })
        return jsonify({"models": models})
    except Exception:
        return jsonify({"error": "Failed to list models", "models": []})


@app.route("/api/models/available", methods=["GET"])
@require_auth
def available_models():
    """List recommended models that can be downloaded."""
    models = [
        {"name": "llama3.2:3b", "size": "~2GB", "vram": "2GB", "desc": "Tiny, very fast", "tier": "light"},
        {"name": "llama3.1:8b", "size": "~4.7GB", "vram": "4GB", "desc": "Good all-rounder", "tier": "light"},
        {"name": "phi3:3.8b", "size": "~2.2GB", "vram": "2GB", "desc": "Microsoft — compact reasoning", "tier": "light"},
        {"name": "mistral:7b", "size": "~4.1GB", "vram": "4GB", "desc": "Fast and capable", "tier": "light"},
        {"name": "qwen2.5:14b", "size": "~9GB", "vram": "8GB", "desc": "Best tool calling at this size", "tier": "medium"},
        {"name": "deepseek-r1:14b", "size": "~9GB", "vram": "8GB", "desc": "Strong reasoning", "tier": "medium"},
        {"name": "gemma2:27b", "size": "~17GB", "vram": "16GB", "desc": "Very capable, needs RAM", "tier": "heavy"},
        {"name": "qwen2.5:32b", "size": "~20GB", "vram": "16GB", "desc": "Best reasoning under 70B", "tier": "heavy"},
        {"name": "mixtral:8x7b", "size": "~26GB", "vram": "24GB", "desc": "Mixture of experts", "tier": "heavy"},
        {"name": "llama3.1:70b", "size": "~42GB", "vram": "24GB+", "desc": "Maximum capability (needs swap)", "tier": "extreme"},
        {"name": "nomic-embed-text", "size": "~275MB", "vram": "CPU", "desc": "Embedding model for knowledge base", "tier": "utility"},
    ]
    return jsonify({"models": models})


@app.route("/api/models/pull", methods=["POST"])
@require_auth
@limiter.limit("5 per minute")
def pull_model():
    """Start downloading a model. Streams progress."""
    data = request.json or {}
    model_name = data.get("model", "")

    if not validate_model_name(model_name):
        return safe_error("Invalid model name", 400)

    def generate():
        process = subprocess.Popen(
            ["ollama", "pull", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in process.stdout:
            yield f"data: {json.dumps({'line': line.strip()})}\n\n"
        process.wait()
        yield f"data: {json.dumps({'done': True, 'success': process.returncode == 0})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/models/delete", methods=["POST"])
@require_auth
def delete_model():
    """Delete a locally downloaded model."""
    data = request.json or {}
    model_name = data.get("model", "")

    if not validate_model_name(model_name):
        return safe_error("Invalid model name", 400)

    try:
        result = subprocess.run(
            ["ollama", "rm", model_name],
            capture_output=True, text=True, timeout=30
        )
        return jsonify({"success": result.returncode == 0, "output": result.stdout})
    except Exception:
        return safe_error("Failed to delete model")


@app.route("/api/config", methods=["GET"])
@require_auth
def get_config():
    """Get current configuration (masks API keys)."""
    safe = json.loads(json.dumps(config))
    if "cloud" in safe:
        for k in ["anthropic_api_key", "openai_api_key"]:
            if safe["cloud"].get(k):
                v = safe["cloud"][k]
                safe["cloud"][k] = v[:8] + "..." + v[-4:] if len(v) > 14 else "***set***"
    return jsonify(safe)


@app.route("/api/config", methods=["POST"])
@require_auth
def update_config():
    """Update configuration and apply to running agent."""
    global config, agent
    data = request.json or {}

    for section, values in data.items():
        if section in config and isinstance(config[section], dict):
            config[section].update(values)
        else:
            config[section] = values

    # Save to disk (strip API keys)
    config_path = Path(__file__).parent / "config" / "settings.json"
    config_path.parent.mkdir(exist_ok=True)
    safe_config = json.loads(json.dumps(config))
    if "cloud" in safe_config:
        safe_config["cloud"].pop("anthropic_api_key", None)
        safe_config["cloud"].pop("openai_api_key", None)
    try:
        with open(config_path, "w") as f:
            json.dump(safe_config, f, indent=4)
    except IOError:
        return safe_error("Failed to save config")

    # Update running agent's config instead of destroying it
    with agent_lock:
        if agent is not None:
            agent.config = config

    return jsonify({"success": True})


@app.route("/api/mode", methods=["GET"])
@require_auth
def get_mode():
    a = get_agent()
    return jsonify({"mode": a.current_mode})


@app.route("/api/mode", methods=["POST"])
@require_auth
def set_mode():
    data = request.json or {}
    mode = data.get("mode", "general")
    a = get_agent()
    result = a.set_mode(mode)
    return jsonify({"success": "Invalid" not in result, "mode": a.mode, "message": result})


@app.route("/api/tools", methods=["GET"])
@require_auth
def list_tools():
    a = get_agent()
    enabled_tools = config.get("enabled_tools", {})
    tools = {}
    for name, instance in a.tool_instances.items():
        funcs = []
        for defn in instance.get_tool_definitions():
            funcs.append({
                "name": defn["function"]["name"],
                "description": defn["function"].get("description", ""),
            })
        is_enabled = enabled_tools.get(name, True)
        tools[name] = {"enabled": is_enabled, "function_count": len(funcs), "functions": funcs}
    return jsonify(tools)


@app.route("/api/tools/toggle", methods=["POST"])
@require_auth
def toggle_tool():
    data = request.json or {}
    tool_name = data.get("tool", "")
    enabled = data.get("enabled", True)

    if "enabled_tools" not in config:
        config["enabled_tools"] = {}
    config["enabled_tools"][tool_name] = enabled

    # Apply to running agent without destroying it
    with agent_lock:
        if agent is not None:
            agent.apply_tool_toggles(config.get("enabled_tools", {}))

    return jsonify({"success": True, "tool": tool_name, "enabled": enabled})


@app.route("/api/system", methods=["GET"])
@require_auth
@limiter.limit("30 per minute")
def system_info():
    import psutil

    info = {
        "ram": {
            "total": psutil.virtual_memory().total,
            "used": psutil.virtual_memory().used,
            "available": psutil.virtual_memory().available,
            "percent": psutil.virtual_memory().percent,
        },
        "swap": {
            "total": psutil.swap_memory().total,
            "used": psutil.swap_memory().used,
            "percent": psutil.swap_memory().percent,
        },
        "disk": {
            "total": psutil.disk_usage("/").total,
            "used": psutil.disk_usage("/").used,
            "percent": psutil.disk_usage("/").percent,
        },
        "cpu_percent": psutil.cpu_percent(interval=0.5),
        "cpu_count": psutil.cpu_count(),
    }

    # Network I/O
    try:
        net = psutil.net_io_counters()
        info["network"] = {
            "bytes_sent": net.bytes_sent,
            "bytes_recv": net.bytes_recv,
            "packets_sent": net.packets_sent,
            "packets_recv": net.packets_recv,
        }
    except Exception:
        info["network"] = None

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            info["gpu"] = {
                "name": parts[0],
                "vram_used_mb": int(parts[1]),
                "vram_total_mb": int(parts[2]),
                "utilization": int(parts[3]),
                "temp_c": int(parts[4]),
            }
    except Exception:
        info["gpu"] = None

    return jsonify(info)


@app.route("/api/knowledge/stats", methods=["GET"])
@require_auth
def knowledge_stats():
    a = get_agent()
    if "knowledge_base" in a.tool_instances:
        kb = a.tool_instances["knowledge_base"]
        files = kb.manifest.get("files", {})
        home = str(Path.home())

        # Breakdown by doc type via manifest file extensions
        ext_counts = {}
        total_size = 0
        for path, info in files.items():
            ext = Path(path).suffix.lower() or "(no ext)"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
            total_size += info.get("size", 0)

        # Top directories
        dir_counts = {}
        for path in files:
            rel = path.replace(home, "~")
            parts = Path(rel).parts
            top_dir = "/".join(parts[:3]) if len(parts) > 2 else "/".join(parts[:2])
            dir_counts[top_dir] = dir_counts.get(top_dir, 0) + 1

        return jsonify({
            "total_files": kb.manifest["stats"].get("total_files", 0),
            "total_chunks": kb.manifest["stats"].get("total_chunks", 0),
            "total_size": total_size,
            "db_path": str(kb.db_path),
            "by_extension": ext_counts,
            "by_directory": dict(sorted(dir_counts.items(), key=lambda x: -x[1])[:15]),
        })
    return jsonify({"total_files": 0, "total_chunks": 0, "db_path": "not loaded",
                     "total_size": 0, "by_extension": {}, "by_directory": {}})


@app.route("/api/knowledge/sources", methods=["GET"])
@require_auth
def knowledge_sources():
    """List all ingested sources with metadata."""
    a = get_agent()
    if "knowledge_base" not in a.tool_instances:
        return jsonify({"sources": []})

    kb = a.tool_instances["knowledge_base"]
    files = kb.manifest.get("files", {})
    home = str(Path.home())
    q = request.args.get("q", "").lower()

    sources = []
    for path, info in sorted(files.items()):
        if q and q not in path.lower():
            continue
        sources.append({
            "path": path,
            "display": path.replace(home, "~"),
            "filename": Path(path).name,
            "extension": Path(path).suffix,
            "chunks": info.get("chunks", 0),
            "size": info.get("size", 0),
            "ingested_at": info.get("ingested_at", ""),
            "hash": info.get("hash", ""),
        })

    return jsonify({"sources": sources, "total": len(sources)})


@app.route("/api/knowledge/search", methods=["POST"])
@require_auth
def knowledge_search():
    """Search the knowledge base."""
    a = get_agent()
    if "knowledge_base" not in a.tool_instances:
        return jsonify({"error": "Knowledge base not loaded"}), 400

    kb = a.tool_instances["knowledge_base"]
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "No query provided"}), 400

    max_results = min(data.get("max_results", 10), 20)
    filters = {}
    for key in ("filter_doc_type", "filter_topic", "filter_jurisdiction"):
        if data.get(key):
            filters[key] = data[key]

    result_text = kb.query_knowledge(query, max_results=max_results, **filters)
    return jsonify({"query": query, "results": result_text})


@app.route("/api/knowledge/ingest", methods=["POST"])
@require_auth
def knowledge_ingest():
    """Ingest a file or directory into the knowledge base."""
    a = get_agent()
    if "knowledge_base" not in a.tool_instances:
        return jsonify({"error": "Knowledge base not loaded"}), 400

    kb = a.tool_instances["knowledge_base"]
    data = request.get_json(silent=True) or {}
    path = data.get("path", "").strip()
    if not path:
        return jsonify({"error": "No path provided"}), 400

    target = Path(path).expanduser().resolve()
    if target.is_file():
        result = kb.ingest_file(str(target))
    elif target.is_dir():
        file_types = data.get("file_types", "all")
        result = kb.ingest_directory(str(target), recursive=True, file_types=file_types)
    else:
        return jsonify({"error": f"Path not found: {path}"}), 404

    return jsonify({"result": result})


@app.route("/api/knowledge/remove", methods=["POST"])
@require_auth
def knowledge_remove():
    """Remove a source from the knowledge base."""
    a = get_agent()
    if "knowledge_base" not in a.tool_instances:
        return jsonify({"error": "Knowledge base not loaded"}), 400

    kb = a.tool_instances["knowledge_base"]
    data = request.get_json(silent=True) or {}
    path = data.get("path", "").strip()
    if not path:
        return jsonify({"error": "No path provided"}), 400

    result = kb.remove_source(path)
    return jsonify({"result": result})


@app.route("/api/legal/files", methods=["GET"])
@require_auth
def legal_files():
    legal_dir = Path.home() / "LegalResearch"
    if not legal_dir.exists():
        return jsonify({"categories": {}})

    q = request.args.get("q", "").lower()
    categories = {}

    for d in sorted(legal_dir.iterdir()):
        if d.is_dir() and d.name != "projects" and not d.name.startswith("."):
            all_files = sorted(d.rglob("*"))
            all_files = [f for f in all_files if f.is_file() and not f.name.startswith(".")]
            if q:
                all_files = [f for f in all_files if q in f.name.lower()]

            categories[d.name] = [{
                "name": f.name,
                "path": str(f),
                "size": f.stat().st_size,
                "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                "relative": str(f.relative_to(d)),
            } for f in all_files[:100]]

    return jsonify({"categories": categories})


@app.route("/api/legal/read", methods=["POST"])
@require_auth
def legal_read():
    """Read the content of a legal research file."""
    data = request.get_json(silent=True) or {}
    path = data.get("path", "").strip()
    if not path:
        return jsonify({"error": "No path provided"}), 400

    filepath = Path(path)
    legal_dir = Path.home() / "LegalResearch"

    # Security: only allow reading files under LegalResearch
    try:
        filepath.resolve().relative_to(legal_dir.resolve())
    except ValueError:
        return jsonify({"error": "Access denied"}), 403

    if not filepath.exists() or not filepath.is_file():
        return jsonify({"error": "File not found"}), 404

    try:
        if filepath.suffix.lower() == ".pdf":
            return jsonify({"content": "(PDF file — view in a PDF reader)", "path": str(filepath)})
        text = filepath.read_text(encoding="utf-8", errors="replace")[:50000]
        return jsonify({"content": text, "path": str(filepath)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/legal/delete", methods=["POST"])
@require_auth
def legal_delete():
    """Delete a legal research file."""
    data = request.get_json(silent=True) or {}
    path = data.get("path", "").strip()
    if not path:
        return jsonify({"error": "No path provided"}), 400

    filepath = Path(path)
    legal_dir = Path.home() / "LegalResearch"

    try:
        filepath.resolve().relative_to(legal_dir.resolve())
    except ValueError:
        return jsonify({"error": "Access denied"}), 403

    if not filepath.exists():
        return jsonify({"error": "File not found"}), 404

    filepath.unlink()
    return jsonify({"success": True, "deleted": str(filepath)})


@app.route("/api/chat/clear", methods=["POST"])
@require_auth
def clear_chat():
    a = get_agent()
    if a.history:
        a.session_stats.save_chat_session(a.history)
    a.clear_history()
    return jsonify({"success": True, "session_id": a.session_stats.session_id})


# ── Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  Local AI Assistant — Dashboard")
    print("  Open in browser: http://localhost:5000\n")
    host = os.environ.get("DASHBOARD_HOST", "127.0.0.1")
    port = int(os.environ.get("DASHBOARD_PORT", "5000"))
    app.run(host=host, port=port, debug=False, threaded=True)
