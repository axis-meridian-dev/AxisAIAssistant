#!/usr/bin/env python3
"""
Local AI Assistant — Web Dashboard Server
Serves a browser-based control panel + chat interface.

Run: python server.py
Then open: http://localhost:5000
"""

import json
import asyncio
import subprocess
import threading
from pathlib import Path
from datetime import datetime

from flask import Flask, request, jsonify, send_file, Response
import ollama

from agent import Agent
from config import load_config
from cloud_reasoning import MODELS as CLOUD_MODELS
from log import setup_logging

setup_logging()

app = Flask(__name__, template_folder="templates")

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
    return send_file("templates/dashboard.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """Send a message to the agent and get a response."""
    data = request.json
    message = data.get("message", "")

    if not message.strip():
        return jsonify({"error": "Empty message"}), 400

    import time as _time
    start = _time.time()

    with agent_lock:
        a = get_agent()
        # Apply model override if provided
        model_override = data.get("cloud_model")
        if model_override:
            a.cloud_model_override = model_override

        loop = asyncio.new_event_loop()
        try:
            response = loop.run_until_complete(a.process(message))
        except Exception as e:
            response = f"Error: {e}"
        finally:
            loop.close()

    elapsed = _time.time() - start
    balances = a.cloud.get_balances()

    return jsonify({
        "response": response,
        "model": config["llm"]["primary_model"],
        "cloud_model": a.cloud_model_override,
        "session_id": a.session_stats.session_id,
        "timestamp": datetime.now().isoformat(),
        "response_time": round(elapsed, 2),
        "balances": balances,
        "monthly_spend": round(a.cloud.monthly_spend, 4),
    })


# ── Session Management ──────────────────────────────────────────────────────

@app.route("/api/sessions", methods=["GET"])
def list_sessions():
    """List all saved chat sessions."""
    a = get_agent()
    sessions = a.session_stats.list_chat_sessions(limit=50)
    return jsonify({
        "sessions": sessions,
        "current_session": a.session_stats.session_id,
    })


@app.route("/api/sessions/new", methods=["POST"])
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
def load_session(session_id):
    """Resume a previous session."""
    with agent_lock:
        a = get_agent()
        success = a.load_session(session_id)
    if not success:
        return jsonify({"error": "Session not found"}), 404
    return jsonify({
        "success": True,
        "session_id": a.session_stats.session_id,
        "messages": a.history,
    })


@app.route("/api/sessions/<session_id>/rename", methods=["POST"])
def rename_session(session_id):
    """Rename a chat session."""
    data = request.json
    title = data.get("title", "").strip()
    if not title:
        return jsonify({"error": "Title required"}), 400

    a = get_agent()
    # If renaming current session
    if session_id == a.session_stats.session_id:
        a.session_stats.rename_session(title)
    else:
        # Load, rename, don't switch
        session_file = a.session_stats.history_dir / f"chat_{session_id}.json"
        if session_file.exists():
            with open(session_file) as f:
                sdata = json.load(f)
            sdata["title"] = title
            with open(session_file, "w") as f:
                json.dump(sdata, f, indent=2)
        else:
            return jsonify({"error": "Session not found"}), 404

    return jsonify({"success": True})


@app.route("/api/sessions/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    """Delete a saved chat session."""
    a = get_agent()
    if a.session_stats.delete_chat_session(session_id):
        return jsonify({"success": True})
    return jsonify({"error": "Session not found"}), 404


# ── Cloud Model Selection & Balances ────────────────────────────────────────

@app.route("/api/cloud/models", methods=["GET"])
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
def set_cloud_model():
    """Set the active cloud model (or 'auto' for smart routing)."""
    data = request.json
    model_id = data.get("model", "auto")

    with agent_lock:
        a = get_agent()
        if model_id == "auto":
            a.cloud_model_override = None
        elif model_id in CLOUD_MODELS:
            a.cloud_model_override = model_id
        else:
            return jsonify({"error": f"Unknown model: {model_id}"}), 400

    return jsonify({
        "success": True,
        "active_model": a.cloud_model_override,
    })


@app.route("/api/cloud/balances", methods=["GET"])
def get_balances():
    """Get per-provider balances and spend."""
    a = get_agent()
    return jsonify({
        "balances": a.cloud.get_balances(),
        "provider_spend": dict(a.cloud.provider_spend),
        "monthly_spend": round(a.cloud.monthly_spend, 4),
    })


@app.route("/api/cloud/balances", methods=["POST"])
def set_balances():
    """Update starting balances (e.g., after topping up an account)."""
    data = request.json
    a = get_agent()
    if "anthropic" in data:
        a.cloud.provider_balances["anthropic"] = float(data["anthropic"])
    if "openai" in data:
        a.cloud.provider_balances["openai"] = float(data["openai"])
    a.cloud._save_spend()
    return jsonify({"success": True, "balances": a.cloud.get_balances()})


# ── Existing Routes (cleaned up) ────────────────────────────────────────────

@app.route("/api/models", methods=["GET"])
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
    except Exception as e:
        return jsonify({"error": str(e), "models": []})


@app.route("/api/models/available", methods=["GET"])
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
def pull_model():
    """Start downloading a model. Streams progress."""
    data = request.json
    model_name = data.get("model", "")

    if not model_name:
        return jsonify({"error": "No model specified"}), 400

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
def delete_model():
    """Delete a locally downloaded model."""
    data = request.json
    model_name = data.get("model", "")

    try:
        result = subprocess.run(
            ["ollama", "rm", model_name],
            capture_output=True, text=True, timeout=30
        )
        return jsonify({"success": result.returncode == 0, "output": result.stdout})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/config", methods=["GET"])
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
def update_config():
    """Update configuration and reload agent."""
    global config, agent
    data = request.json

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
    with open(config_path, "w") as f:
        json.dump(safe_config, f, indent=4)

    with agent_lock:
        agent = None

    return jsonify({"success": True})


@app.route("/api/mode", methods=["GET"])
def get_mode():
    a = get_agent()
    return jsonify({"mode": a.current_mode})


@app.route("/api/mode", methods=["POST"])
def set_mode():
    data = request.json
    mode = data.get("mode", "general")
    a = get_agent()
    result = a.set_mode(mode)
    return jsonify({"success": "Invalid" not in result, "mode": a.mode, "message": result})


@app.route("/api/tools", methods=["GET"])
def list_tools():
    a = get_agent()
    tools = {}
    for name, instance in a.tool_instances.items():
        funcs = []
        for defn in instance.get_tool_definitions():
            funcs.append({
                "name": defn["function"]["name"],
                "description": defn["function"].get("description", ""),
            })
        tools[name] = {"enabled": True, "function_count": len(funcs), "functions": funcs}
    return jsonify(tools)


@app.route("/api/tools/toggle", methods=["POST"])
def toggle_tool():
    global agent
    data = request.json
    tool_name = data.get("tool", "")
    enabled = data.get("enabled", True)

    if "enabled_tools" not in config:
        config["enabled_tools"] = {}
    config["enabled_tools"][tool_name] = enabled

    with agent_lock:
        agent = None

    return jsonify({"success": True, "tool": tool_name, "enabled": enabled})


@app.route("/api/system", methods=["GET"])
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
def knowledge_stats():
    a = get_agent()
    if "knowledge_base" in a.tool_instances:
        kb = a.tool_instances["knowledge_base"]
        return jsonify({
            "total_files": kb.manifest["stats"].get("total_files", 0),
            "total_chunks": kb.manifest["stats"].get("total_chunks", 0),
            "db_path": str(kb.db_path),
        })
    return jsonify({"total_files": 0, "total_chunks": 0, "db_path": "not loaded"})


@app.route("/api/legal/files", methods=["GET"])
def legal_files():
    legal_dir = Path.home() / "LegalResearch"
    if not legal_dir.exists():
        return jsonify({"categories": {}})

    categories = {}
    for d in sorted(legal_dir.iterdir()):
        if d.is_dir():
            files = sorted(d.glob("*"))
            files = [f for f in files if f.is_file()]
            categories[d.name] = [{
                "name": f.name,
                "size": f.stat().st_size,
                "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            } for f in files[:50]]

    return jsonify({"categories": categories})


@app.route("/api/chat/clear", methods=["POST"])
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
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
