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

app = Flask(__name__)

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
    return send_file("dashboard.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """Send a message to the agent and get a response."""
    data = request.json
    message = data.get("message", "")
    
    if not message.strip():
        return jsonify({"error": "Empty message"}), 400
    
    with agent_lock:
        a = get_agent()
        loop = asyncio.new_event_loop()
        try:
            response = loop.run_until_complete(a.process(message))
        except Exception as e:
            response = f"Error: {e}"
        finally:
            loop.close()
    
    return jsonify({
        "response": response,
        "model": config["llm"]["primary_model"],
        "timestamp": datetime.now().isoformat()
    })


@app.route("/api/models", methods=["GET"])
def list_models():
    """List all locally available Ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10
        )
        models = []
        for line in result.stdout.strip().split("\n")[1:]:  # Skip header
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
    """Get current configuration."""
    return jsonify(config)


@app.route("/api/config", methods=["POST"])
def update_config():
    """Update configuration and reload agent."""
    global config, agent
    data = request.json
    
    # Deep merge
    for section, values in data.items():
        if section in config and isinstance(config[section], dict):
            config[section].update(values)
        else:
            config[section] = values
    
    # Save to disk
    config_path = Path(__file__).parent / "config" / "settings.json"
    config_path.parent.mkdir(exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    
    # Reload agent
    with agent_lock:
        agent = None
    
    return jsonify({"success": True, "config": config})


@app.route("/api/mode", methods=["GET"])
def get_mode():
    """Get current agent mode."""
    a = get_agent()
    return jsonify({"mode": a.mode})


@app.route("/api/mode", methods=["POST"])
def set_mode():
    """Set agent operating mode."""
    data = request.json
    mode = data.get("mode", "general")
    a = get_agent()
    result = a.set_mode(mode)
    return jsonify({"success": "Invalid" not in result, "mode": a.mode, "message": result})


@app.route("/api/tools", methods=["GET"])
def list_tools():
    """List all available tool modules and their functions."""
    a = get_agent()
    tools = {}

    # All known tool names
    all_tools = [
        "file_manager", "web_search", "desktop_control", "system_info",
        "knowledge_base", "legal_research", "document_writer"
    ]

    for name in all_tools:
        if name in a.tool_instances:
            instance = a.tool_instances[name]
            funcs = []
            for defn in instance.get_tool_definitions():
                funcs.append({
                    "name": defn["function"]["name"],
                    "description": defn["function"].get("description", ""),
                })
            tools[name] = {
                "enabled": True,
                "function_count": len(funcs),
                "functions": funcs,
            }
        else:
            tools[name] = {"enabled": False, "function_count": 0, "functions": []}

    return jsonify(tools)


@app.route("/api/tools/toggle", methods=["POST"])
def toggle_tool():
    """Enable or disable a tool module. Requires agent reload."""
    global agent
    data = request.json
    tool_name = data.get("tool", "")
    enabled = data.get("enabled", True)
    
    # Store enabled/disabled state in config
    if "enabled_tools" not in config:
        config["enabled_tools"] = {}
    config["enabled_tools"][tool_name] = enabled
    
    # Save config
    config_path = Path(__file__).parent / "config" / "settings.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    
    # Reload agent
    with agent_lock:
        agent = None
    
    return jsonify({"success": True, "tool": tool_name, "enabled": enabled})


@app.route("/api/system", methods=["GET"])
def system_info():
    """Get system stats — RAM, GPU, disk, swap."""
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
    
    # GPU info
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
    """Get knowledge base statistics."""
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
    """List legal research library contents."""
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
    """Clear conversation history."""
    a = get_agent()
    a.clear_history()
    return jsonify({"success": True})


# ── Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  Local AI Assistant — Dashboard")
    print("  Open in browser: http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
