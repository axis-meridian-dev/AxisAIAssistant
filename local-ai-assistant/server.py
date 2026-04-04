from config_utils import save_config_safe
from config_utils import save_config_safe
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

    import time as _time
    start = _time.time()

    with agent_lock:
        a = get_agent()
        loop = asyncio.new_event_loop()
        try:
            response = loop.run_until_complete(a.process(message))
        except Exception as e:
            response = f"Error: {e}"
        finally:
            loop.close()

    elapsed = _time.time() - start

    return jsonify({
        "response": response,
        "model": config["llm"]["primary_model"],
        "timestamp": datetime.now().isoformat(),
        "response_time": round(elapsed, 2),
    })


@app.route("/api/models", methods=["GET"])
def list_models():
    """List all locally available Ollama.current_models."""
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
    """Start downloading a.current_model. Streams progress."""
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
    return jsonify({"mode": a.current_mode})


@app.route("/api/mode", methods=["POST"])
def set_mode():
    """Set agent operating mode."""
    data = request.json
    mode = data.get("mode", "general")
    a = get_agent()
    result = setattr(a, "current_mode", mode) or mode
    return jsonify({"success": "Invalid" not in result, "mode": a.current_mode, "message": result})


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
    """Clear conversation history, saving the old session first."""
    a = get_agent()
    if a.history:
        a.session_stats.save_chat_session(a.history)
    a.clear_history()
    return jsonify({"success": True})


@app.route("/api/chat/history", methods=["GET"])
def chat_history_list():
    """List saved chat sessions."""
    a = get_agent()
    sessions = a.session_stats.list_chat_sessions(limit=30)
    return jsonify({"sessions": sessions})


@app.route("/api/chat/history/<session_id>", methods=["GET"])
def chat_history_load(session_id):
    """Load a specific chat session."""
    a = get_agent()
    session = a.session_stats.load_chat_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(session)


@app.route("/api/chat/history/<session_id>", methods=["DELETE"])
def chat_history_delete(session_id):
    """Delete a saved chat session."""
    a = get_agent()
    session_file = a.session_stats.history_dir / f"chat_{session_id}.json"
    if session_file.exists():
        session_file.unlink()
        return jsonify({"success": True})
    return jsonify({"error": "Session not found"}), 404




@app.route("/api/cloud", methods=["GET"])
def get_cloud_config():
    """Get cloud reasoning configuration (hides API key)."""
    cloud = config.get("cloud", {})
    safe = dict(cloud)
    # Mask the API keys
    for k in ["anthropic_api_key", "openai_api_key"]:
        if safe.get(k):
            safe[k] = safe[k][:10] + "..." + safe[k][-4:] if len(safe.get(k,"")) > 14 else "***set***"
    
    # Add spend info
    try:
        from cloud_reasoning import CloudReasoner
        cr = CloudReasoner(config)
        safe["monthly_spend"] = round(cr.monthly_spend, 4)
        safe["budget_remaining"] = round(max(0, cr.max_budget - cr.monthly_spend), 2)
    except:
        safe["monthly_spend"] = 0
        safe["budget_remaining"] = safe.get("max_monthly_budget", 0)
    
    return jsonify(safe)


@app.route("/api/cloud", methods=["POST"])
def update_cloud_config():
    """Update cloud reasoning settings."""
    global config, agent
    data = request.json
    
    if "cloud" not in config:
        config["cloud"] = {}
    
    # Update only provided fields
    for key in ["provider", "anthropic_model", "openai_model", "enabled", 
                 "auto_route", "max_monthly_budget"]:
        if key in data:
            config["cloud"][key] = data[key]
    
    # Handle API key updates (only if non-masked value provided)
    for key in ["anthropic_api_key", "openai_api_key"]:
        if key in data and "..." not in data[key] and data[key] != "***set***":
            config["cloud"][key] = data[key]
    
    # Save config
    config_path = Path(__file__).parent / "config" / "settings.json"
    config_path.parent.mkdir(exist_ok=True)
    import json as j
    with open(config_path, "w") as f:
        j.dump(config, f, indent=4)
    
    # Reload agent
    with agent_lock:
        agent = None
    
    return jsonify({"success": True})




@app.route("/api/chat/cloud", methods=["POST"])
def chat_cloud():
    """Send a message directly to Claude API (bypasses Ollama entirely)."""
    data = request.json
    message = data.get("message", "")
    history = data.get("history", [])
    
    if not message.strip():
        return jsonify({"error": "Empty message"}), 400
    
    cloud_cfg = config.get("cloud", {})
    if not cloud_cfg.get("enabled"):
        return jsonify({"error": "Cloud not enabled. Enable in Settings.", "response": "Cloud mode is not enabled. Go to Settings > Cloud Reasoning and enable it."}), 200
    
    api_key = cloud_cfg.get("anthropic_api_key", "")
    if not api_key:
        return jsonify({"error": "No API key", "response": "No Anthropic API key configured. Add it to config/settings.json"}), 200
    
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        model = cloud_cfg.get("anthropic_model", "claude-sonnet-4-20250514")
        
        # Build messages
        api_messages = []
        for msg in history[-20:]:  # Last 20 messages for context
            if msg.get("role") in ("user", "assistant"):
                api_messages.append({"role": msg["role"], "content": msg["content"]})
        api_messages.append({"role": "user", "content": message})
        
        # Ensure alternating roles
        cleaned = []
        for msg in api_messages:
            if cleaned and cleaned[-1]["role"] == msg["role"]:
                cleaned[-1]["content"] += "\n\n" + msg["content"]
            else:
                cleaned.append(msg)
        if cleaned and cleaned[0]["role"] != "user":
            cleaned.insert(0, {"role": "user", "content": "(continuing)"})
        
        system_prompt = """You are a legal research assistant for a defendant in Connecticut with active court cases.
You have access to a local knowledge base of 430+ legal files including CT General Statutes, federal statutes, 
case law, international human rights instruments, and crime statistics.

The user is Jonathan Sewell, a US Army National Guard infantry veteran (Bravo Company, 1/102nd, Middletown CT) 
with a registered PTSD service dog. He has active cases in Rockville Superior Court stemming from a 
August 27, 2024 incident in Mansfield, CT. He has a 15+ year history of encounters with CT law enforcement 
across multiple departments that he believes constitute a pattern of civil rights violations.

RULES:
1. Always cite specific statutes (CGS sections, USC sections) and case law with full citations
2. Be direct, thorough, and adversarial on behalf of the defendant
3. Reference CT-specific law including Article I § 7 (broader than 4th Amendment) and PA 20-1 (Police Accountability Act)
4. When discussing his cases, identify every weakness in the state's case
5. End legal responses with confidence scoring
6. This is legal research, not legal advice — include disclaimer when producing formal analysis

Be direct. No filler. The user is intelligent and capable — he built this entire AI system himself."""

        response = client.messages.create(
            model=model,
            max_tokens=8192,
            system=system_prompt,
            messages=cleaned,
        )
        
        result = ""
        for block in response.content:
            if hasattr(block, "text"):
                result += block.text
        
        # Track cost
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        
        pricing = {
            "claude-opus-4-6": {"input": 5.0, "output": 25.0},
            "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
            "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
        }
        rates = pricing.get(model, {"input": 5.0, "output": 25.0})
        cost = (input_tokens / 1_000_000 * rates["input"] + 
                output_tokens / 1_000_000 * rates["output"])
        
        # Update spend tracker
        import json as j
        spend_file = Path.home() / ".local" / "share" / "ai-assistant" / "cloud_spend.json"
        spend_file.parent.mkdir(parents=True, exist_ok=True)
        spend_data = {"month": "", "spend": 0}
        if spend_file.exists():
            with open(spend_file) as f:
                spend_data = j.load(f)
        
        from datetime import datetime as dt
        current_month = dt.now().strftime("%Y-%m")
        if spend_data.get("month") != current_month:
            spend_data = {"month": current_month, "spend": 0}
        spend_data["spend"] = round(spend_data["spend"] + cost, 4)
        spend_data["updated"] = dt.now().isoformat()
        with open(spend_file, "w") as f:
            j.dump(spend_data, f, indent=2)
        
        return jsonify({
            "response": result,
            "model": model,
            "cost": round(cost, 4),
            "monthly_spend": spend_data["spend"],
            "tokens": {"input": input_tokens, "output": output_tokens},
            "timestamp": dt.now().isoformat()
        })
        
    except ImportError:
        return jsonify({"response": "Error: pip install anthropic", "model": "error"})
    except Exception as e:
        return jsonify({"response": f"Cloud error: {e}", "model": "error"})


# ── Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  Local AI Assistant — Dashboard")
    print("  Open in browser: http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
