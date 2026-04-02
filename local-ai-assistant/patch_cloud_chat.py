#!/usr/bin/env python3
"""
Patch: Add Cloud Chat mode to server.py and dashboard.html
Adds a toggle in the chat header to switch between Local (Ollama) and Cloud (Claude API).
Cloud mode sends messages directly to the Anthropic API with your legal research context.

Run: python patch_cloud_chat.py
"""

from pathlib import Path

PROJECT = Path(__file__).parent
SERVER = PROJECT / "server.py"
DASHBOARD = PROJECT / "dashboard.html"

# ── Server route for cloud chat ─────────────────────────────────────────────

SERVER_ROUTE = '''

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
                cleaned[-1]["content"] += "\\n\\n" + msg["content"]
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

'''


def patch():
    print("Adding Cloud Chat to dashboard...")
    
    # ── Patch server.py ─────────────────────────────────────────────────
    if SERVER.exists():
        code = SERVER.read_text()
        if "/api/chat/cloud" not in code:
            if '# ── Run' in code:
                code = code.replace('# ── Run', SERVER_ROUTE + '\n# ── Run')
            elif 'if __name__' in code:
                code = code.replace('if __name__', SERVER_ROUTE + '\nif __name__')
            else:
                code += SERVER_ROUTE
            SERVER.write_text(code)
            print(f"  ✓ Added /api/chat/cloud route to server.py")
        else:
            print(f"  ⏭ server.py already has cloud chat route")
    else:
        print(f"  ✗ server.py not found")
    
    # ── Patch dashboard.html ────────────────────────────────────────────
    if DASHBOARD.exists():
        html = DASHBOARD.read_text()
        
        # 1. Add chat mode toggle button next to model badge
        if 'chat-mode-toggle' not in html:
            old_badge = '''<span class="model-badge" id="active-model">loading...</span>'''
            new_badge = '''<button class="btn btn-sm" id="chat-mode-toggle" onclick="toggleChatMode()" 
                        style="font-size:11px;padding:4px 10px;">🖥 Local</button>
                <span class="model-badge" id="active-model">loading...</span>'''
            html = html.replace(old_badge, new_badge)
            print("  ✓ Added chat mode toggle button")
        
        # 2. Add cloud chat JS functions
        if 'toggleChatMode' not in html:
            cloud_chat_js = '''
// ── Cloud Chat Mode ──────────────────────────────────────────────────────
let chatMode = 'local';  // 'local' or 'cloud'
let cloudHistory = [];

function toggleChatMode() {
    chatMode = chatMode === 'local' ? 'cloud' : 'local';
    const btn = document.getElementById('chat-mode-toggle');
    if (chatMode === 'cloud') {
        btn.textContent = '☁️ Cloud';
        btn.style.background = 'var(--accent-glow)';
        btn.style.borderColor = 'var(--accent)';
        btn.style.color = 'var(--accent)';
    } else {
        btn.textContent = '🖥 Local';
        btn.style.background = '';
        btn.style.borderColor = '';
        btn.style.color = '';
    }
}

'''
            # Insert before the existing sendMessage function
            old_send = 'async function sendMessage() {'
            html = html.replace(old_send, cloud_chat_js + old_send)
            print("  ✓ Added toggleChatMode function")
        
        # 3. Modify sendMessage to check chatMode
        if "chatMode === 'cloud'" not in html:
            # Replace the fetch call in sendMessage
            old_fetch = """try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({message: msg})
        });
        const data = await res.json();
        addMessage('assistant', data.response, data.model);
    }"""
            
            new_fetch = """try {
        let data;
        if (chatMode === 'cloud') {
            const res = await fetch('/api/chat/cloud', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message: msg, history: cloudHistory})
            });
            data = await res.json();
            cloudHistory.push({role: 'user', content: msg});
            cloudHistory.push({role: 'assistant', content: data.response});
            if (cloudHistory.length > 40) cloudHistory = cloudHistory.slice(-30);
            const costInfo = data.cost ? ` · $${data.cost.toFixed(4)}` : '';
            addMessage('assistant', data.response, (data.model || 'cloud') + costInfo);
        } else {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message: msg})
            });
            data = await res.json();
            addMessage('assistant', data.response, data.model);
        }
    }"""
            
            if old_fetch in html:
                html = html.replace(old_fetch, new_fetch)
                print("  ✓ Modified sendMessage for dual-mode chat")
            else:
                print("  ⚠ Could not find exact sendMessage pattern — may need manual edit")
        
        DASHBOARD.write_text(html)
        print(f"  ✓ Saved dashboard.html")
    else:
        print(f"  ✗ dashboard.html not found")
    
    print("\nDone! Restart server.py and you'll see a Local/Cloud toggle in the chat header.")
    print("Click it to switch between Ollama (free, local) and Claude API (paid, powerful).")

if __name__ == "__main__":
    patch()
