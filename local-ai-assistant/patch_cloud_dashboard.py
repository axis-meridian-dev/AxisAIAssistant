#!/usr/bin/env python3
"""
Patch script — Adds cloud model controls to server.py and dashboard.html
Run from your project directory:
  python patch_cloud_dashboard.py
"""

import re
from pathlib import Path

PROJECT = Path(__file__).parent
SERVER = PROJECT / "server.py"
DASHBOARD = PROJECT / "dashboard.html"

# ── Patch server.py: Add /api/cloud routes ──────────────────────────────────

SERVER_PATCH = '''

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

'''

# ── Patch dashboard.html: Add Cloud Settings card ───────────────────────────

CLOUD_SETTINGS_HTML = '''
            <div class="card">
                <h3>☁️ Cloud Reasoning (Hybrid Mode)</h3>
                <div class="setting-row">
                    <div>
                        <div class="setting-label">Cloud Enabled</div>
                        <div class="setting-desc">Route complex/legal queries to cloud API for better reasoning</div>
                    </div>
                    <label class="toggle">
                        <input type="checkbox" id="setting-cloud-enabled" onchange="saveCloudSettings()">
                        <div class="toggle-slider"></div>
                    </label>
                </div>
                <div class="setting-row">
                    <div>
                        <div class="setting-label">Auto-Route</div>
                        <div class="setting-desc">Automatically send legal/complex queries to cloud (vs manual "use cloud")</div>
                    </div>
                    <label class="toggle">
                        <input type="checkbox" id="setting-cloud-autoroute" onchange="saveCloudSettings()">
                        <div class="toggle-slider"></div>
                    </label>
                </div>
                <div class="setting-row">
                    <div>
                        <div class="setting-label">Provider</div>
                        <div class="setting-desc">Which cloud API to use</div>
                    </div>
                    <select class="setting-input" id="setting-cloud-provider" onchange="saveCloudSettings()">
                        <option value="anthropic">Anthropic (Claude)</option>
                        <option value="openai">OpenAI (GPT-4)</option>
                    </select>
                </div>
                <div class="setting-row">
                    <div>
                        <div class="setting-label">Anthropic Model</div>
                        <div class="setting-desc">claude-opus-4-6 (best, $5/$25) or claude-sonnet-4-20250514 (fast, $3/$15)</div>
                    </div>
                    <select class="setting-input" id="setting-cloud-anthropic-model" onchange="saveCloudSettings()">
                        <option value="claude-opus-4-6">claude-opus-4-6</option>
                        <option value="claude-sonnet-4-20250514">claude-sonnet-4-20250514</option>
                        <option value="claude-haiku-4-5-20251001">claude-haiku-4-5 (cheapest)</option>
                    </select>
                </div>
                <div class="setting-row">
                    <div>
                        <div class="setting-label">OpenAI Model</div>
                        <div class="setting-desc">Used when provider is set to OpenAI</div>
                    </div>
                    <select class="setting-input" id="setting-cloud-openai-model" onchange="saveCloudSettings()">
                        <option value="gpt-4o">gpt-4o</option>
                        <option value="gpt-4o-mini">gpt-4o-mini (cheapest)</option>
                    </select>
                </div>
                <div class="setting-row">
                    <div>
                        <div class="setting-label">Monthly Budget</div>
                        <div class="setting-desc">Maximum monthly spend on cloud API (stops routing when exceeded)</div>
                    </div>
                    <input type="number" class="setting-input" id="setting-cloud-budget" 
                           min="1" max="200" step="5" value="40" onchange="saveCloudSettings()">
                </div>
                <div class="setting-row">
                    <div>
                        <div class="setting-label">Monthly Spend</div>
                        <div class="setting-desc">Current month's cloud API usage</div>
                    </div>
                    <span id="cloud-spend-display" style="font-family:var(--font-mono);color:var(--green)">$0.00</span>
                </div>
                <div class="setting-row">
                    <div>
                        <div class="setting-label">API Key Status</div>
                        <div class="setting-desc">Set in config/settings.json (never shown in dashboard)</div>
                    </div>
                    <span id="cloud-key-status" style="font-family:var(--font-mono);font-size:12px">checking...</span>
                </div>
            </div>
'''

CLOUD_JS = '''
// ── Cloud Settings ──────────────────────────────────────────────────────
async function loadCloudSettings() {
    try {
        const cloud = await fetch('/api/cloud').then(r=>r.json());
        document.getElementById('setting-cloud-enabled').checked = cloud.enabled || false;
        document.getElementById('setting-cloud-autoroute').checked = cloud.auto_route || false;
        document.getElementById('setting-cloud-provider').value = cloud.provider || 'anthropic';
        document.getElementById('setting-cloud-anthropic-model').value = cloud.anthropic_model || 'claude-sonnet-4-20250514';
        document.getElementById('setting-cloud-openai-model').value = cloud.openai_model || 'gpt-4o';
        document.getElementById('setting-cloud-budget').value = cloud.max_monthly_budget || 40;
        
        const spend = cloud.monthly_spend || 0;
        const budget = cloud.max_monthly_budget || 40;
        const remaining = cloud.budget_remaining || budget;
        document.getElementById('cloud-spend-display').textContent = `$${spend.toFixed(2)} / $${budget.toFixed(0)}`;
        document.getElementById('cloud-spend-display').style.color = 
            spend/budget > 0.8 ? 'var(--red)' : spend/budget > 0.5 ? 'var(--yellow)' : 'var(--green)';
        
        // Key status
        const hasAnthropicKey = cloud.anthropic_api_key && cloud.anthropic_api_key !== '';
        const hasOpenAIKey = cloud.openai_api_key && cloud.openai_api_key !== '';
        const keyEl = document.getElementById('cloud-key-status');
        if (cloud.provider === 'anthropic') {
            keyEl.textContent = hasAnthropicKey ? '✓ Anthropic key set' : '✗ No key';
            keyEl.style.color = hasAnthropicKey ? 'var(--green)' : 'var(--red)';
        } else {
            keyEl.textContent = hasOpenAIKey ? '✓ OpenAI key set' : '✗ No key';
            keyEl.style.color = hasOpenAIKey ? 'var(--green)' : 'var(--red)';
        }
    } catch(e) {
        console.error('Cloud settings error:', e);
    }
}

async function saveCloudSettings() {
    const cfg = {
        enabled: document.getElementById('setting-cloud-enabled').checked,
        auto_route: document.getElementById('setting-cloud-autoroute').checked,
        provider: document.getElementById('setting-cloud-provider').value,
        anthropic_model: document.getElementById('setting-cloud-anthropic-model').value,
        openai_model: document.getElementById('setting-cloud-openai-model').value,
        max_monthly_budget: parseFloat(document.getElementById('setting-cloud-budget').value),
    };
    
    await fetch('/api/cloud', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(cfg)
    });
    
    // Reload to show updated status
    setTimeout(loadCloudSettings, 500);
}
'''

def patch():
    print("Patching cloud controls into dashboard...")
    
    # ── Patch server.py ─────────────────────────────────────────────────
    if SERVER.exists():
        code = SERVER.read_text()
        if "/api/cloud" not in code:
            # Insert before the "# ── Run" section
            if '# ── Run' in code:
                code = code.replace('# ── Run', SERVER_PATCH + '\n# ── Run')
            elif 'if __name__' in code:
                code = code.replace('if __name__', SERVER_PATCH + '\nif __name__')
            else:
                code += SERVER_PATCH
            SERVER.write_text(code)
            print(f"  ✓ Patched {SERVER}")
        else:
            print(f"  ⏭ {SERVER} already has /api/cloud routes")
    else:
        print(f"  ✗ {SERVER} not found")
    
    # ── Patch dashboard.html ────────────────────────────────────────────
    if DASHBOARD.exists():
        html = DASHBOARD.read_text()
        
        # Add Cloud Settings card after Search Settings card
        if 'Cloud Reasoning' not in html:
            # Find the Search Settings card closing div and add after it
            search_card_end = html.find('</div>\n        </div>\n    </div>\n\n    <!-- Knowledge')
            if search_card_end == -1:
                # Try alternate: just before Knowledge Base page
                search_card_end = html.find('<!-- Knowledge Base Page')
                if search_card_end != -1:
                    html = html[:search_card_end] + CLOUD_SETTINGS_HTML + '\n        </div>\n    </div>\n\n    ' + html[search_card_end:]
            else:
                # Insert after the second card in settings
                insert_at = html.find('</div>\n            <div class="card">\n                <h3>Search Settings</h3>')
                if insert_at != -1:
                    # Find end of search card
                    search_end = html.find('</div>\n        </div>\n    </div>', insert_at + 100)
                    if search_end != -1:
                        html = html[:search_end] + '\n' + CLOUD_SETTINGS_HTML + html[search_end:]
            
            # If we still haven't found a spot, just add before </div> of page-content in settings
            if 'Cloud Reasoning' not in html:
                settings_content = html.find('id="page-settings"')
                if settings_content != -1:
                    # Find last </div> before next page
                    next_page = html.find('<!-- Knowledge', settings_content)
                    if next_page != -1:
                        html = html[:next_page] + CLOUD_SETTINGS_HTML + '\n    ' + html[next_page:]
            
            print(f"  ✓ Added Cloud Settings card to dashboard")
        else:
            print(f"  ⏭ Dashboard already has Cloud Reasoning section")
        
        # Add Cloud JS functions
        if 'loadCloudSettings' not in html:
            # Insert before the Init section
            init_marker = '// ── Init'
            if init_marker in html:
                html = html.replace(init_marker, CLOUD_JS + '\n' + init_marker)
            
            # Add loadCloudSettings() call to loadSettings function
            if 'async function loadSettings()' in html:
                # Add call at end of loadSettings
                html = html.replace(
                    "} catch(e) {}\n}\n\nasync function saveSettings",
                    "} catch(e) {}\n    loadCloudSettings();\n}\n\nasync function saveSettings"
                )
            
            print(f"  ✓ Added Cloud JS functions to dashboard")
        else:
            print(f"  ⏭ Dashboard already has cloud JS")
        
        DASHBOARD.write_text(html)
        print(f"  ✓ Saved {DASHBOARD}")
    else:
        print(f"  ✗ {DASHBOARD} not found")
    
    print("\nDone! Restart server.py to apply changes.")

if __name__ == "__main__":
    patch()
