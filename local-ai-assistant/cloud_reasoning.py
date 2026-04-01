"""
Hybrid Reasoning Layer — Local + Cloud LLM routing.

Architecture:
  LOCAL (Ollama) → Fast tasks: tool routing, file ops, simple queries
  CLOUD (Claude/OpenAI) → Complex tasks: legal analysis, document generation, deep reasoning

The local model handles 80% of interactions instantly.
The cloud model handles the 20% that requires serious reasoning.

Setup:
  pip install anthropic openai
  
  Set API keys in config/settings.json:
  {
    "cloud": {
      "provider": "anthropic",          # or "openai"
      "anthropic_api_key": "sk-ant-...",
      "openai_api_key": "sk-...",
      "anthropic_model": "claude-sonnet-4-20250514",
      "openai_model": "gpt-4o",
      "enabled": true,
      "auto_route": true,               # Auto-escalate complex queries
      "max_monthly_budget": 20.00,       # Spending cap
      "monthly_spend": 0.0
    }
  }
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Cost tracking ───────────────────────────────────────────────────────────

PRICING = {
    # Anthropic pricing (per million tokens, approximate)
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
    # OpenAI pricing
    "gpt-4o": {"input": 2.50, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


class CloudReasoner:
    """Routes queries to cloud LLMs for complex reasoning tasks."""
    
    def __init__(self, config: dict):
        self.config = config
        self.cloud_cfg = config.get("cloud", {})
        self.enabled = self.cloud_cfg.get("enabled", False)
        self.provider = self.cloud_cfg.get("provider", "anthropic")
        self.auto_route = self.cloud_cfg.get("auto_route", True)
        self.max_budget = self.cloud_cfg.get("max_monthly_budget", 20.0)
        
        # Track spending
        self.spend_file = Path.home() / ".local" / "share" / "ai-assistant" / "cloud_spend.json"
        self.spend_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_spend()
        
        # Lazy-load API clients
        self._anthropic = None
        self._openai = None
    
    @property
    def anthropic_client(self):
        if self._anthropic is None:
            try:
                import anthropic
                key = self.cloud_cfg.get("anthropic_api_key", "")
                if not key:
                    key = os.environ.get("ANTHROPIC_API_KEY", "")
                if key:
                    self._anthropic = anthropic.Anthropic(api_key=key)
                else:
                    print("  [Cloud] No Anthropic API key configured", flush=True)
            except ImportError:
                print("  [Cloud] pip install anthropic", flush=True)
        return self._anthropic
    
    @property
    def openai_client(self):
        if self._openai is None:
            try:
                import openai
                key = self.cloud_cfg.get("openai_api_key", "")
                if not key:
                    key = os.environ.get("OPENAI_API_KEY", "")
                if key:
                    self._openai = openai.OpenAI(api_key=key)
                else:
                    print("  [Cloud] No OpenAI API key configured", flush=True)
            except ImportError:
                print("  [Cloud] pip install openai", flush=True)
        return self._openai
    
    def _load_spend(self):
        """Load monthly spending tracker."""
        if self.spend_file.exists():
            with open(self.spend_file) as f:
                data = json.load(f)
            # Reset if new month
            if data.get("month") != datetime.now().strftime("%Y-%m"):
                self.monthly_spend = 0.0
                self._save_spend()
            else:
                self.monthly_spend = data.get("spend", 0.0)
        else:
            self.monthly_spend = 0.0
    
    def _save_spend(self):
        with open(self.spend_file, "w") as f:
            json.dump({
                "month": datetime.now().strftime("%Y-%m"),
                "spend": round(self.monthly_spend, 4),
                "updated": datetime.now().isoformat(),
            }, f, indent=2)
    
    def _track_cost(self, model: str, input_tokens: int, output_tokens: int):
        """Track API costs."""
        pricing = PRICING.get(model, {"input": 5.0, "output": 15.0})
        cost = (input_tokens / 1_000_000 * pricing["input"] + 
                output_tokens / 1_000_000 * pricing["output"])
        self.monthly_spend += cost
        self._save_spend()
        print(f"  [Cloud] Cost: ${cost:.4f} | Monthly: ${self.monthly_spend:.2f} / ${self.max_budget:.2f}", flush=True)
        return cost
    
    def can_afford(self) -> bool:
        """Check if within budget."""
        return self.monthly_spend < self.max_budget
    
    def should_use_cloud(self, query: str, intent: str, mode: str) -> bool:
        """
        Decide whether to route to cloud vs local.
        Cloud is used for:
          - Legal analysis and document generation
          - Complex reasoning that requires strong citation
          - Writing mode (essays, briefs, arguments)
          - When user explicitly requests it
        """
        if not self.enabled:
            return False
        if not self.auto_route:
            return False
        if not self.can_afford():
            print(f"  [Cloud] Budget exceeded (${self.monthly_spend:.2f} / ${self.max_budget:.2f})", flush=True)
            return False
        
        lower = query.lower()
        
        # Explicit cloud request
        if any(p in lower for p in ["use cloud", "use claude", "use gpt", "cloud mode",
                                     "better model", "full power", "deep analysis"]):
            return True
        
        # Auto-route based on intent and mode
        if intent == "legal" and mode in ("analysis", "argument", "writing"):
            return True
        
        # Document generation always uses cloud
        if any(p in lower for p in ["write a brief", "write a memo", "write an essay",
                                     "draft a motion", "create a document", "generate a report",
                                     "legal memorandum", "debate prep"]):
            return True
        
        # Complex multi-step reasoning
        if any(p in lower for p in ["analyze this case", "compare these", "build arguments",
                                     "what are my options", "evaluate the strength",
                                     "prepare for court", "research brief"]):
            return True
        
        return False
    
    def query_anthropic(self, messages: list, system_prompt: str, 
                         tool_results: str = "") -> Optional[str]:
        """Send query to Anthropic Claude API."""
        client = self.anthropic_client
        if not client:
            return None
        
        model = self.cloud_cfg.get("anthropic_model", "claude-sonnet-4-20250514")
        
        # Build message list for Claude API format
        api_messages = []
        for msg in messages:
            if msg["role"] == "user":
                api_messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                api_messages.append({"role": "assistant", "content": msg["content"]})
            elif msg["role"] == "tool":
                # Inject tool results as user context
                api_messages.append({
                    "role": "user", 
                    "content": f"[Tool Result]\n{msg['content']}"
                })
        
        # Add tool results context if available
        if tool_results:
            if api_messages and api_messages[-1]["role"] == "user":
                api_messages[-1]["content"] += f"\n\n[Research Data from Local Tools]\n{tool_results}"
            else:
                api_messages.append({
                    "role": "user",
                    "content": f"[Research Data from Local Tools]\n{tool_results}"
                })
        
        # Ensure alternating user/assistant messages
        cleaned = []
        for msg in api_messages:
            if cleaned and cleaned[-1]["role"] == msg["role"]:
                cleaned[-1]["content"] += "\n\n" + msg["content"]
            else:
                cleaned.append(msg)
        
        # Ensure starts with user message
        if cleaned and cleaned[0]["role"] != "user":
            cleaned.insert(0, {"role": "user", "content": "(continuing conversation)"})
        
        try:
            print(f"  [Cloud] Sending to {model}...", flush=True)
            start = time.time()
            
            response = client.messages.create(
                model=model,
                max_tokens=8192,
                system=system_prompt,
                messages=cleaned,
            )
            
            elapsed = time.time() - start
            
            # Extract text
            result = ""
            for block in response.content:
                if hasattr(block, "text"):
                    result += block.text
            
            # Track costs
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            self._track_cost(model, input_tokens, output_tokens)
            
            print(f"  [Cloud] Response: {len(result)} chars in {elapsed:.1f}s "
                  f"({input_tokens} in / {output_tokens} out)", flush=True)
            
            return result
            
        except Exception as e:
            print(f"  [Cloud] Anthropic error: {e}", flush=True)
            return None
    
    def query_openai(self, messages: list, system_prompt: str,
                      tool_results: str = "") -> Optional[str]:
        """Send query to OpenAI API."""
        client = self.openai_client
        if not client:
            return None
        
        model = self.cloud_cfg.get("openai_model", "gpt-4o")
        
        # Build message list for OpenAI format
        api_messages = [{"role": "system", "content": system_prompt}]
        
        for msg in messages:
            if msg["role"] in ("user", "assistant"):
                api_messages.append({"role": msg["role"], "content": msg["content"]})
            elif msg["role"] == "tool":
                api_messages.append({
                    "role": "user",
                    "content": f"[Tool Result]\n{msg['content']}"
                })
        
        if tool_results:
            api_messages.append({
                "role": "user",
                "content": f"[Research Data from Local Tools]\n{tool_results}"
            })
        
        try:
            print(f"  [Cloud] Sending to {model}...", flush=True)
            start = time.time()
            
            response = client.chat.completions.create(
                model=model,
                messages=api_messages,
                max_tokens=8192,
                temperature=0.3,
            )
            
            elapsed = time.time() - start
            result = response.choices[0].message.content
            
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            self._track_cost(model, input_tokens, output_tokens)
            
            print(f"  [Cloud] Response: {len(result)} chars in {elapsed:.1f}s "
                  f"({input_tokens} in / {output_tokens} out)", flush=True)
            
            return result
            
        except Exception as e:
            print(f"  [Cloud] OpenAI error: {e}", flush=True)
            return None
    
    def query(self, messages: list, system_prompt: str,
              tool_results: str = "") -> Optional[str]:
        """Route to configured cloud provider."""
        if self.provider == "anthropic":
            return self.query_anthropic(messages, system_prompt, tool_results)
        elif self.provider == "openai":
            return self.query_openai(messages, system_prompt, tool_results)
        else:
            print(f"  [Cloud] Unknown provider: {self.provider}", flush=True)
            return None
    
    def get_status(self) -> str:
        """Get cloud reasoning status."""
        if not self.enabled:
            return "Cloud reasoning: DISABLED"
        
        return (
            f"Cloud reasoning: ENABLED\n"
            f"  Provider: {self.provider}\n"
            f"  Model: {self.cloud_cfg.get(f'{self.provider}_model', 'unknown')}\n"
            f"  Auto-route: {'ON' if self.auto_route else 'OFF'}\n"
            f"  Monthly spend: ${self.monthly_spend:.2f} / ${self.max_budget:.2f}\n"
            f"  Budget remaining: ${max(0, self.max_budget - self.monthly_spend):.2f}"
        )
