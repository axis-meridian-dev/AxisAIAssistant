"""
Hybrid Reasoning Layer v2 — Multi-Provider Cloud LLM Routing.

Architecture:
  LOCAL (Ollama) → Offline fallback only
  CLOUD PRIMARY:
    - Anthropic: Claude Opus 4.6, Sonnet 4.6, Haiku 4.5
    - OpenAI: GPT-5, GPT-5 Mini, GPT-5 Nano, GPT-4.1, o3, o4-mini

Smart routing picks the best model based on:
  1. Task complexity (legal analysis vs simple query)
  2. Budget remaining
  3. Context length needs
  4. Reasoning depth required

Setup:
  pip install anthropic openai python-dotenv
  
  Set API keys in .env (NOT in settings.json):
    ANTHROPIC_API_KEY=sk-ant-...
    OPENAI_API_KEY=sk-proj-...
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Model Registry ──────────────────────────────────────────────────────────

MODELS = {
    # ── Anthropic ────────────────────────────────────────────────────────
    "claude-opus-4-6": {
        "provider": "anthropic",
        "name": "Claude Opus 4.6",
        "input": 15.0,      # per 1M tokens
        "output": 75.0,
        "context": 200000,
        "tier": "premium",
        "strengths": ["deep reasoning", "legal analysis", "long documents", "nuanced writing"],
        "speed": "slow",
    },
    "claude-sonnet-4-20250514": {
        "provider": "anthropic",
        "name": "Claude Sonnet 4",
        "input": 3.0,
        "output": 15.0,
        "context": 200000,
        "tier": "standard",
        "strengths": ["balanced", "coding", "analysis", "good value"],
        "speed": "medium",
    },
    "claude-haiku-4-5-20251001": {
        "provider": "anthropic",
        "name": "Claude Haiku 4.5",
        "input": 0.80,
        "output": 4.0,
        "context": 200000,
        "tier": "fast",
        "strengths": ["speed", "simple tasks", "routing", "cheap"],
        "speed": "fast",
    },
    # ── OpenAI GPT-5 Family ──────────────────────────────────────────────
    "gpt-5": {
        "provider": "openai",
        "name": "GPT-5",
        "input": 1.25,
        "output": 10.0,
        "context": 128000,
        "tier": "standard",
        "strengths": ["reasoning", "coding", "general purpose", "great value"],
        "speed": "medium",
    },
    "gpt-5-mini": {
        "provider": "openai",
        "name": "GPT-5 Mini",
        "input": 0.25,
        "output": 2.0,
        "context": 128000,
        "tier": "fast",
        "strengths": ["speed", "tool routing", "simple tasks", "very cheap"],
        "speed": "fast",
    },
    "gpt-5-nano": {
        "provider": "openai",
        "name": "GPT-5 Nano",
        "input": 0.05,
        "output": 0.40,
        "context": 128000,
        "tier": "ultra-fast",
        "strengths": ["classification", "intent detection", "cheapest"],
        "speed": "fastest",
    },
    # ── OpenAI GPT-4.1 Family ────────────────────────────────────────────
    "gpt-4.1": {
        "provider": "openai",
        "name": "GPT-4.1",
        "input": 2.0,
        "output": 8.0,
        "context": 1000000,  # 1M context!
        "tier": "standard",
        "strengths": ["huge context", "document analysis", "agentic"],
        "speed": "medium",
    },
    "gpt-4.1-mini": {
        "provider": "openai",
        "name": "GPT-4.1 Mini",
        "input": 0.40,
        "output": 1.60,
        "context": 1000000,
        "tier": "fast",
        "strengths": ["huge context", "cheap", "fine-tunable"],
        "speed": "fast",
    },
    "gpt-4.1-nano": {
        "provider": "openai",
        "name": "GPT-4.1 Nano",
        "input": 0.10,
        "output": 0.40,
        "context": 1000000,
        "tier": "ultra-fast",
        "strengths": ["huge context", "cheapest with big context"],
        "speed": "fastest",
    },
    # ── OpenAI Reasoning Models ──────────────────────────────────────────
    "o3": {
        "provider": "openai",
        "name": "o3 (Reasoning)",
        "input": 2.0,
        "output": 8.0,
        "context": 200000,
        "tier": "reasoning",
        "strengths": ["chain-of-thought", "math", "logic", "multi-step legal reasoning"],
        "speed": "slow",
    },
    "o4-mini": {
        "provider": "openai",
        "name": "o4-mini (Fast Reasoning)",
        "input": 1.10,
        "output": 4.40,
        "context": 200000,
        "tier": "reasoning",
        "strengths": ["reasoning", "faster than o3", "coding", "visual tasks"],
        "speed": "medium",
    },
    "o3-mini": {
        "provider": "openai",
        "name": "o3-mini (Budget Reasoning)",
        "input": 1.10,
        "output": 4.40,
        "context": 200000,
        "tier": "reasoning",
        "strengths": ["reasoning on a budget", "routing", "filtering"],
        "speed": "medium",
    },
    # ── Legacy (still available) ─────────────────────────────────────────
    "gpt-4o": {
        "provider": "openai",
        "name": "GPT-4o (Legacy)",
        "input": 2.50,
        "output": 10.0,
        "context": 128000,
        "tier": "legacy",
        "strengths": ["stable", "well-tested"],
        "speed": "medium",
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "name": "GPT-4o Mini (Legacy)",
        "input": 0.15,
        "output": 0.60,
        "context": 128000,
        "tier": "legacy",
        "strengths": ["very cheap", "fine-tunable", "stable"],
        "speed": "fast",
    },
}

# Model routing presets
ROUTE_PRESETS = {
    "legal_analysis": ["claude-sonnet-4-20250514", "gpt-5", "o3"],
    "legal_premium": ["claude-opus-4-6"],
    "quick_task": ["gpt-5-mini", "claude-haiku-4-5-20251001", "gpt-5-nano"],
    "tool_routing": ["gpt-5-nano", "gpt-5-mini"],
    "document_writing": ["claude-sonnet-4-20250514", "claude-opus-4-6", "gpt-5"],
    "reasoning": ["o3", "o4-mini", "claude-opus-4-6"],
    "long_document": ["gpt-4.1", "gpt-4.1-mini"],
    "fine_tuned": [],  # populated at runtime if fine-tuned models exist
}


class CloudReasoner:
    """Multi-provider cloud LLM router with smart model selection."""
    
    def __init__(self, config: dict):
        self.config = config
        self.cloud_cfg = config.get("cloud", {})
        self.enabled = self.cloud_cfg.get("enabled", False)
        self.provider = self.cloud_cfg.get("provider", "anthropic")
        self.auto_route = self.cloud_cfg.get("auto_route", True)
        self.max_budget = self.cloud_cfg.get("max_monthly_budget", 40.0)
        
        # Default models per provider
        self.default_anthropic = self.cloud_cfg.get("anthropic_model", "claude-sonnet-4-20250514")
        self.default_openai = self.cloud_cfg.get("openai_model", "gpt-5")
        
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
                key = self.cloud_cfg.get("anthropic_api_key", "") or os.environ.get("ANTHROPIC_API_KEY", "")
                if key:
                    self._anthropic = anthropic.Anthropic(api_key=key)
                else:
                    print("  [Cloud] No Anthropic API key — set ANTHROPIC_API_KEY in .env", flush=True)
            except ImportError:
                print("  [Cloud] pip install anthropic", flush=True)
        return self._anthropic
    
    @property
    def openai_client(self):
        if self._openai is None:
            try:
                from openai import OpenAI
                key = self.cloud_cfg.get("openai_api_key", "") or os.environ.get("OPENAI_API_KEY", "")
                if key:
                    self._openai = OpenAI(api_key=key)
                else:
                    print("  [Cloud] No OpenAI API key — set OPENAI_API_KEY in .env", flush=True)
            except ImportError:
                print("  [Cloud] pip install openai", flush=True)
        return self._openai
    
    # ── Spending Tracker ────────────────────────────────────────────────
    
    def _load_spend(self):
        if self.spend_file.exists():
            with open(self.spend_file) as f:
                data = json.load(f)
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
    
    def _track_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        model_info = MODELS.get(model, {})
        input_rate = model_info.get("input", 5.0)
        output_rate = model_info.get("output", 15.0)
        cost = (input_tokens / 1_000_000 * input_rate + 
                output_tokens / 1_000_000 * output_rate)
        self.monthly_spend += cost
        self._save_spend()
        remaining = max(0, self.max_budget - self.monthly_spend)
        print(f"  [Cloud] Cost: ${cost:.4f} | Monthly: ${self.monthly_spend:.2f} / ${self.max_budget:.2f} (${remaining:.2f} left)", flush=True)
        return cost
    
    def can_afford(self, model: str = None) -> bool:
        if self.monthly_spend >= self.max_budget:
            return False
        if model and model in MODELS:
            # Estimate: can we afford at least one query?
            info = MODELS[model]
            min_cost = (2000 / 1_000_000 * info["input"] + 1000 / 1_000_000 * info["output"])
            return (self.max_budget - self.monthly_spend) > min_cost
        return True
    
    # ── Smart Model Selection ───────────────────────────────────────────
    
    def select_model(self, query: str, intent: str, mode: str) -> str:
        """
        Pick the best model based on task, budget, and available providers.
        Returns model ID string.
        """
        lower = query.lower()
        
        # Explicit model request
        for model_id, info in MODELS.items():
            if model_id in lower or info["name"].lower() in lower:
                if self.can_afford(model_id):
                    return model_id
        
        # Explicit provider request
        if any(p in lower for p in ["use claude", "use anthropic"]):
            return self.default_anthropic
        if any(p in lower for p in ["use gpt", "use openai", "use o3", "use reasoning"]):
            if "o3" in lower or "reasoning" in lower:
                return "o3"
            return self.default_openai
        
        # Premium legal document generation
        if any(p in lower for p in ["write a brief", "write a memo", "legal memorandum",
                                     "draft a motion", "final document", "full power"]):
            if self.can_afford("claude-opus-4-6"):
                return "claude-opus-4-6"
            return "claude-sonnet-4-20250514"
        
        # Complex reasoning (multi-step legal logic)
        if any(p in lower for p in ["analyze this case", "build arguments", "both sides",
                                     "evaluate the strength", "compare these cases",
                                     "what are my options", "step by step"]):
            if self.can_afford("o3"):
                return "o3"
            if self.can_afford("gpt-5"):
                return "gpt-5"
            return "claude-sonnet-4-20250514"
        
        # Legal analysis (standard)
        if intent == "legal" and mode in ("analysis", "argument", "writing"):
            # Rotate between providers based on budget
            budget_pct = self.monthly_spend / self.max_budget if self.max_budget > 0 else 1
            if budget_pct < 0.5:
                return "claude-sonnet-4-20250514"  # Best legal quality
            elif budget_pct < 0.8:
                return "gpt-5"  # Cheaper but still strong
            else:
                return "gpt-5-mini"  # Budget mode
        
        # Research mode (raw sources)
        if mode == "research":
            return "gpt-5-mini"  # Don't need heavy reasoning for fetching
        
        # Simple tasks
        if intent in ("general", "technical"):
            return "gpt-5-mini"
        
        # Default: balanced choice
        return self.default_openai if self.provider == "openai" else self.default_anthropic
    
    def should_use_cloud(self, query: str, intent: str, mode: str) -> bool:
        """Decide if cloud should handle this query."""
        if not self.enabled:
            return False
        if not self.can_afford():
            print(f"  [Cloud] Budget exceeded (${self.monthly_spend:.2f} / ${self.max_budget:.2f})", flush=True)
            return False
        
        lower = query.lower()
        
        # Never route file operations to cloud
        file_words = ["download", "list files", "search files", "ingest", "folder",
                      "directory", "copy file", "move file", "delete file", "open file",
                      "list_directory", "disk_usage"]
        if any(w in lower for w in file_words):
            return False
        
        # Explicit cloud request
        if any(p in lower for p in ["use cloud", "use claude", "use gpt", "cloud mode",
                                     "better model", "full power", "deep analysis",
                                     "use o3", "use reasoning", "use openai"]):
            return True
        
        # Auto-route enabled — send most things to cloud
        if self.auto_route:
            # Legal queries always go cloud
            if intent in ("legal", "legal_adjacent"):
                return True
            # Document generation
            if any(p in lower for p in ["write", "draft", "create a document", "generate",
                                         "memorandum", "essay", "brief", "report"]):
                return True
            # Complex queries
            if any(p in lower for p in ["analyze", "compare", "explain", "research",
                                         "what are", "how does", "why did"]):
                return True
            # Simple stuff stays local
            return False
        
        return False
    
    # ── Query Execution ─────────────────────────────────────────────────
    
    def query(self, messages: list, system_prompt: str,
              tool_results: str = "", model: str = None) -> Optional[str]:
        """Route query to the appropriate provider and model."""
        if model is None:
            model = self.default_anthropic if self.provider == "anthropic" else self.default_openai
        
        model_info = MODELS.get(model, {})
        provider = model_info.get("provider", self.provider)
        
        print(f"  [Cloud] Model: {model} ({model_info.get('name', '?')}) via {provider}", flush=True)
        
        if provider == "anthropic":
            return self._query_anthropic(messages, system_prompt, tool_results, model)
        elif provider == "openai":
            return self._query_openai(messages, system_prompt, tool_results, model)
        else:
            print(f"  [Cloud] Unknown provider: {provider}", flush=True)
            return None
    
    def _query_anthropic(self, messages: list, system_prompt: str,
                          tool_results: str = "", model: str = "claude-sonnet-4-20250514") -> Optional[str]:
        client = self.anthropic_client
        if not client:
            return None
        
        # Build message list
        api_messages = []
        for msg in messages:
            if msg["role"] == "user":
                api_messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                api_messages.append({"role": "assistant", "content": msg["content"]})
            elif msg["role"] == "tool":
                api_messages.append({"role": "user", "content": f"[Tool Result]\n{msg['content']}"})
        
        if tool_results:
            if api_messages and api_messages[-1]["role"] == "user":
                api_messages[-1]["content"] += f"\n\n[Research Data from Local Tools]\n{tool_results}"
            else:
                api_messages.append({"role": "user", "content": f"[Research Data]\n{tool_results}"})
        
        # Ensure alternating roles
        cleaned = []
        for msg in api_messages:
            if cleaned and cleaned[-1]["role"] == msg["role"]:
                cleaned[-1]["content"] += "\n\n" + msg["content"]
            else:
                cleaned.append(msg)
        if cleaned and cleaned[0]["role"] != "user":
            cleaned.insert(0, {"role": "user", "content": "(continuing conversation)"})
        
        try:
            start = time.time()
            response = client.messages.create(
                model=model,
                max_tokens=8192,
                system=system_prompt,
                messages=cleaned,
            )
            elapsed = time.time() - start
            
            result = ""
            for block in response.content:
                if hasattr(block, "text"):
                    result += block.text
            
            cost = self._track_cost(model, response.usage.input_tokens, response.usage.output_tokens)
            print(f"  [Cloud] {len(result)} chars in {elapsed:.1f}s "
                  f"({response.usage.input_tokens} in / {response.usage.output_tokens} out)", flush=True)
            
            return result
        except Exception as e:
            print(f"  [Cloud] Anthropic error: {e}", flush=True)
            return None
    
    def _query_openai(self, messages: list, system_prompt: str,
                       tool_results: str = "", model: str = "gpt-5") -> Optional[str]:
        client = self.openai_client
        if not client:
            return None
        
        # Build message list
        api_messages = [{"role": "system", "content": system_prompt}]
        
        for msg in messages:
            if msg["role"] in ("user", "assistant"):
                api_messages.append({"role": msg["role"], "content": msg["content"]})
            elif msg["role"] == "tool":
                api_messages.append({"role": "user", "content": f"[Tool Result]\n{msg['content']}"})
        
        if tool_results:
            api_messages.append({"role": "user", "content": f"[Research Data]\n{tool_results}"})
        
        # Reasoning models (o3, o4-mini) don't support system messages the same way
        is_reasoning = model.startswith("o3") or model.startswith("o4")
        if is_reasoning:
            # Merge system prompt into first user message
            system_content = api_messages.pop(0)["content"]
            if api_messages and api_messages[0]["role"] == "user":
                api_messages[0]["content"] = f"[Instructions]\n{system_content}\n\n[Query]\n{api_messages[0]['content']}"
            else:
                api_messages.insert(0, {"role": "user", "content": system_content})
        
        try:
            start = time.time()
            
            kwargs = {
                "model": model,
                "messages": api_messages,
                "max_tokens": 8192,
            }
            # Reasoning models don't support temperature
            if not is_reasoning:
                kwargs["temperature"] = 0.3
            
            response = client.chat.completions.create(**kwargs)
            elapsed = time.time() - start
            
            result = response.choices[0].message.content
            cost = self._track_cost(model, response.usage.prompt_tokens, response.usage.completion_tokens)
            
            print(f"  [Cloud] {len(result)} chars in {elapsed:.1f}s "
                  f"({response.usage.prompt_tokens} in / {response.usage.completion_tokens} out)", flush=True)
            
            return result
        except Exception as e:
            print(f"  [Cloud] OpenAI error: {e}", flush=True)
            return None
    
    # ── Model Info for Dashboard ────────────────────────────────────────
    
    def get_available_models(self) -> list[dict]:
        """Return all models with availability status for the dashboard."""
        has_anthropic = bool(self.anthropic_client)
        has_openai = bool(self.openai_client)
        
        models = []
        for model_id, info in MODELS.items():
            available = (info["provider"] == "anthropic" and has_anthropic) or \
                       (info["provider"] == "openai" and has_openai)
            affordable = self.can_afford(model_id)
            
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
                "available": available,
                "affordable": affordable,
                "is_default": model_id in (self.default_anthropic, self.default_openai),
            })
        
        return models
    
    def get_status(self) -> str:
        if not self.enabled:
            return "Cloud reasoning: DISABLED"
        
        remaining = max(0, self.max_budget - self.monthly_spend)
        has_a = "✓" if self.anthropic_client else "✗"
        has_o = "✓" if self.openai_client else "✗"
        
        return (
            f"Cloud reasoning: ENABLED\n"
            f"  Anthropic: {has_a} | Default: {self.default_anthropic}\n"
            f"  OpenAI:    {has_o} | Default: {self.default_openai}\n"
            f"  Auto-route: {'ON' if self.auto_route else 'OFF'}\n"
            f"  Monthly: ${self.monthly_spend:.2f} / ${self.max_budget:.2f} (${remaining:.2f} left)\n"
            f"  Models available: {sum(1 for m in MODELS if MODELS[m]['provider'] == 'anthropic')} Anthropic, "
            f"{sum(1 for m in MODELS if MODELS[m]['provider'] == 'openai')} OpenAI"
        )
