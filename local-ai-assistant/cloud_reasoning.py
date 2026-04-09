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
import logging
import os
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ai_assistant.cloud")

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


def check_network(timeout: float = 2.0) -> bool:
    """Check internet connectivity by attempting a socket connection.

    Uses DNS resolution + TCP connect to a reliable host.
    Cached for a short period to avoid repeated checks within a single query.
    """
    targets = [
        ("8.8.8.8", 53),        # Google DNS
        ("1.1.1.1", 53),        # Cloudflare DNS
        ("208.67.222.222", 53), # OpenDNS
    ]
    for host, port in targets:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((host, port))
            sock.close()
            return True
        except (OSError, socket.timeout):
            continue
    return False


class CloudReasoner:
    """Multi-provider cloud LLM router with smart model selection."""

    def __init__(self, config: dict):
        self.config = config
        self.cloud_cfg = config.get("cloud", {})
        self.enabled = self.cloud_cfg.get("enabled", False)
        self.provider = self.cloud_cfg.get("provider", "anthropic")
        self.auto_route = self.cloud_cfg.get("auto_route", True)
        self.cloud_first = self.cloud_cfg.get("cloud_first", True)
        self.max_budget = self.cloud_cfg.get("max_monthly_budget", 40.0)

        # Network state cache (avoid checking every call)
        self._network_online = None
        self._network_checked_at = 0.0
        self._network_cache_ttl = 30.0  # re-check every 30 seconds
        
        # Default models per provider
        self.default_anthropic = self.cloud_cfg.get("anthropic_model", "claude-sonnet-4-20250514")
        self.default_openai = self.cloud_cfg.get("openai_model", "gpt-5")
        
        # Per-provider balance tracking
        self.spend_file = Path.home() / ".local" / "share" / "ai-assistant" / "cloud_spend.json"
        self.spend_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_spend()

        # Lazy-load API clients
        self._anthropic = None
        self._openai = None

    @property
    def is_online(self) -> bool:
        """Check network connectivity with caching."""
        now = time.time()
        if self._network_online is None or (now - self._network_checked_at) > self._network_cache_ttl:
            self._network_online = check_network()
            self._network_checked_at = now
            status = "ONLINE" if self._network_online else "OFFLINE"
            logger.info("Network status: %s", status)
        return self._network_online

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
        """Load per-provider balance and spend data."""
        default_balances = self.cloud_cfg.get("balances", {})
        if self.spend_file.exists():
            with open(self.spend_file) as f:
                data = json.load(f)
            # Reset on new month
            if data.get("month") != datetime.now().strftime("%Y-%m"):
                self.monthly_spend = 0.0
                self.provider_spend = {"anthropic": 0.0, "openai": 0.0}
                self.provider_balances = {
                    "anthropic": default_balances.get("anthropic", 40.0),
                    "openai": default_balances.get("openai", 20.0),
                }
                self._save_spend()
            else:
                self.monthly_spend = data.get("spend", 0.0)
                self.provider_spend = data.get("provider_spend", {"anthropic": 0.0, "openai": 0.0})
                self.provider_balances = data.get("provider_balances", {
                    "anthropic": default_balances.get("anthropic", 40.0),
                    "openai": default_balances.get("openai", 20.0),
                })
        else:
            self.monthly_spend = 0.0
            self.provider_spend = {"anthropic": 0.0, "openai": 0.0}
            self.provider_balances = {
                "anthropic": default_balances.get("anthropic", 40.0),
                "openai": default_balances.get("openai", 20.0),
            }

    def _save_spend(self):
        with open(self.spend_file, "w") as f:
            json.dump({
                "month": datetime.now().strftime("%Y-%m"),
                "spend": round(self.monthly_spend, 4),
                "provider_spend": {k: round(v, 4) for k, v in self.provider_spend.items()},
                "provider_balances": {k: round(v, 4) for k, v in self.provider_balances.items()},
                "updated": datetime.now().isoformat(),
            }, f, indent=2)

    def _track_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        model_info = MODELS.get(model, {})
        provider = model_info.get("provider", "openai")
        input_rate = model_info.get("input", 5.0)
        output_rate = model_info.get("output", 15.0)
        cost = (input_tokens / 1_000_000 * input_rate +
                output_tokens / 1_000_000 * output_rate)

        # Update totals
        self.monthly_spend += cost
        self.provider_spend[provider] = self.provider_spend.get(provider, 0.0) + cost
        self.provider_balances[provider] = self.provider_balances.get(provider, 0.0) - cost
        self._save_spend()

        bal_a = self.provider_balances.get("anthropic", 0)
        bal_o = self.provider_balances.get("openai", 0)
        print(f"  [Cloud] Cost: ${cost:.4f} | Balances: Anthropic ${bal_a:.2f} · OpenAI ${bal_o:.2f}", flush=True)
        return cost

    def can_afford(self, model: str = None) -> bool:
        """Check if there's enough provider balance for at least one query."""
        if model and model in MODELS:
            info = MODELS[model]
            provider = info["provider"]
            balance = self.provider_balances.get(provider, 0.0)
            min_cost = (2000 / 1_000_000 * info["input"] + 1000 / 1_000_000 * info["output"])
            return balance > min_cost
        # General check: any provider has funds
        return any(b > 0.01 for b in self.provider_balances.values())

    def get_balances(self) -> dict[str, float]:
        """Return current per-provider balances."""
        return dict(self.provider_balances)
    
    # ── Smart Model Selection ───────────────────────────────────────────
    
    def _first_affordable(self, candidates: list[str]) -> str | None:
        """Return the first model from candidates that fits the budget."""
        for model_id in candidates:
            if self.can_afford(model_id):
                return model_id
        return None

    def select_model(self, query: str, intent: str, mode: str) -> str:
        """
        Pick the best model based on task, budget, and available providers.
        Cascades from preferred to cheaper models when budget is tight.
        Returns model ID string.
        """
        lower = query.lower()

        # Budget-aware fallback chain (expensive → cheap)
        ANTHROPIC_CASCADE = ["claude-opus-4-6", "claude-sonnet-4-20250514", "claude-haiku-4-5-20251001"]
        OPENAI_CASCADE = ["gpt-5", "gpt-5-mini", "gpt-5-nano"]
        REASONING_CASCADE = ["o3", "o4-mini", "o3-mini", "gpt-5"]
        FULL_CASCADE = ["claude-opus-4-6", "claude-sonnet-4-20250514", "gpt-5", "claude-haiku-4-5-20251001", "gpt-5-mini", "gpt-5-nano"]

        # Explicit model request
        for model_id, info in MODELS.items():
            if model_id in lower or info["name"].lower() in lower:
                if self.can_afford(model_id):
                    return model_id
                # Requested model too expensive — cascade within its provider
                provider_cascade = ANTHROPIC_CASCADE if info["provider"] == "anthropic" else OPENAI_CASCADE
                fallback = self._first_affordable(provider_cascade)
                if fallback:
                    return fallback

        # Explicit provider request
        if any(p in lower for p in ["use claude", "use anthropic"]):
            return self._first_affordable(ANTHROPIC_CASCADE) or self.default_anthropic
        if any(p in lower for p in ["use gpt", "use openai", "use o3", "use reasoning"]):
            if "o3" in lower or "reasoning" in lower:
                return self._first_affordable(REASONING_CASCADE) or "gpt-5-mini"
            return self._first_affordable(OPENAI_CASCADE) or self.default_openai

        # Premium legal document generation
        if any(p in lower for p in ["write a brief", "write a memo", "legal memorandum",
                                     "draft a motion", "final document", "full power"]):
            return self._first_affordable(["claude-opus-4-6", "claude-sonnet-4-20250514", "gpt-5", "gpt-5-mini"]) or "gpt-5-mini"

        # Complex reasoning (multi-step legal logic)
        if any(p in lower for p in ["analyze this case", "build arguments", "both sides",
                                     "evaluate the strength", "compare these cases",
                                     "what are my options", "step by step"]):
            return self._first_affordable(REASONING_CASCADE) or "gpt-5-mini"

        # Legal analysis (standard)
        if intent == "legal" and mode in ("analysis", "argument", "writing"):
            return self._first_affordable(["claude-sonnet-4-20250514", "gpt-5", "gpt-5-mini"]) or "gpt-5-mini"

        # Research mode (raw sources)
        if mode == "research":
            return self._first_affordable(["gpt-5-mini", "gpt-5-nano"]) or "gpt-5-mini"

        # Simple tasks
        if intent in ("general", "technical"):
            return self._first_affordable(["gpt-5-mini", "gpt-5-nano"]) or "gpt-5-mini"

        # Default: cascade through all models
        return self._first_affordable(FULL_CASCADE) or self.default_openai
    
    def should_use_cloud(self, query: str, intent: str, mode: str) -> bool:
        """Decide if cloud should handle this query.

        Routing priority (when cloud_first is enabled):
          Online  → cloud for everything except pure file/desktop ops
          Offline → local Ollama handles all queries
        """
        if not self.enabled:
            return False

        # ── Network check ────────────────────────────────────────────
        online = self.is_online
        if not online:
            print("  [Cloud] Network offline → routing to LOCAL", flush=True)
            return False

        if not self.can_afford():
            print(f"  [Cloud] Budget exceeded (${self.monthly_spend:.2f} / ${self.max_budget:.2f}) → LOCAL", flush=True)
            return False

        lower = query.lower()

        # Never route file/desktop operations to cloud
        file_words = ["download", "list files", "search files", "ingest", "folder",
                      "directory", "copy file", "move file", "delete file", "open file",
                      "list_directory", "disk_usage"]
        if any(w in lower for w in file_words):
            return False

        # Explicit cloud request — always honor
        if any(p in lower for p in ["use cloud", "use claude", "use gpt", "cloud mode",
                                     "better model", "full power", "deep analysis",
                                     "use o3", "use reasoning", "use openai"]):
            return True

        # Explicit local request — honor it
        if any(p in lower for p in ["use local", "use ollama", "offline mode"]):
            return False

        # ── Cloud-first mode: route everything to cloud when online ──
        if self.cloud_first:
            return True

        # ── Legacy auto-route (cloud_first disabled) ─────────────────
        if self.auto_route:
            if intent in ("legal", "legal_adjacent"):
                return True
            if any(p in lower for p in ["write", "draft", "create a document", "generate",
                                         "memorandum", "essay", "brief", "report"]):
                return True
            if any(p in lower for p in ["analyze", "compare", "explain", "research",
                                         "what are", "how does", "why did"]):
                return True
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
                          tool_results: str = "", model: str = "claude-sonnet-4-20250514",
                          max_retries: int = 3) -> Optional[str]:
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

        for attempt in range(max_retries):
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
                wait = 2 ** attempt
                logger.warning("Anthropic attempt %d/%d failed: %s", attempt + 1, max_retries, e)
                print(f"  [Cloud] Anthropic error (attempt {attempt + 1}/{max_retries}): {e}", flush=True)
                if attempt < max_retries - 1:
                    print(f"  [Cloud] Retrying in {wait}s...", flush=True)
                    time.sleep(wait)

        logger.error("Anthropic query failed after %d attempts", max_retries)
        return None
    
    def _query_openai(self, messages: list, system_prompt: str,
                       tool_results: str = "", model: str = "gpt-5",
                       max_retries: int = 3) -> Optional[str]:
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
            system_content = api_messages.pop(0)["content"]
            if api_messages and api_messages[0]["role"] == "user":
                api_messages[0]["content"] = f"[Instructions]\n{system_content}\n\n[Query]\n{api_messages[0]['content']}"
            else:
                api_messages.insert(0, {"role": "user", "content": system_content})

        for attempt in range(max_retries):
            try:
                start = time.time()

                kwargs = {
                    "model": model,
                    "messages": api_messages,
                    "max_tokens": 8192,
                }
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
                wait = 2 ** attempt
                logger.warning("OpenAI attempt %d/%d failed: %s", attempt + 1, max_retries, e)
                print(f"  [Cloud] OpenAI error (attempt {attempt + 1}/{max_retries}): {e}", flush=True)
                if attempt < max_retries - 1:
                    print(f"  [Cloud] Retrying in {wait}s...", flush=True)
                    time.sleep(wait)

        logger.error("OpenAI query failed after %d attempts", max_retries)
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

        has_a = "✓" if self.anthropic_client else "✗"
        has_o = "✓" if self.openai_client else "✗"
        bal_a = self.provider_balances.get("anthropic", 0)
        bal_o = self.provider_balances.get("openai", 0)
        spent_a = self.provider_spend.get("anthropic", 0)
        spent_o = self.provider_spend.get("openai", 0)
        net = "ONLINE" if self.is_online else "OFFLINE"

        return (
            f"Cloud reasoning: ENABLED\n"
            f"  Network: {net}\n"
            f"  Cloud-first: {'ON — cloud primary, local fallback' if self.cloud_first else 'OFF — selective routing'}\n"
            f"  Anthropic: {has_a} | Default: {self.default_anthropic}\n"
            f"    Balance: ${bal_a:.2f} remaining (${spent_a:.2f} spent)\n"
            f"  OpenAI:    {has_o} | Default: {self.default_openai}\n"
            f"    Balance: ${bal_o:.2f} remaining (${spent_o:.2f} spent)\n"
            f"  Auto-route: {'ON' if self.auto_route else 'OFF'}\n"
            f"  Total spent this month: ${self.monthly_spend:.2f}\n"
            f"  Models available: {sum(1 for m in MODELS if MODELS[m]['provider'] == 'anthropic')} Anthropic, "
            f"{sum(1 for m in MODELS if MODELS[m]['provider'] == 'openai')} OpenAI"
        )
