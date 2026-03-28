"""
Performance Stats & Chat History — tracks timing, tool usage, and session data.

Provides:
  - Per-inquiry timing (total, LLM, tool, retrieval phases)
  - Per-model stats (average response time, query count)
  - Tool usage tracking (calls, offline vs online data processed)
  - Chat history with session persistence
"""

import json
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict


# ── Inquiry Stats (per-query) ─────────────────────────────────────────────

@dataclass
class InquiryStats:
    """Stats for a single query through the agent pipeline."""
    query: str = ""
    model: str = ""
    intent: str = "general"
    mode: str = "general"
    timestamp: str = ""

    # Timing (seconds)
    total_time: float = 0.0
    llm_time: float = 0.0        # Time spent in ollama.chat calls
    tool_time: float = 0.0       # Time spent executing tools
    retrieval_time: float = 0.0  # Time in forced retrieval chain
    fact_extraction_time: float = 0.0

    # LLM iterations
    llm_calls: int = 0

    # Tool usage
    tools_called: list = field(default_factory=list)
    tool_call_count: int = 0

    # Data processed
    offline_data_chars: int = 0   # Data from local KB, files
    online_data_chars: int = 0    # Data from web search, APIs
    response_chars: int = 0

    # Validation
    validation_passed: bool = True
    retry_triggered: bool = False


class StatsTimer:
    """Context manager for timing code blocks."""

    def __init__(self):
        self._start = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start

    def start(self):
        self._start = time.perf_counter()

    def stop(self) -> float:
        self.elapsed = time.perf_counter() - self._start
        return self.elapsed

    def add(self, seconds: float):
        """Accumulate time from a sub-operation."""
        self.elapsed += seconds


# ── Session & History Tracker ──────────────────────────────────────────────

# Tools that pull data from online sources
ONLINE_TOOLS = {
    "web_search", "fetch_webpage", "search_case_law", "fetch_court_opinion",
    "search_legal_news", "search_legal_statistics", "lookup_statute",
    "download_resource", "download_statute_collection", "clip_article",
}

# Tools that pull data from local/offline sources
OFFLINE_TOOLS = {
    "query_knowledge", "knowledge_stats", "list_sources",
    "ingest_directory", "ingest_file", "remove_source",
    "read_file", "list_directory", "search_files", "file_info",
    "list_research_files", "list_documents", "list_civil_rights_statutes",
}


class SessionStats:
    """
    Tracks performance stats across a session and persists chat history.

    Data stored at: ~/.local/share/ai-assistant/stats/
    """

    def __init__(self, stats_dir: str = None):
        self.stats_dir = Path(stats_dir or Path.home() / ".local" / "share" / "ai-assistant" / "stats")
        self.stats_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start = datetime.now().isoformat()
        self.inquiries: list[InquiryStats] = []

        # Persistent model stats
        self.model_stats_path = self.stats_dir / "model_stats.json"
        self.model_stats = self._load_model_stats()

        # Chat history
        self.history_dir = self.stats_dir / "chat_history"
        self.history_dir.mkdir(parents=True, exist_ok=True)

    # ── Model Stats (persistent across sessions) ──────────────────────

    def _load_model_stats(self) -> dict:
        if self.model_stats_path.exists():
            try:
                with open(self.model_stats_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_model_stats(self):
        with open(self.model_stats_path, "w") as f:
            json.dump(self.model_stats, f, indent=2)

    def record_inquiry(self, stats: InquiryStats):
        """Record a completed inquiry and update all stats."""
        self.inquiries.append(stats)

        # Update per-model stats
        model = stats.model
        if model not in self.model_stats:
            self.model_stats[model] = {
                "total_queries": 0,
                "total_time": 0.0,
                "total_llm_time": 0.0,
                "total_tool_time": 0.0,
                "avg_time": 0.0,
                "avg_llm_time": 0.0,
                "fastest": float("inf"),
                "slowest": 0.0,
                "tool_calls": 0,
                "online_chars": 0,
                "offline_chars": 0,
            }

        ms = self.model_stats[model]
        ms["total_queries"] += 1
        ms["total_time"] += stats.total_time
        ms["total_llm_time"] += stats.llm_time
        ms["total_tool_time"] += stats.tool_time
        ms["avg_time"] = ms["total_time"] / ms["total_queries"]
        ms["avg_llm_time"] = ms["total_llm_time"] / ms["total_queries"]
        ms["fastest"] = min(ms["fastest"], stats.total_time)
        ms["slowest"] = max(ms["slowest"], stats.total_time)
        ms["tool_calls"] += stats.tool_call_count
        ms["online_chars"] += stats.online_data_chars
        ms["offline_chars"] += stats.offline_data_chars

        self._save_model_stats()

    # ── Chat History ──────────────────────────────────────────────────

    def save_chat_session(self, history: list[dict]):
        """Save the current chat session to disk."""
        session_file = self.history_dir / f"chat_{self.session_id}.json"
        session_data = {
            "session_id": self.session_id,
            "started_at": self.session_start,
            "saved_at": datetime.now().isoformat(),
            "messages": history,
            "inquiry_count": len(self.inquiries),
            "total_time": sum(s.total_time for s in self.inquiries),
            "inquiries": [asdict(s) for s in self.inquiries],
        }
        with open(session_file, "w") as f:
            json.dump(session_data, f, indent=2)

    def list_chat_sessions(self, limit: int = 10) -> list[dict]:
        """List recent chat sessions."""
        sessions = []
        for f in sorted(self.history_dir.glob("chat_*.json"), reverse=True)[:limit]:
            try:
                with open(f) as fh:
                    data = json.load(fh)
                sessions.append({
                    "file": f.name,
                    "session_id": data.get("session_id", "?"),
                    "started_at": data.get("started_at", "?"),
                    "messages": len(data.get("messages", [])),
                    "inquiries": data.get("inquiry_count", 0),
                    "total_time": data.get("total_time", 0),
                })
            except Exception:
                continue
        return sessions

    def load_chat_session(self, session_id: str) -> dict | None:
        """Load a specific chat session by ID."""
        session_file = self.history_dir / f"chat_{session_id}.json"
        if session_file.exists():
            with open(session_file) as f:
                return json.load(f)
        # Try partial match
        for f in self.history_dir.glob(f"chat_*{session_id}*.json"):
            with open(f) as fh:
                return json.load(fh)
        return None

    # ── Display Helpers ───────────────────────────────────────────────

    def format_inquiry_stats(self, stats: InquiryStats) -> str:
        """Format a single inquiry's stats for display."""
        lines = [
            f"  Total:     {stats.total_time:.2f}s",
            f"  LLM:       {stats.llm_time:.2f}s ({stats.llm_calls} call{'s' if stats.llm_calls != 1 else ''})",
            f"  Tools:     {stats.tool_time:.2f}s ({stats.tool_call_count} call{'s' if stats.tool_call_count != 1 else ''})",
        ]
        if stats.retrieval_time > 0:
            lines.append(f"  Retrieval: {stats.retrieval_time:.2f}s")
        if stats.fact_extraction_time > 0:
            lines.append(f"  Facts:     {stats.fact_extraction_time:.2f}s")

        if stats.tools_called:
            tool_counts = {}
            for t in stats.tools_called:
                tool_counts[t] = tool_counts.get(t, 0) + 1
            tool_summary = ", ".join(f"{t}({n})" if n > 1 else t for t, n in tool_counts.items())
            lines.append(f"  Used:      {tool_summary}")

        if stats.offline_data_chars > 0 or stats.online_data_chars > 0:
            offline_kb = stats.offline_data_chars / 1024
            online_kb = stats.online_data_chars / 1024
            lines.append(f"  Data:      {offline_kb:.1f}KB offline, {online_kb:.1f}KB online")

        return "\n".join(lines)

    def format_session_summary(self) -> str:
        """Format current session summary."""
        if not self.inquiries:
            return "No queries in this session yet."

        total = sum(s.total_time for s in self.inquiries)
        avg = total / len(self.inquiries)
        total_tools = sum(s.tool_call_count for s in self.inquiries)
        total_offline = sum(s.offline_data_chars for s in self.inquiries)
        total_online = sum(s.online_data_chars for s in self.inquiries)

        lines = [
            f"Session: {self.session_id}",
            f"  Queries:       {len(self.inquiries)}",
            f"  Total time:    {total:.2f}s",
            f"  Avg per query: {avg:.2f}s",
            f"  Tool calls:    {total_tools}",
            f"  Offline data:  {total_offline / 1024:.1f}KB",
            f"  Online data:   {total_online / 1024:.1f}KB",
        ]
        return "\n".join(lines)

    def format_model_stats(self) -> str:
        """Format persistent per-model stats."""
        if not self.model_stats:
            return "No model stats recorded yet."

        lines = ["Model Performance (all sessions):\n"]
        for model, ms in sorted(self.model_stats.items()):
            fastest = ms["fastest"] if ms["fastest"] != float("inf") else 0
            lines.extend([
                f"  {model}:",
                f"    Queries:     {ms['total_queries']}",
                f"    Avg time:    {ms['avg_time']:.2f}s",
                f"    Avg LLM:     {ms['avg_llm_time']:.2f}s",
                f"    Fastest:     {fastest:.2f}s",
                f"    Slowest:     {ms['slowest']:.2f}s",
                f"    Tool calls:  {ms['tool_calls']}",
                f"    Offline:     {ms['offline_chars'] / 1024:.1f}KB",
                f"    Online:      {ms['online_chars'] / 1024:.1f}KB",
                "",
            ])
        return "\n".join(lines)

    @staticmethod
    def classify_tool_data(tool_name: str) -> str:
        """Classify whether a tool produces online or offline data."""
        if tool_name in ONLINE_TOOLS:
            return "online"
        if tool_name in OFFLINE_TOOLS:
            return "offline"
        return "offline"  # Default to offline for unknown tools
