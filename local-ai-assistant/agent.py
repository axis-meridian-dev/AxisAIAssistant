"""
Core Agent — LLM reasoning + tool dispatch loop.

Uses Ollama's tool-calling API to let the LLM decide which tools to invoke.
Runs a loop: LLM → tool call → result → LLM → ... until final answer.

Architecture layers:
  1. Intent Detection — classify user input before LLM sees it
  2. Tool Gate — safety tiers control which tools require confirmation
  3. Mode System — stabilizes behavior per task type
  4. Forced Retrieval — legal/research queries must hit tools before reasoning
  5. Citation Validation — post-processing check on legal responses
"""

import re
import json
import ollama
from rich.console import Console
from rich.table import Table

from tools.file_manager import FileManagerTool
from tools.web_search import WebSearchTool
from tools.desktop_control import DesktopControlTool
from tools.system_info import SystemInfoTool
from tools.knowledge_base import KnowledgeBaseTool
from tools.legal_research import LegalResearchTool
from tools.document_writer import DocumentWriterTool

console = Console()


# ── Intent Detection ───────────────────────────────────────────────────────

LEGAL_KEYWORDS = [
    "statute", "statutes", "law", "legal", "case law", "case", "court",
    "ruling", "precedent", "plaintiff", "defendant", "amendment", "constitutional",
    "rights", "civil rights", "section 1983", "1983", "usc", "u.s.c.",
    "qualified immunity", "excessive force", "fourth amendment", "due process",
    "search and seizure", "miranda", "habeas", "brief", "motion",
    "jurisdiction", "appeal", "certiorari", "injunction", "tort",
    "liability", "damages", "negligence", "malpractice", "prosecution",
    "indictment", "arraignment", "sentencing", "probation", "parole",
]

RESEARCH_KEYWORDS = [
    "research", "find", "look up", "search for", "statistics",
    "data on", "studies", "evidence", "sources", "articles about",
]


def detect_intent(user_input: str) -> str:
    """
    Classify user intent to determine routing behavior.
    Returns: 'legal', 'research', 'general'
    """
    lower = user_input.lower()

    # Legal intent — strongest signal
    if any(kw in lower for kw in LEGAL_KEYWORDS):
        return "legal"

    # Research intent
    if any(kw in lower for kw in RESEARCH_KEYWORDS):
        return "research"

    return "general"


# ── Tool Safety Tiers ──────────────────────────────────────────────────────

TOOL_TIERS = {
    # Tier 1: SAFE — no confirmation needed
    "safe": {
        "web_search", "fetch_webpage",
        "query_knowledge", "knowledge_stats", "list_sources",
        "lookup_statute", "search_case_law", "fetch_court_opinion",
        "search_legal_statistics", "search_legal_news", "compare_cases",
        "list_civil_rights_statutes", "list_research_files", "list_documents",
        "list_directory", "read_file", "file_info", "disk_usage",
        "search_files",
    },
    # Tier 2: WRITE — allowed but logged
    "write": {
        "write_file", "make_directory", "copy_file",
        "ingest_directory", "ingest_file", "remove_source",
        "write_document", "write_debate_prep", "append_to_document",
        "clip_article", "download_resource", "download_statute_collection",
        "generate_research_brief",
    },
    # Tier 3: SYSTEM — requires confirmation
    "system": {
        "move_file", "delete_file",
        "launch_app", "run_command", "close_window",
        "send_keys", "type_text", "clipboard_write",
        "screenshot", "focus_window", "list_windows",
        "get_active_window", "clipboard_read",
        "system_stats", "list_processes", "network_info",
    },
}


def get_tool_tier(tool_name: str) -> str:
    """Return the safety tier for a tool."""
    for tier, tools in TOOL_TIERS.items():
        if tool_name in tools:
            return tier
    return "system"  # Unknown tools default to highest restriction


# ── Agent Modes ────────────────────────────────────────────────────────────

VALID_MODES = {"research", "analysis", "argument", "write", "general"}

MODE_INSTRUCTIONS = {
    "research": (
        "\n[MODE: RESEARCH] Return raw sources only. No interpretation. "
        "List statutes, cases, and data with full citations. "
        "Use query_knowledge, lookup_statute, search_case_law, and search_legal_news."
    ),
    "analysis": (
        "\n[MODE: ANALYSIS] Apply law to the specific scenario. "
        "Every claim must be cited to a statute or case. "
        "Structure: APPLICABLE LAW → CASE LAW → APPLICATION → SOURCE FILES."
    ),
    "argument": (
        "\n[MODE: ARGUMENT] Build BOTH plaintiff AND defense arguments. "
        "Cite authority for each side. Identify the strongest and weakest points."
    ),
    "write": (
        "\n[MODE: WRITE] Produce a polished document (essay, brief, article, memo). "
        "Must include a Sources section with full citations at the end."
    ),
    "general": "",
}


# ── Citation Validation ────────────────────────────────────────────────────

def validate_legal_response(response_text: str, intent: str) -> str:
    """
    Post-processing check: if the intent was legal, verify the response
    contains at least one citation. If not, append a warning.
    """
    if intent != "legal":
        return response_text

    if not response_text or len(response_text.strip()) < 50:
        return response_text

    # Check for citation patterns
    has_statute = bool(re.search(
        r'\d+\s*U\.?S\.?C\.?\s*§?\s*\d+|\bSection\s+\d+|CT Gen Stat', response_text
    ))
    has_case = bool(re.search(
        r'[A-Z][a-z]+\s+v\.\s+[A-Z][a-z]+|\d+\s+(?:U\.S\.|S\.Ct\.|F\.\d+)', response_text
    ))
    has_disclaimer = "not legal advice" in response_text.lower()

    warnings = []
    if not has_statute and not has_case:
        warnings.append(
            "⚠ CITATION GAP: This response contains no verifiable legal citations. "
            "Results may be unreliable. Use lookup_statute or search_case_law to ground claims."
        )
    if not has_disclaimer:
        warnings.append(
            "Note: This is AI-generated legal research, not legal advice. "
            "Verify all citations independently."
        )

    if warnings:
        return response_text + "\n\n---\n" + "\n".join(warnings)

    return response_text

SYSTEM_PROMPT = """You are a powerful local AI assistant running directly on the user's Linux computer.
You have full access to the local filesystem, web search, desktop control, and a local knowledge base.

IMPORTANT RULES:
- You are running LOCALLY — all data stays on this machine. Be direct and helpful.
- When the user asks to do something (open files, search, manage apps), USE YOUR TOOLS. Don't just describe what you would do.
- For file operations, always use absolute paths. Expand ~ to the actual home directory.
- For destructive operations (delete, move, overwrite), confirm with the user first by stating what you plan to do.
- When searching the web, synthesize results into a clear answer — don't just dump raw results.
- For desktop control, describe what you're doing as you do it.
- Be concise. No filler. The user is technical.

KNOWLEDGE BASE:
- You have a local vector database of the user's files. Use query_knowledge to search it.
- When the user asks about their projects, code, notes, or research, ALWAYS query the knowledge base first.
- The user can ingest directories or files to grow the knowledge base.
- The knowledge base persists between sessions — ingested files stay indexed.

You have these tool categories:
1. FILE MANAGEMENT — read, write, search, organize, list files and directories
2. WEB SEARCH — search the internet and fetch web pages
3. DESKTOP CONTROL — launch apps, manage windows, clipboard, screenshots, run commands
4. SYSTEM INFO — hardware stats, running processes, disk usage, network info
5. KNOWLEDGE BASE — ingest files into vector DB, semantic search across all your documents
6. LEGAL RESEARCH — look up statutes, search case law, find statistics, clip news articles, generate research briefs
7. DOCUMENT WRITER — create essays, articles, debate prep docs, briefs, and reports from your research

═══════════════════════════════════════════════════════════
LEGAL RESEARCH RULES (MANDATORY — FOLLOW WITHOUT EXCEPTION)
═══════════════════════════════════════════════════════════

RULE 1: RETRIEVAL BEFORE REASONING
- For ANY legal question, ALWAYS retrieve sources FIRST using tools (lookup_statute, search_case_law, query_knowledge, search_legal_news).
- NEVER answer a legal question from memory alone. Always ground in retrieved sources.
- If no relevant legal sources are found, respond: "Insufficient legal authority found to support a conclusion. I recommend broadening the search or consulting additional sources."

RULE 2: MANDATORY CITATION FORMAT
- Every legal response MUST include citations in this structure:

  APPLICABLE LAW:
  - [Statute citation] — [Short description]

  CASE LAW:
  - [Case name], [Volume] [Reporter] [Page] ([Court] [Year]) — [Holding]

  APPLICATION:
  [Your analysis tying law + cases to the facts]

  SOURCE FILES:
  [Local file paths where the user can read the full text]

- If you cannot provide at least one statute OR one case citation, DO NOT provide legal analysis. State what's missing.

RULE 3: LEGAL MODES
The user can activate these modes explicitly, or you infer from context:
- "research mode" → Return raw sources only. No interpretation. Just statutes, cases, and data.
- "analysis mode" → Apply law to a specific scenario. Must cite every claim.
- "argument mode" → Build plaintiff AND defense arguments. Cite authority for each side.
- "writing mode" → Produce essays, briefs, articles. Must include a Sources section.
Default is analysis mode unless the user indicates otherwise.

RULE 4: NEWS-TO-LAW LINKING
When discussing news articles, ALWAYS connect them to:
- The specific law or statute at issue
- Any relevant case precedent
- The legal standard being applied

RULE 5: DISCLAIMER
End every legal response with: "Note: This is AI-generated legal research, not legal advice. Verify all citations independently."

GENERAL WORKFLOW:
- For case prep: generate_research_brief → look up statutes → search case law → find stats → write document
- Downloaded statutes/opinions go to ~/LegalResearch/ — ingest them into the knowledge base
- For news monitoring: search_legal_news → clip_article → auto-tag → link to legal authority

Think step by step about which tools to use, then use them."""


class Agent:
    def __init__(self, config: dict):
        self.config = config
        self.history: list[dict] = []
        self.ollama_host = config["llm"]["ollama_host"]
        self.mode = "general"  # Current agent mode

        # Initialize tools
        self.tool_instances = {
            "file_manager": FileManagerTool(config),
            "web_search": WebSearchTool(config),
            # "desktop_control": DesktopControlTool(config),
            # "system_info": SystemInfoTool(config),
            "knowledge_base": KnowledgeBaseTool(config),
            "legal_research": LegalResearchTool(config),
            "document_writer": DocumentWriterTool(config),
        }

        # Build tool definitions for Ollama
        self.tools = []
        for instance in self.tool_instances.values():
            self.tools.extend(instance.get_tool_definitions())

        # Map tool names to handler functions
        self.tool_handlers = {}
        for instance in self.tool_instances.values():
            self.tool_handlers.update(instance.get_handlers())

    def list_tools(self) -> list[str]:
        return list(self.tool_instances.keys())

    def print_tools(self):
        table = Table(title="Available Tools", border_style="cyan")
        table.add_column("Category", style="bold")
        table.add_column("Functions", style="dim")
        table.add_column("Tier", style="dim")

        for name, instance in self.tool_instances.items():
            for defn in instance.get_tool_definitions():
                fname = defn["function"]["name"]
                tier = get_tool_tier(fname)
                table.add_row(name, fname, tier)

        console.print(table)

    def clear_history(self):
        self.history = []

    def set_mode(self, mode: str) -> str:
        """Set the agent's operating mode."""
        mode = mode.lower().strip()
        if mode not in VALID_MODES:
            return f"Invalid mode '{mode}'. Valid modes: {', '.join(sorted(VALID_MODES))}"
        self.mode = mode
        return f"Mode set to: {mode}"

    def _force_retrieval(self, user_input: str, intent: str, messages: list) -> list:
        """
        For legal/research intents, inject tool results BEFORE the LLM reasons.
        This ensures retrieval-before-reasoning is enforced programmatically,
        not just via prompt instructions.
        """
        if intent not in ("legal", "research"):
            return messages

        print("  [Forced Retrieval] Intent requires grounding — searching knowledge base...", flush=True)

        # Step 1: Query the knowledge base
        kb = self.tool_instances.get("knowledge_base")
        if kb:
            try:
                # For legal intent, use legal-specific filters and reranking
                if intent == "legal":
                    kb_result = kb.query_knowledge(
                        query=user_input,
                        max_results=5,
                        filter_doc_type=None,  # Don't over-filter — let vector search decide
                        rerank=True,
                    )
                else:
                    kb_result = kb.query_knowledge(query=user_input, max_results=5)

                if kb_result and "No relevant results" not in kb_result and "empty" not in kb_result.lower():
                    print(f"  [Forced Retrieval] Knowledge base returned results ({len(kb_result)} chars)", flush=True)
                    messages.append({
                        "role": "system",
                        "content": (
                            f"[RETRIEVAL CONTEXT — grounding data from knowledge base]\n"
                            f"The following sources were retrieved BEFORE reasoning. "
                            f"You MUST use these to ground your response. "
                            f"Do NOT ignore retrieved sources.\n\n{kb_result}"
                        )
                    })
                else:
                    print("  [Forced Retrieval] Knowledge base had no relevant results.", flush=True)
                    messages.append({
                        "role": "system",
                        "content": (
                            "[RETRIEVAL CONTEXT] No relevant documents found in local knowledge base. "
                            "You MUST use lookup_statute, search_case_law, or web_search to find sources. "
                            "Do NOT answer from memory alone."
                        )
                    })
            except Exception as e:
                print(f"  [Forced Retrieval] Knowledge base error: {e}", flush=True)

        return messages

    def _gate_tool_call(self, func_name: str) -> tuple[bool, str]:
        """
        Check if a tool call is allowed based on its safety tier.
        Returns (allowed, message).
        """
        tier = get_tool_tier(func_name)

        if tier == "safe":
            return True, ""
        elif tier == "write":
            return True, f"[write-tier tool: {func_name}]"
        elif tier == "system":
            console.print(
                f"  [GATE] System-tier tool requested: {func_name}",
                style="bold yellow"
            )
            # In non-interactive contexts, block system tools by default.
            # In interactive mode, this is where confirmation would go.
            # For now, allow but warn prominently.
            return True, f"[SYSTEM-TIER WARNING] Executing privileged tool: {func_name}"

        return True, ""

    async def process(self, user_input: str) -> str:
        """Process user input through the agent loop."""

        # ── Step 0: Mode detection from input ──────────────────────────
        lower_input = user_input.lower()
        if "research mode" in lower_input:
            self.set_mode("research")
        elif "analysis mode" in lower_input:
            self.set_mode("analysis")
        elif "argument mode" in lower_input:
            self.set_mode("argument")
        elif "writing mode" in lower_input or "write mode" in lower_input:
            self.set_mode("write")

        # ── Step 1: Intent detection ───────────────────────────────────
        intent = detect_intent(user_input)
        if intent != "general":
            print(f"  [Intent: {intent}] [Mode: {self.mode}]", flush=True)

        self.history.append({"role": "user", "content": user_input})

        # Determine which model to use
        model = self._select_model(user_input)
        print(f"  [Model: {model}]", flush=True)

        # Build message list with system prompt + mode instruction
        mode_instruction = MODE_INSTRUCTIONS.get(self.mode, "")
        system_content = SYSTEM_PROMPT + mode_instruction
        messages = [{"role": "system", "content": system_content}] + self.history

        # ── Step 2: Forced retrieval for legal/research intents ────────
        messages = self._force_retrieval(user_input, intent, messages)

        # ── Step 3: Agent loop ─────────────────────────────────────────
        max_iterations = 10
        for i in range(max_iterations):
            print(f"  [Step {i+1}] Sending to LLM...", flush=True)

            try:
                response = ollama.chat(
                    model=model,
                    messages=messages,
                    tools=self.tools,
                    options={
                        "temperature": self.config["llm"]["temperature"],
                        "num_ctx": self.config["llm"]["context_window"],
                    }
                )
            except Exception as e:
                error_msg = f"LLM error: {e}"
                print(f"  [ERROR] {error_msg}", flush=True)
                return error_msg

            msg = response["message"]
            messages.append(msg)

            # Check if there are tool calls
            if msg.get("tool_calls"):
                print(f"  [Step {i+1}] LLM wants to call {len(msg['tool_calls'])} tool(s):", flush=True)

                for tool_call in msg["tool_calls"]:
                    func_name = tool_call["function"]["name"]
                    func_args = tool_call["function"]["arguments"]

                    # ── Tool gate check ────────────────────────────────
                    allowed, gate_msg = self._gate_tool_call(func_name)
                    if gate_msg:
                        print(f"    {gate_msg}", flush=True)
                    if not allowed:
                        messages.append({
                            "role": "tool",
                            "content": f"Tool '{func_name}' blocked by safety gate."
                        })
                        continue

                    # Show what's being called
                    args_preview = json.dumps(func_args, default=str)
                    if len(args_preview) > 150:
                        args_preview = args_preview[:150] + "..."
                    print(f"    → {func_name}({args_preview})", flush=True)

                    # Execute the tool
                    handler = self.tool_handlers.get(func_name)
                    if handler:
                        try:
                            result = handler(**func_args)
                            # Show a preview of the result
                            result_str = str(result)
                            if len(result_str) > 200:
                                print(f"    ✓ Got result ({len(result_str)} chars)", flush=True)
                            else:
                                print(f"    ✓ {result_str}", flush=True)
                        except Exception as e:
                            result = f"Tool error: {e}"
                            print(f"    ✗ {result}", flush=True)
                    else:
                        result = f"Unknown tool: {func_name}"
                        print(f"    ✗ {result}", flush=True)

                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "content": str(result)
                    })
            else:
                # No tool calls — this is the final response
                final = msg.get("content", "")

                # ── Step 4: Citation validation for legal responses ────
                final = validate_legal_response(final, intent)

                print(f"  [Done] Got final response ({len(final)} chars)", flush=True)
                self.history.append({"role": "assistant", "content": final})

                # Trim history to prevent context overflow
                if len(self.history) > 40:
                    self.history = self.history[-30:]

                return final

        return "Reached maximum tool iterations. Please try a simpler request."

    def _select_model(self, user_input: str) -> str:
        """Always use primary model. Fast model routing disabled
        until a larger GPU can handle tool definitions properly."""
        return self.config["llm"]["primary_model"]