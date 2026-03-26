"""
Core Agent — LLM reasoning + tool dispatch loop.

Uses Ollama's tool-calling API to let the LLM decide which tools to invoke.
Runs a loop: LLM → tool call → result → LLM → ... until final answer.

Architecture layers:
  1. Intent Detection — classify user input before LLM sees it
  2. Tool Gate — safety tiers control which tools require confirmation
  3. Mode System — stabilizes behavior per task type
  4. Forced Retrieval — cascading fallback chain (KB → statutes → case law → web)
  5. Fact Extraction — structured fact parsing for legal queries
  6. Two-Pass Reasoning — retrieval then structured analysis
  7. Citation Validation — hard-fail post-processing check
  8. Source Trace — transparency layer for debugging and audit
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

# Primary legal keywords — exact domain terms
LEGAL_KEYWORDS_PRIMARY = [
    "statute", "statutes", "case law", "case precedent",
    "plaintiff", "defendant", "appellant", "appellee",
    "amendment", "constitutional", "civil rights",
    "section 1983", "1983", "usc", "u.s.c.",
    "qualified immunity", "excessive force", "fourth amendment", "due process",
    "search and seizure", "miranda", "habeas", "certiorari", "injunction",
    "indictment", "arraignment", "sentencing",
]

# Fallback legal keywords — broader terms that indicate legal context
# These catch queries the primary list might miss
LEGAL_KEYWORDS_FALLBACK = [
    "law", "legal", "court", "ruling", "precedent",
    "rights", "brief", "motion", "jurisdiction", "appeal",
    "tort", "liability", "damages", "negligence", "malpractice",
    "prosecution", "probation", "parole", "illegal", "unconstitutional",
    "violation", "enforcement", "officer", "police", "arrest",
    "detained", "custody", "warrant", "probable cause",
    "terry stop", "traffic stop", "use of force",
]

RESEARCH_KEYWORDS = [
    "research", "find", "look up", "search for", "statistics",
    "data on", "studies", "evidence", "sources", "articles about",
]


def detect_intent(user_input: str) -> str:
    """
    Classify user intent to determine routing behavior.
    Uses a two-tier keyword system: primary (high confidence) and
    fallback (broader catch) to minimize missed legal intents.

    Returns: 'legal', 'research', 'general'
    """
    lower = user_input.lower()

    # Tier 1: Primary legal keywords — high confidence
    if any(kw in lower for kw in LEGAL_KEYWORDS_PRIMARY):
        return "legal"

    # Tier 2: Fallback legal keywords — broader catch
    # Require at least 2 matches to reduce false positives on casual use
    fallback_hits = sum(1 for kw in LEGAL_KEYWORDS_FALLBACK if kw in lower)
    if fallback_hits >= 2:
        return "legal"

    # Single fallback hit in combination with research keywords = legal
    if fallback_hits >= 1 and any(kw in lower for kw in RESEARCH_KEYWORDS):
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
        "\n[MODE: ARGUMENT] You MUST structure your response using this exact format:\n\n"
        "LEGAL ISSUE:\n[One-sentence framing of the core legal question]\n\n"
        "FORCE SEVERITY CHECK:\n[minimal / moderate / deadly, with one-sentence justification]\n\n"
        "PLAINTIFF ARGUMENT:\n[Numbered points with citations for each]\n\n"
        "DEFENSE ARGUMENT:\n[Numbered points with citations for each]\n\n"
        "KEY PRECEDENTS:\n"
        "- Include at least 1 controlling SCOTUS precedent\n"
        "- Include at least 1 supporting circuit/lower-court precedent\n"
        "- Weight precedent by authority (SCOTUS > circuit > lower court)\n"
        "- Only include deadly-force cases (e.g., Tennessee v. Garner) when facts involve deadly force\n\n"
        "WEAKNESSES:\n- Plaintiff: [biggest vulnerability]\n- Defense: [biggest vulnerability]\n\n"
        "LIKELY OUTCOME:\n[Brief assessment based on weight of authority]\n\n"
        "CONFIDENCE: [High / Medium / Low]\n"
        "REASON: [One to three sentences tied to precedent strength + factual ambiguity]\n\n"
        "DOCTRINE SCOPE RULE:\n"
        "- Only include legal doctrines that directly apply to the fact pattern.\n"
        "- For force during a stop/arrest, analyze under the Fourth Amendment and Graham v. Connor.\n"
        "- Do not introduce unrelated amendments or causes of action unless facts require them.\n\n"
        "DEFENSE QUALITY RULE:\n"
        "- Defense must be realistic and steelmanned (officer-safety, uncertainty, split-second context), never a placeholder.\n\n"
        "Do NOT deviate from this structure. Every claim must cite authority."
    ),
    "write": (
        "\n[MODE: WRITE] Produce a polished document (essay, brief, article, memo). "
        "Must include a Sources section with full citations at the end."
    ),
    "general": "",
}


# ── Two-Pass Legal Reasoning ───────────────────────────────────────────────

TWO_PASS_LEGAL_FRAMEWORK = """

═══════════════════════════════════════════════════════════
TWO-PASS LEGAL REASONING (MANDATORY FOR ALL LEGAL QUERIES)
═══════════════════════════════════════════════════════════

You MUST follow this structured reasoning process. Do NOT skip steps.

PASS 1 — RETRIEVAL (gather all materials first):
  Step 1: Search the retrieval context provided above
  Step 2: If gaps remain, use lookup_statute for specific citations
  Step 3: If gaps remain, use search_case_law for relevant precedent
  Step 4: If gaps remain, use web_search or search_legal_news

PASS 2 — STRUCTURED REASONING (only after Pass 1 is complete):
  Step 1: IDENTIFY the legal issue(s) — state them precisely
  Step 2: MAP relevant law — which statutes and cases apply, and why
  Step 3: APPLY law to facts — connect the legal standard to the specific scenario
  Step 4: COUNTERARGUMENTS — identify the strongest opposing position
  Step 5: CONCLUDE — state your conclusion using this EXACT format:

CONFIDENCE: High / Medium / Low
REASONING:
- Strength of precedent: [strong/moderate/weak — are there on-point SCOTUS/Circuit cases?]
- Jurisdiction match: [does the authority come from the relevant jurisdiction?]
- Factual similarity: [how closely do cited cases match the current facts?]
- Source conflicts: [do any authorities point in different directions?]

PRECEDENT WEIGHTING RULES:
When multiple cases are available, weight them in this order:
  1. SCOTUS decisions (highest authority)
  2. Federal Circuit decisions (from the relevant circuit)
  3. State Supreme Court decisions
  4. Federal District / lower court decisions (lowest weight)
If sources CONFLICT, you MUST:
  - Note the conflict explicitly
  - Prioritize the higher court's holding
  - Explain why the lower court may have diverged

If Pass 1 yields no relevant sources, STOP. Do not proceed to Pass 2.
Respond: "Insufficient legal authority found to support a conclusion."
"""


# ── Citation Validation ────────────────────────────────────────────────────

HARD_FAIL_RESPONSE = (
    "**Insufficient legal authority to provide an answer.**\n\n"
    "No verifiable statute or case citation could be produced for this query. "
    "The system requires grounded legal sources before delivering analysis.\n\n"
    "**Recommended next steps:**\n"
    "1. Use `lookup_statute` with a specific citation (e.g., '42 USC 1983')\n"
    "2. Use `search_case_law` with targeted keywords\n"
    "3. Ingest relevant documents into the knowledge base with `ingest_file`\n"
    "4. Try rephrasing your question with more specific legal terms\n\n"
    "Note: This is AI-generated legal research, not legal advice. "
    "Verify all citations independently."
)

DISCLAIMER = (
    "Note: This is AI-generated legal research, not legal advice. "
    "Verify all citations independently."
)


def validate_legal_response(response_text: str, intent: str, mode: str = "general") -> str:
    """
    Post-processing HARD enforcement: if the intent was legal, verify the
    response contains at least one verifiable citation. If not, REPLACE
    the response with a hard-fail message. No uncited legal output passes.
    """
    if intent != "legal":
        return response_text

    if not response_text or len(response_text.strip()) < 50:
        return response_text

    # The LLM already said "insufficient" — that's a valid grounded response
    if "insufficient" in response_text.lower() and "authority" in response_text.lower():
        if DISCLAIMER.lower() not in response_text.lower():
            return response_text + "\n\n---\n" + DISCLAIMER
        return response_text

    # Check for citation patterns
    has_statute = bool(re.search(
        r'\d+\s*U\.?S\.?C\.?\s*§?\s*\d+|\bSection\s+\d+|CT Gen Stat', response_text
    ))
    has_case = bool(re.search(
        r'[A-Z][a-z]+\s+v\.\s+[A-Z][a-z]+|\d+\s+(?:U\.S\.|S\.Ct\.|F\.\d+)', response_text
    ))

    # HARD FAIL: no citations at all → replace entire response
    if not has_statute and not has_case:
        return HARD_FAIL_RESPONSE

    # Citations exist — ensure disclaimer is present
    if mode == "argument":
        missing = []
        if "confidence:" not in response_text.lower():
            missing.append("CONFIDENCE")
        if "reason:" not in response_text.lower():
            missing.append("REASON")
        if missing:
            return (
                "**Output validation failed.**\n\n"
                f"Missing required section(s) for argument mode: {', '.join(missing)}.\n"
                "Regenerate with all required sections and legal citations.\n\n"
                + DISCLAIMER
            )

    if "not legal advice" not in response_text.lower():
        return response_text + "\n\n---\n" + DISCLAIMER

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
            #"document_writer": DocumentWriterTool(config),
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

    # Retrieval boundaries — prevent context flooding
    MAX_RETRIEVAL_CHUNKS = 3
    MAX_RETRIEVAL_CHARS = 4000

    def _force_retrieval(self, user_input: str, intent: str, messages: list) -> tuple[list, list]:
        """
        Cascading fallback chain for legal/research intents.
        Each step only fires if the previous yielded no results.

        Chain: 1. Local KB → 2. Statute lookup → 3. Case law search → 4. Web → 5. Hard fail instruction

        Returns: (messages, retrieval_trace) where retrieval_trace logs each step.
        """
        trace = []

        if intent not in ("legal", "research"):
            return messages, trace

        print(f"  [Forced Retrieval] Intent='{intent}' → cascading retrieval chain", flush=True)
        grounding_context = None

        # ── Step 1: Local Knowledge Base ───────────────────────────────
        kb = self.tool_instances.get("knowledge_base")
        if kb:
            print(f"  [Fallback 1/4] Querying knowledge base (max_chunks={self.MAX_RETRIEVAL_CHUNKS})...", flush=True)
            try:
                kb_result = kb.query_knowledge(
                    query=user_input,
                    max_results=self.MAX_RETRIEVAL_CHUNKS,
                    rerank=(intent == "legal"),
                )
                if self._has_valid_results(kb_result):
                    grounding_context = kb_result
                    trace.append("knowledge_base: results found")
                    print(f"  [Fallback 1/4] KB returned results ({len(kb_result)} chars)", flush=True)
                else:
                    trace.append("knowledge_base: no results")
                    print("  [Fallback 1/4] KB empty — escalating", flush=True)
            except Exception as e:
                trace.append(f"knowledge_base: error ({e})")
                print(f"  [Fallback 1/4] KB error: {e}", flush=True)

        # ── Step 2: Statute Lookup (legal only) ────────────────────────
        if not grounding_context and intent == "legal":
            legal = self.tool_instances.get("legal_research")
            if legal and hasattr(legal, 'search_case_law'):
                print("  [Fallback 2/4] Searching case law via CourtListener...", flush=True)
                try:
                    case_result = legal.search_case_law(query=user_input, max_results=3)
                    if case_result and "error" not in str(case_result).lower() and len(str(case_result)) > 50:
                        grounding_context = str(case_result)
                        trace.append("case_law_search: results found")
                        print(f"  [Fallback 2/4] Case law returned results ({len(grounding_context)} chars)", flush=True)
                    else:
                        trace.append("case_law_search: no results")
                        print("  [Fallback 2/4] Case law empty — escalating", flush=True)
                except Exception as e:
                    trace.append(f"case_law_search: error ({e})")
                    print(f"  [Fallback 2/4] Case law error: {e}", flush=True)

        # ── Step 3: Legal News Search ──────────────────────────────────
        if not grounding_context and intent == "legal":
            legal = self.tool_instances.get("legal_research")
            if legal and hasattr(legal, 'search_legal_news'):
                print("  [Fallback 3/4] Searching legal news...", flush=True)
                try:
                    news_result = legal.search_legal_news(topic=user_input)
                    if news_result and "error" not in str(news_result).lower() and len(str(news_result)) > 50:
                        grounding_context = str(news_result)
                        trace.append("legal_news: results found")
                        print(f"  [Fallback 3/4] Legal news returned results ({len(grounding_context)} chars)", flush=True)
                    else:
                        trace.append("legal_news: no results")
                        print("  [Fallback 3/4] Legal news empty — escalating", flush=True)
                except Exception as e:
                    trace.append(f"legal_news: error ({e})")
                    print(f"  [Fallback 3/4] Legal news error: {e}", flush=True)

        # ── Step 4: Web Search (last resort) ───────────────────────────
        if not grounding_context:
            web = self.tool_instances.get("web_search")
            if web and hasattr(web, 'web_search'):
                search_query = f"legal {user_input}" if intent == "legal" else user_input
                print(f"  [Fallback 4/4] Web search: '{search_query[:60]}...'", flush=True)
                try:
                    web_result = web.web_search(query=search_query, max_results=3)
                    if web_result and "error" not in str(web_result).lower() and len(str(web_result)) > 50:
                        grounding_context = str(web_result)
                        trace.append("web_search: results found")
                        print(f"  [Fallback 4/4] Web returned results ({len(grounding_context)} chars)", flush=True)
                    else:
                        trace.append("web_search: no results")
                        print("  [Fallback 4/4] Web search empty", flush=True)
                except Exception as e:
                    trace.append(f"web_search: error ({e})")
                    print(f"  [Fallback 4/4] Web search error: {e}", flush=True)

        # ── Inject context or hard-fail instruction ────────────────────
        if grounding_context:
            # Enforce character boundary
            if len(grounding_context) > self.MAX_RETRIEVAL_CHARS:
                grounding_context = grounding_context[:self.MAX_RETRIEVAL_CHARS] + "\n\n[...truncated for context limits]"
                print(f"  [Forced Retrieval] Truncated to {self.MAX_RETRIEVAL_CHARS} chars", flush=True)

            print(f"  [Forced Retrieval] Injecting {len(grounding_context)} chars of grounding context", flush=True)
            messages.append({
                "role": "system",
                "content": (
                    "[RETRIEVAL CONTEXT — grounding data retrieved before reasoning]\n"
                    "The following sources were retrieved BEFORE reasoning. "
                    "You MUST use these to ground your response. "
                    "Only use the most relevant portions. Ignore unrelated sections. "
                    "Do NOT ignore retrieved sources.\n\n" + grounding_context
                )
            })
        else:
            trace.append("ALL SOURCES EXHAUSTED — no grounding found")
            print("  [Forced Retrieval] ALL FALLBACKS EXHAUSTED — no grounding data found", flush=True)
            messages.append({
                "role": "system",
                "content": (
                    "[RETRIEVAL CONTEXT] All retrieval sources exhausted. No relevant documents found in: "
                    "knowledge base, case law databases, legal news, or web search. "
                    "You MUST respond with: 'Insufficient legal authority found to support a conclusion. "
                    "I recommend broadening the search or consulting additional sources.' "
                    "Do NOT answer from memory. Do NOT fabricate citations."
                )
            })

        return messages, trace

    @staticmethod
    def _has_valid_results(result) -> bool:
        """Check if a retrieval result contains usable content."""
        if not result:
            return False
        result_str = str(result)
        if len(result_str) < 50:
            return False
        if "No relevant results" in result_str:
            return False
        if "empty" in result_str.lower() and len(result_str) < 100:
            return False
        return True

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

    def _extract_facts(self, user_input: str, intent: str) -> str | None:
        """
        For legal intents, extract structured facts from the user's input
        before reasoning. This improves consistency and argument quality
        by giving the LLM a clean fact pattern to work with.

        Returns structured fact block or None if not applicable.
        """
        if intent != "legal":
            return None

        # Only extract facts if the input looks like a scenario (not a pure lookup)
        lower = user_input.lower()
        lookup_signals = ["what is", "define", "look up", "find me", "search for", "download"]
        if any(lower.startswith(s) for s in lookup_signals):
            return None

        # Check if input has enough substance for fact extraction (scenario-like)
        if len(user_input) < 80:
            return None

        print("  [Fact Extraction] Input looks like a scenario — extracting structured facts...", flush=True)

        try:
            fast_model = self.config.get("llm", {}).get("fast_model", "llama3.1:8b")
            response = ollama.chat(
                model=fast_model,
                messages=[{
                    "role": "user",
                    "content": (
                        "Extract the key legal facts from this scenario. "
                        "Be precise and brief. Use this EXACT format:\n\n"
                        "ACTORS: [who is involved — roles, not names if unknown]\n"
                        "ACTIONS: [what happened — chronological sequence]\n"
                        "LOCATION/CONTEXT: [where, when, under what circumstances]\n"
                        "FORCE USED: [type and level, if applicable — or 'N/A']\n"
                        "RESISTANCE: [level of resistance, if applicable — or 'N/A']\n"
                        "OUTCOME: [what resulted — injury, arrest, charge, etc.]\n"
                        "LEGAL CLAIMS: [what legal issues are likely at play]\n\n"
                        f"Scenario:\n{user_input}"
                    )
                }],
                options={"temperature": 0, "num_ctx": 4096}
            )
            facts = response["message"]["content"].strip()
            if facts and len(facts) > 30:
                print(f"  [Fact Extraction] Extracted {len(facts)} chars of structured facts", flush=True)
                return facts
        except Exception as e:
            print(f"  [Fact Extraction] Error: {e}", flush=True)

        return None

    @staticmethod
    def _build_source_trace(retrieval_trace: list, tools_invoked: list) -> str:
        """
        Build a transparency block showing which sources and tools were used.
        Appended to legal responses for audit and debugging.
        """
        if not retrieval_trace and not tools_invoked:
            return ""

        lines = ["\n\n---\n**Source Trace:**"]

        if retrieval_trace:
            lines.append("Retrieval chain:")
            for step in retrieval_trace:
                lines.append(f"  - {step}")

        if tools_invoked:
            lines.append("Tools invoked:")
            for tool in tools_invoked:
                lines.append(f"  - {tool}")

        return "\n".join(lines)

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

        # Build message list with system prompt + mode instruction + reasoning framework
        mode_instruction = MODE_INSTRUCTIONS.get(self.mode, "")
        system_content = SYSTEM_PROMPT + mode_instruction

        # Two-pass reasoning framework for legal intents
        if intent == "legal":
            system_content += TWO_PASS_LEGAL_FRAMEWORK

        messages = [{"role": "system", "content": system_content}] + self.history

        # ── Step 1.5: Fact extraction for legal scenarios ──────────────
        extracted_facts = self._extract_facts(user_input, intent)
        if extracted_facts:
            messages.append({
                "role": "system",
                "content": (
                    "[STRUCTURED FACTS — extracted from user's scenario]\n"
                    "Use these structured facts to guide your legal analysis. "
                    "Do not re-interpret the facts; work from this extraction.\n\n"
                    + extracted_facts
                )
            })

        # ── Step 2: Forced retrieval for legal/research intents ────────
        messages, retrieval_trace = self._force_retrieval(user_input, intent, messages)

        # ── Step 3: Agent loop ─────────────────────────────────────────
        tools_invoked = []  # Track for source trace
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

                    # Track tool usage for source trace
                    tools_invoked.append(func_name)

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
                final = validate_legal_response(final, intent, self.mode)

                # ── Step 5: Append source trace for legal responses ────
                if intent == "legal":
                    source_trace = self._build_source_trace(retrieval_trace, tools_invoked)
                    if source_trace:
                        final += source_trace

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
