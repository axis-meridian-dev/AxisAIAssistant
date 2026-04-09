"""
Core Agent v4 — Hybrid Local + Cloud Architecture.

Flow:
  1. User query → Intent detection → Mode escalation
  2. Local model (Ollama) → Tool calls (file ops, search, statute lookup)
  3. Tool results collected
  4. Decision: Local or Cloud for final response?
     - Simple/general → Local model generates response
     - Legal/complex → Cloud API (Claude/GPT-4) generates response using tool results
  5. Response validation → Auto-patching → Return

The local model is the "hands" — it knows which tools to call.
The cloud model is the "brain" — it does the deep reasoning and writing.
"""

import logging
import re
import json
import ollama
from rich.console import Console
from rich.table import Table

logger = logging.getLogger("ai_assistant.agent")

from tools.file_manager import FileManagerTool
from tools.web_search import WebSearchTool
from tools.desktop_control import DesktopControlTool
from tools.system_info import SystemInfoTool
from tools.knowledge_base import KnowledgeBaseTool
from tools.legal_research import LegalResearchTool
from tools.document_writer import DocumentWriterTool
from cloud_reasoning import CloudReasoner
from stats import SessionStats, InquiryStats, StatsTimer

console = Console()

# ── Intent Detection ────────────────────────────────────────────────────────

LEGAL_KEYWORDS = [
    "statute", "law", "legal", "court", "case", "judge", "attorney",
    "lawyer", "defendant", "plaintiff", "prosecution", "defense",
    "trial", "verdict", "sentence", "sentencing", "plea", "bail",
    "arrest", "search", "seizure", "warrant", "miranda", "rights",
    "amendment", "constitutional", "civil rights", "section 1983",
    "1983", "usc", "u.s.c", "habeas", "appeal", "motion",
    "evidence", "testimony", "witness", "jury", "indictment",
    "arraignment", "discovery", "subpoena", "deposition",
    "excessive force", "qualified immunity", "due process",
    "equal protection", "probable cause", "reasonable suspicion",
    "police", "officer", "misconduct", "brutality", "use of force",
    "public defender", "pro se", "felony", "misdemeanor",
    "violation", "criminal", "charge", "conviction", "acquittal",
    "suppress", "motion to suppress", "fourth amendment", "fifth amendment",
    "sixth amendment", "eighth amendment", "fourteenth amendment",
    "discrimination", "retaliation", "housing", "employment",
    "title vii", "ada", "fair housing", "racial profiling",
    "body cam", "body camera", "traffic stop", "terry stop",
    "stop and frisk", "consent search", "pat down",
    "ct gen stat", "connecticut general statutes", "practice book",
    "penal code", "53a", "54-", "46a", "29-6",
]

LEGAL_CASE_PATTERNS = [
    r'\bv\.\s', r'\d+\s*U\.?S\.?C\.?\s*§?\s*\d+',
    r'\d+\s+(?:U\.S\.|S\.Ct\.|F\.\d)', r'§\s*\d+',
]


def detect_intent(text):
    lower = text.lower()
    legal_score = sum(1 for kw in LEGAL_KEYWORDS if kw in lower)
    for pattern in LEGAL_CASE_PATTERNS:
        if re.search(pattern, text):
            legal_score += 3
    if legal_score >= 3:
        return "legal"
    if legal_score >= 1:
        return "legal_adjacent"
    tech_keywords = ["code", "python", "script", "function", "api", "debug",
                     "error", "install", "config", "server", "database"]
    if any(kw in lower for kw in tech_keywords):
        return "technical"
    return "general"


def detect_legal_mode(text):
    lower = text.lower()
    if any(p in lower for p in ["research mode", "just show sources", "raw sources"]):
        return "research"
    if any(p in lower for p in ["argument mode", "build arguments", "both sides"]):
        return "argument"
    if any(p in lower for p in ["writing mode", "write an essay", "write a brief",
                                 "draft a", "compose", "memorandum"]):
        return "writing"
    return "analysis"


# ── Response Validation ─────────────────────────────────────────────────────

LANDMARK_CASES = [
    "Graham v. Connor", "Terry v. Ohio", "Tennessee v. Garner",
    "Miranda v. Arizona", "Mapp v. Ohio", "Gideon v. Wainwright",
    "Brady v. Maryland", "Monell v. Department", "Whren v. United States",
    "Batson v. Kentucky", "Strickland v. Washington", "Pearson v. Callahan",
    "Hope v. Pelzer", "Payton v. New York", "Schneckloth v. Bustamonte",
    "Georgia v. Randolph", "Wong Sun v. United States", "Doyle v. Ohio",
    "Arizona v. Gant", "Rodriguez v. United States", "Carpenter v. United States",
    "Riley v. California", "Kingsley v. Hendrickson", "City of Canton v. Harris",
]


def compute_confidence(response_text, intent):
    if intent not in ("legal", "legal_adjacent"):
        return "N/A", []
    score = 0
    reasons = []
    
    statute_refs = re.findall(r'\d+\s*U\.?S\.?C\.?\s*§?\s*\d+', response_text)
    ct_refs = re.findall(r'(?:§|Sec\.?|CGS|Section)\s*\d+[a-z]?-\d+', response_text)
    all_statutes = statute_refs + ct_refs
    if len(all_statutes) >= 3: score += 4; reasons.append(f"Strong statutory basis: {len(all_statutes)} citations")
    elif len(all_statutes) >= 1: score += 2; reasons.append(f"Statutory basis: {len(all_statutes)} citation(s)")
    else: reasons.append("No statute citations — reliability reduced")
    
    case_matches = re.findall(r'[A-Z][a-z]+\s+v\.\s+[A-Z][a-z]+', response_text)
    landmark_found = [c for c in LANDMARK_CASES if c.lower() in response_text.lower()]
    if len(case_matches) >= 3: score += 4; reasons.append(f"Strong case authority: {len(case_matches)} citations")
    elif len(case_matches) >= 1: score += 2; reasons.append(f"Case authority: {len(case_matches)} citation(s)")
    else: reasons.append("No case law cited")
    
    if any(j in response_text.lower() for j in ["connecticut", "ct gen stat", "cgs §", "conn."]):
        score += 2; reasons.append("CT jurisdiction-specific")
    
    structured = ["APPLICABLE LAW:", "CASE LAW:", "APPLICATION:", "ANALYSIS:", "CONCLUSION:"]
    if sum(1 for m in structured if m in response_text) >= 2:
        score += 2; reasons.append("Well-structured analysis")
    
    if score >= 10: return "High", reasons
    elif score >= 6: return "Medium-High", reasons
    elif score >= 3: return "Medium", reasons
    elif score >= 1: return "Low-Medium", reasons
    return "Low", reasons


def validate_and_patch(response_text, intent, mode, tool_results=None):
    if intent not in ("legal", "legal_adjacent"):
        return response_text
    patched = False

    # Patch: warn if no tools were used for a legal query
    if not tool_results:
        if "[UNVERIFIED]" not in response_text and "no sources were retrieved" not in response_text.lower():
            response_text = (
                "**WARNING: No legal research tools were called for this query. "
                "All citations below are unverified and may be inaccurate. "
                "Ask me to look up specific statutes or search case law for grounded results.**\n\n"
                + response_text
            )
            patched = True

    # Patch: cross-check cited statutes against tool results
    if tool_results:
        tool_text = "\n".join(tool_results).lower()

        # Find statutes cited in response
        cited_federal = re.findall(r'(\d+)\s*U\.?S\.?C\.?\s*§?\s*(\d+\w*)', response_text)
        cited_ct = re.findall(r'(?:CGS|CT Gen\.? Stat\.?|Conn\.? Gen\.? Stat\.?|§)\s*([\d]+[a-z]?[-–]\d+\w*)', response_text, re.IGNORECASE)

        unverified = []
        for title, sec in cited_federal:
            # Check if this statute appeared in tool output
            patterns = [f"{title} usc § {sec}", f"{title} u.s.c. § {sec}",
                        f"text/{title}/{sec}", f"{title}_usc_{sec}"]
            if not any(p in tool_text for p in patterns):
                unverified.append(f"{title} USC § {sec}")

        for sec in cited_ct:
            patterns = [sec.lower(), sec.replace("-", "_").lower(), f"sec_{sec.lower()}"]
            if not any(p in tool_text for p in patterns):
                unverified.append(f"CGS § {sec}")

        if unverified:
            warning = (
                f"\n\n**CITATION VERIFICATION WARNING:** The following citations were NOT found in the "
                f"retrieved research data and may need independent verification:\n"
            )
            for u in unverified[:10]:  # Cap at 10
                warning += f"- {u}\n"
            warning += "Run `lookup_statute` on these to verify they exist and are correctly cited.\n"
            response_text += warning
            patched = True

    # Patch: confidence scoring
    if "CONFIDENCE:" not in response_text:
        confidence, reasons = compute_confidence(response_text, intent)
        block = f"\n\nCONFIDENCE: {confidence}\n\nREASONING:\n"
        for r in reasons: block += f"- {r}\n"
        response_text += block
        patched = True

    # Patch: disclaimer
    if "not legal advice" not in response_text.lower():
        response_text += "\n\nNote: This is AI-generated legal research, not legal advice. Verify all citations independently."
        patched = True

    if patched:
        print("  [Validator] Auto-patched response", flush=True)
    return response_text


# ── Prompt Templates ────────────────────────────────────────────────────────

CLOUD_LEGAL_PROMPT = """You are a legal research assistant analyzing Connecticut criminal defense cases.
You have access to local research data including CT General Statutes, federal statutes, case law from CourtListener,
and the user's personal case files stored in ~/LegalResearch/.

CRITICAL ANTI-HALLUCINATION RULES:
1. ONLY cite statutes and cases that appear in the [Research Data] provided below. Do NOT fabricate citations from memory.
2. If the research data does not contain a specific statute or case, you MUST say "I could not find [X] in the available data — please run a search for it" instead of guessing.
3. If no research data was provided (no tool results), you MUST state: "No sources were retrieved for this query. The following is based on general knowledge and MUST be independently verified." Then prefix any citation with [UNVERIFIED].
4. Never invent statute section numbers. If you know a statute exists but don't have the exact text, say "see [statute name] (exact section number needs verification)".
5. For case law, always include the citation (volume, reporter, page, year) ONLY if it appeared in the research data. Otherwise mark as [UNVERIFIED].

RESPONSE RULES:
6. Structure every legal response with: APPLICABLE LAW, CASE LAW, APPLICATION/ANALYSIS sections
7. When analyzing a CT case, always reference both federal constitutional protections AND Connecticut's Article I, § 7 (which provides broader protections)
8. End with CONFIDENCE scoring and a disclaimer that this is research, not legal advice
9. Be direct, thorough, and adversarial on behalf of the defendant — identify every weakness in the state's case
10. Reference the CT Police Accountability Act (PA 20-1) when use of force is at issue
11. When writing legal documents, use proper legal formatting and citation style

The user is a defendant in Connecticut with active court cases. He has a public defender who has not been filing motions.
He is building his own legal research to bring to counsel. Treat every query with the seriousness it deserves."""

LOCAL_SYSTEM_PROMPT = """IMPORTANT: Always respond in English. The user's home directory is /home/axmh. Always use absolute paths starting with /home/axmh. All paths are case-sensitive - use lowercase. Never respond in any other language under any circumstances.
When searching for files by extension or filename, ALWAYS use search_type "name", never "content". Only use "content" when searching for text inside files.

You are a local AI assistant on a Linux computer with access to tools.
Your job is to USE TOOLS to gather information, then provide answers.

For legal questions: Use lookup_statute, search_case_law, query_knowledge, and search_legal_news 
to gather research data BEFORE answering. Always use tools first.

For file operations: Use file management tools directly.
For web queries: Use web_search and fetch_webpage.

Be concise in tool selection. Call the right tools efficiently.

You have these tools:
1. FILE MANAGEMENT — read, write, search, organize files
2. WEB SEARCH — search internet, fetch web pages
3. DESKTOP CONTROL — launch apps, manage windows, commands
4. SYSTEM INFO — hardware stats, processes
5. KNOWLEDGE BASE — semantic search across ingested documents
6. LEGAL RESEARCH — statutes, case law, statistics, news
7. DOCUMENT WRITER — generate essays, briefs, documents

Think step by step about which tools to use, then use them."""

LEGAL_TOOL_ENFORCEMENT = """

MANDATORY FOR ALL LEGAL QUESTIONS: You MUST call at least one research tool (lookup_statute, search_case_law, query_knowledge, or search_legal_news) BEFORE providing your answer. NEVER answer a legal question from memory alone — always retrieve sources first. If you are unsure which tool to use, call search_case_law with the topic, then lookup_statute for any relevant statutes."""

LEGAL_PROMPT_SUFFIX = {
    "analysis": LEGAL_TOOL_ENFORCEMENT + "\n\nYou MUST structure your response with: APPLICABLE LAW, CASE LAW, APPLICATION, CONFIDENCE sections.",
    "research": LEGAL_TOOL_ENFORCEMENT + "\n\nRESEARCH MODE: Return ONLY raw sources. No interpretation. STATUTES, CASES, DATA, SOURCES.",
    "argument": LEGAL_TOOL_ENFORCEMENT + "\n\nARGUMENT MODE: Build BOTH sides. PLAINTIFF/PROSECUTION arguments + DEFENSE arguments, each with citations. Then STRONGEST ARGUMENT.",
    "writing": LEGAL_TOOL_ENFORCEMENT + "\n\nWRITING MODE: Produce a complete, well-structured legal document. Include SOURCES section.",
}


# ── Agent ───────────────────────────────────────────────────────────────────

class Agent:
    def __init__(self, config: dict):
        self.config = config
        self.history: list[dict] = []
        self.ollama_host = config["llm"]["ollama_host"]
        self.current_mode = "general"
        self.session_stats = SessionStats()
        self.last_inquiry_stats = None
        self.mode = self.current_mode
        self.cloud_model_override: str | None = None  # user-selected model

        # Cloud reasoning layer
        self.cloud = CloudReasoner(config)
        
        # Initialize tools
        self._all_tool_instances = {
            "file_manager": FileManagerTool(config),
            "web_search": WebSearchTool(config),
            "desktop_control": DesktopControlTool(config),
            "system_info": SystemInfoTool(config),
            "knowledge_base": KnowledgeBaseTool(config),
            "legal_research": LegalResearchTool(config),
            "document_writer": DocumentWriterTool(config),
        }
        self.tool_instances = dict(self._all_tool_instances)

        # Apply tool toggles from config
        self.apply_tool_toggles(config.get("enabled_tools", {}))
    
    def apply_tool_toggles(self, enabled_tools: dict):
        """Rebuild active tools based on enabled_tools config."""
        self.tool_instances = {
            name: inst for name, inst in self._all_tool_instances.items()
            if enabled_tools.get(name, True)
        }
        self._rebuild_tool_index()

    def _rebuild_tool_index(self):
        """Rebuild tool definitions and handler lookup from active instances."""
        self.tools = []
        for instance in self.tool_instances.values():
            self.tools.extend(instance.get_tool_definitions())
        self.tool_handlers = {}
        for instance in self.tool_instances.values():
            self.tool_handlers.update(instance.get_handlers())

    def list_tools(self) -> list[str]:
        return list(self.tool_instances.keys())
    
    def print_tools(self):
        table = Table(title="Available Tools", border_style="cyan")
        table.add_column("Category", style="bold")
        table.add_column("Functions", style="dim")
        for name, instance in self.tool_instances.items():
            funcs = ", ".join(d["function"]["name"] for d in instance.get_tool_definitions())
            table.add_row(name, funcs)
        console.print(table)
        console.print(f"\n{self.cloud.get_status()}")
    
    def set_mode(self, new_mode: str) -> str:
        """Change the agent's operating mode."""
        valid_modes = ("research", "analysis", "argument", "write", "general")
        new_mode = new_mode.strip().lower()
        if new_mode in valid_modes:
            self.current_mode = new_mode
            self.mode = new_mode
            logger.info("Mode changed to %s", new_mode)
            return f"Mode set to: {new_mode}"
        return f"Invalid mode '{new_mode}'. Choose from: {', '.join(valid_modes)}"

    def load_session(self, session_id: str) -> bool:
        """Load and resume a previous chat session."""
        data = self.session_stats.resume_session(session_id)
        if data:
            self.history = data.get("messages", [])
            return True
        return False

    def new_session(self):
        """Start a fresh session (like clicking 'New Chat' in ChatGPT)."""
        # Save current session first if it has content
        if self.history:
            self.session_stats.save_chat_session(self.history)
        self.history = []
        self.current_mode = "general"
        self.mode = "general"
        self.cloud_model_override = None
        self.session_stats = SessionStats()
        self.last_inquiry_stats = None

    def clear_history(self):
        self.history = []
        self.current_mode = "general"
        self.session_stats = SessionStats()
        self.last_inquiry_stats = None
        self.mode = self.current_mode
    
    async def process(self, user_input: str) -> str:
        """Process user input through the hybrid agent loop."""
        
        self.history.append({"role": "user", "content": user_input})
        self._verbose_log = []

        # ── Step 1: Intent + Mode ─────────────────────────────────────
        intent = detect_intent(user_input)
        
        if intent in ("legal", "legal_adjacent"):
            legal_mode = detect_legal_mode(user_input)
            if self.current_mode == "general" or legal_mode != "analysis":
                self.current_mode = legal_mode
                print(f"  [Mode] {legal_mode}", flush=True)
        
        print(f"  [Intent: {intent} | Mode: {self.current_mode}]", flush=True)
        self._verbose_log.append(f"Intent: {intent} | Mode: {self.current_mode}")

        # ── Step 2: Check cloud routing ───────────────────────────────
        use_cloud = self.cloud.should_use_cloud(user_input, intent, self.current_mode)
        if use_cloud:
            print(f"  [Routing] → CLOUD ({self.cloud.provider})", flush=True)
            self._verbose_log.append(f"Routing → CLOUD ({self.cloud.provider})")
        else:
            print(f"  [Routing] → LOCAL (Ollama)", flush=True)
            self._verbose_log.append("Routing → LOCAL (Ollama)")
        
        # ── Step 3: Local tool calling phase ──────────────────────────
        # Always use local model for tool calls — it's fast and good at routing
        model = self.config["llm"]["primary_model"]
        print(f"  [Tools Model: {model}]", flush=True)
        self._verbose_log.append(f"Tools Model: {model}")
        
        tool_prompt = LOCAL_SYSTEM_PROMPT
        if intent in ("legal", "legal_adjacent"):
            tool_prompt += LEGAL_PROMPT_SUFFIX.get(self.current_mode, "")
        
        messages = [{"role": "system", "content": tool_prompt}] + self.history
        
        tool_results_collected = []  # Accumulate tool results for cloud
        
        max_iterations = 10
        for i in range(max_iterations):
            print(f"  [Step {i+1}] Local LLM...", flush=True)
            
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
                logger.error("Ollama call failed: %s", e)
                print(f"  [ERROR] {error_msg}", flush=True)
                return error_msg
            
            msg = response["message"]
            messages.append(msg)
            
            if msg.get("tool_calls"):
                print(f"  [Step {i+1}] Calling {len(msg['tool_calls'])} tool(s):", flush=True)
                
                for tool_call in msg["tool_calls"]:
                    func_name = tool_call["function"]["name"]
                    func_args = tool_call["function"]["arguments"]
                    
                    args_preview = json.dumps(func_args, default=str)
                    if len(args_preview) > 150:
                        args_preview = args_preview[:150] + "..."
                    print(f"    → {func_name}({args_preview})", flush=True)
                    self._verbose_log.append(f"Tool call: {func_name}({args_preview})")
                    
                    handler = self.tool_handlers.get(func_name)
                    if handler:
                        try:
                            result = handler(**func_args)
                            result_str = str(result)
                            if len(result_str) > 200:
                                print(f"    ✓ Got result ({len(result_str)} chars)", flush=True)
                            else:
                                print(f"    ✓ {result_str}", flush=True)
                            
                            # Collect for cloud
                            tool_results_collected.append(
                                f"[{func_name}] {result_str[:3000]}"
                            )
                        except Exception as e:
                            result = f"Tool error: {e}"
                            logger.error("Tool %s failed: %s", func_name, e)
                            print(f"    ✗ {result}", flush=True)
                    else:
                        result = f"Unknown tool: {func_name}"
                        print(f"    ✗ {result}", flush=True)
                    
                    messages.append({"role": "tool", "content": str(result)})
            else:
                # ── Step 4: Final response ────────────────────────────
                local_response = msg.get("content", "")
                print(f"  [Local] Got response ({len(local_response)} chars)", flush=True)
                self._verbose_log.append(f"Local response: {len(local_response)} chars")
                
                # ── Step 5: Cloud reasoning (if enabled) ──────────────
                if use_cloud and self.cloud.enabled:
                    tool_context = "\n\n".join(tool_results_collected) if tool_results_collected else ""
                    
                    # Add local model's response as additional context
                    if local_response and len(local_response) > 50:
                        tool_context += f"\n\n[Local Model Analysis]\n{local_response[:2000]}"
                    
                    selected_model = self.cloud_model_override or self.cloud.select_model(user_input, intent, self.current_mode)
                    cloud_response = self.cloud.query(
                        messages=self.history,
                        system_prompt=CLOUD_LEGAL_PROMPT,
                        tool_results=tool_context,
                        model=selected_model,
                    )
                    
                    if cloud_response:
                        final = cloud_response
                        print(f"  [Cloud] Using cloud response ({len(final)} chars)", flush=True)
                        self._verbose_log.append(f"Cloud response ({selected_model}): {len(final)} chars")
                    else:
                        final = local_response
                        print(f"  [Cloud] Failed — falling back to local", flush=True)
                        self._verbose_log.append("Cloud failed — fallback to local")
                else:
                    final = local_response
                
                # ── Step 6: Validate and patch ────────────────────────
                final = validate_and_patch(final, intent, self.current_mode,
                                           tool_results=tool_results_collected or None)
                
                self.history.append({"role": "assistant", "content": final})

                # Auto-save after every exchange
                self.session_stats.save_chat_session(self.history)

                if len(self.history) > 40:
                    self.history = self.history[-30:]
                    self._history_truncated = True
                else:
                    self._history_truncated = False

                return final
        
        return "Reached maximum tool iterations. Please try a simpler request."
