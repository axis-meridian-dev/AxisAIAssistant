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
from cloud_reasoning import CloudReasoner

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


def validate_and_patch(response_text, intent, mode):
    if intent not in ("legal", "legal_adjacent"):
        return response_text
    patched = False
    
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

RULES:
1. ALWAYS cite specific statutes (e.g., CGS § 53a-22, 42 U.S.C. § 1983) and cases (e.g., Graham v. Connor, 490 U.S. 386 (1989))
2. Structure every legal response with: APPLICABLE LAW, CASE LAW, APPLICATION/ANALYSIS sections
3. When analyzing a CT case, always reference both federal constitutional protections AND Connecticut's Article I, § 7 (which provides broader protections)
4. End with CONFIDENCE scoring and a disclaimer that this is research, not legal advice
5. Be direct, thorough, and adversarial on behalf of the defendant — identify every weakness in the state's case
6. Reference the CT Police Accountability Act (PA 20-1) when use of force is at issue
7. When writing legal documents, use proper legal formatting and citation style

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

LEGAL_PROMPT_SUFFIX = {
    "analysis": "\n\nYou MUST structure your response with: APPLICABLE LAW, CASE LAW, APPLICATION, CONFIDENCE sections.",
    "research": "\n\nRESEARCH MODE: Return ONLY raw sources. No interpretation. STATUTES, CASES, DATA, SOURCES.",
    "argument": "\n\nARGUMENT MODE: Build BOTH sides. PLAINTIFF/PROSECUTION arguments + DEFENSE arguments, each with citations. Then STRONGEST ARGUMENT.",
    "writing": "\n\nWRITING MODE: Produce a complete, well-structured legal document. Include SOURCES section.",
}


# ── Agent ───────────────────────────────────────────────────────────────────

class Agent:
    def __init__(self, config: dict):
        self.config = config
        self.history: list[dict] = []
        self.ollama_host = config["llm"]["ollama_host"]
        self.current_mode = "general"
        
        # Cloud reasoning layer
        self.cloud = CloudReasoner(config)
        
        # Initialize tools
        self.tool_instances = {
            "file_manager": FileManagerTool(config),
            "web_search": WebSearchTool(config),
            "desktop_control": DesktopControlTool(config),
            "system_info": SystemInfoTool(config),
            "knowledge_base": KnowledgeBaseTool(config),
            "legal_research": LegalResearchTool(config),
            "document_writer": DocumentWriterTool(config),
        }
        
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
    
    def clear_history(self):
        self.history = []
        self.current_mode = "general"
    
    async def process(self, user_input: str) -> str:
        """Process user input through the hybrid agent loop."""
        
        self.history.append({"role": "user", "content": user_input})
        
        # ── Step 1: Intent + Mode ─────────────────────────────────────
        intent = detect_intent(user_input)
        
        if intent in ("legal", "legal_adjacent"):
            legal_mode = detect_legal_mode(user_input)
            if self.current_mode == "general" or legal_mode != "analysis":
                self.current_mode = legal_mode
                print(f"  [Mode] {legal_mode}", flush=True)
        
        print(f"  [Intent: {intent} | Mode: {self.current_mode}]", flush=True)
        
        # ── Step 2: Check cloud routing ───────────────────────────────
        use_cloud = self.cloud.should_use_cloud(user_input, intent, self.current_mode)
        if use_cloud:
            print(f"  [Routing] → CLOUD ({self.cloud.provider})", flush=True)
        else:
            print(f"  [Routing] → LOCAL (Ollama)", flush=True)
        
        # ── Step 3: Local tool calling phase ──────────────────────────
        # Always use local model for tool calls — it's fast and good at routing
        model = self.config["llm"]["primary_model"]
        print(f"  [Tools Model: {model}]", flush=True)
        
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
                            print(f"    ✗ {result}", flush=True)
                    else:
                        result = f"Unknown tool: {func_name}"
                        print(f"    ✗ {result}", flush=True)
                    
                    messages.append({"role": "tool", "content": str(result)})
            else:
                # ── Step 4: Final response ────────────────────────────
                local_response = msg.get("content", "")
                print(f"  [Local] Got response ({len(local_response)} chars)", flush=True)
                
                # ── Step 5: Cloud reasoning (if enabled) ──────────────
                if use_cloud and self.cloud.enabled:
                    tool_context = "\n\n".join(tool_results_collected) if tool_results_collected else ""
                    
                    # Add local model's response as additional context
                    if local_response and len(local_response) > 50:
                        tool_context += f"\n\n[Local Model Analysis]\n{local_response[:2000]}"
                    
                    cloud_response = self.cloud.query(
                        messages=self.history,
                        system_prompt=CLOUD_LEGAL_PROMPT,
                        tool_results=tool_context,
                    )
                    
                    if cloud_response:
                        final = cloud_response
                        print(f"  [Cloud] Using cloud response ({len(final)} chars)", flush=True)
                    else:
                        final = local_response
                        print(f"  [Cloud] Failed — falling back to local", flush=True)
                else:
                    final = local_response
                
                # ── Step 6: Validate and patch ────────────────────────
                final = validate_and_patch(final, intent, self.current_mode)
                
                self.history.append({"role": "assistant", "content": final})
                
                if len(self.history) > 40:
                    self.history = self.history[-30:]
                
                return final
        
        return "Reached maximum tool iterations. Please try a simpler request."
