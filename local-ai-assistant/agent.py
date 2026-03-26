"""
Core Agent — LLM reasoning + tool dispatch loop.

Uses Ollama's tool-calling API to let the LLM decide which tools to invoke.
Runs a loop: LLM → tool call → result → LLM → ... until final answer.
"""

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
        
        for name, instance in self.tool_instances.items():
            funcs = ", ".join(d["function"]["name"] for d in instance.get_tool_definitions())
            table.add_row(name, funcs)
        
        console.print(table)
    
    def clear_history(self):
        self.history = []
    
    async def process(self, user_input: str) -> str:
        """Process user input through the agent loop."""
        
        self.history.append({"role": "user", "content": user_input})
        
        # Determine which model to use
        model = self._select_model(user_input)
        print(f"  [Model: {model}]", flush=True)
        
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history
        
        # Agent loop — keep going until LLM produces a final text response
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