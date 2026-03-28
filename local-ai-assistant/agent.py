"""
Core Agent — LLM reasoning + tool dispatch loop.

Uses Ollama's tool-calling API to let the LLM decide which tools to invoke.
Runs a loop: LLM → tool call → result → LLM → ... until final answer.

Architecture layers:
  1. Intent Detection — classify user input, with context-aware inheritance
  2. Tool Gate — safety tiers control which tools require confirmation
  3. Mode System — stabilizes behavior per task type (sticky for legal)
  4. Forced Retrieval — cascading fallback chain (KB → statutes → case law → web)
  5. Anchor Precedent Injection — mandatory cases for recognized legal patterns
  6. Fact Extraction — structured fact parsing for legal scenarios
  7. Two-Pass Reasoning — retrieval then structured analysis with fact-to-law mapping
  8. Citation + Confidence Validation — hard-fail post-processing check
  9. Output Sanitization — strip raw tool JSON from final response
  10. Source Trace — transparency layer for debugging and audit
"""

import re
import json
import time
import ollama
from datetime import datetime
from rich.console import Console
from rich.table import Table

from tools.file_manager import FileManagerTool
from tools.web_search import WebSearchTool
from tools.desktop_control import DesktopControlTool
from tools.system_info import SystemInfoTool
from tools.knowledge_base import KnowledgeBaseTool
from tools.legal_research import LegalResearchTool
from tools.document_writer import DocumentWriterTool
from stats import InquiryStats, StatsTimer, SessionStats

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

# Follow-up phrases that indicate the user is continuing the previous topic
FOLLOWUP_PHRASES = [
    "same scenario", "same case", "same situation", "same facts",
    "what if", "now assume", "now suppose", "instead",
    "but what if", "but now", "but assume", "but instead",
    "change the facts", "modify the scenario", "alternatively",
    "in that case", "given that", "under those",
    "how would that change", "would that affect",
    "what about", "and if",
]


def is_followup(user_input: str) -> bool:
    """Detect if the user's input is a follow-up to the previous topic."""
    lower = user_input.lower()
    return any(phrase in lower for phrase in FOLLOWUP_PHRASES)


def detect_intent(user_input: str, last_intent: str = "general") -> str:
    """
    Classify user intent to determine routing behavior.
    Uses a two-tier keyword system with context-aware inheritance:
    if the input is a follow-up and no new intent is detected, inherit last_intent.

    Returns: 'legal', 'research', 'general'
    """
    lower = user_input.lower()

    # Tier 1: Primary legal keywords — high confidence
    if any(kw in lower for kw in LEGAL_KEYWORDS_PRIMARY):
        return "legal"

    # Tier 2: Fallback legal keywords — broader catch
    fallback_hits = sum(1 for kw in LEGAL_KEYWORDS_FALLBACK if kw in lower)
    if fallback_hits >= 2:
        return "legal"

    # Single fallback hit in combination with research keywords = legal
    if fallback_hits >= 1 and any(kw in lower for kw in RESEARCH_KEYWORDS):
        return "legal"

    # Research intent
    if any(kw in lower for kw in RESEARCH_KEYWORDS):
        return "research"

    # Context-aware inheritance: follow-up queries inherit previous intent
    if is_followup(lower) and last_intent in ("legal", "research"):
        return last_intent

    return "general"


# ── Anchor Precedents ─────────────────────────────────────────────────────
# Mandatory cases that MUST be included for recognized legal patterns.
# These are injected into the system prompt when the pattern is detected.

ANCHOR_PRECEDENTS = {
    "excessive_force": {
        "trigger_keywords": [
            "excessive force", "use of force", "police force", "officer force",
            "physical force", "restrain", "taser", "chokehold", "brutality",
        ],
        "cases": [
            "Graham v. Connor, 490 U.S. 386 (1989) — The controlling standard for excessive force claims under the Fourth Amendment. Force must be judged by 'objective reasonableness' considering: (1) severity of the crime, (2) whether the suspect poses an immediate threat, (3) whether the suspect is actively resisting or attempting to flee.",
        ],
        "statutes": [
            "42 U.S.C. § 1983 — Civil action for deprivation of rights under color of law.",
        ],
        "force_classification": (
            "FORCE SEVERITY CLASSIFICATION (you MUST identify which applies):\n"
            "  - MINIMAL: verbal commands, handcuffing a compliant suspect\n"
            "  - MODERATE: takedowns, strikes, OC spray, control holds\n"
            "  - SERIOUS: Taser, baton strikes, K-9 deployment\n"
            "  - DEADLY: firearm, chokehold, vehicle ramming\n"
            "Only cite Tennessee v. Garner if DEADLY force is at issue.\n"
            "Only cite Graham v. Connor factors — do NOT introduce unrelated constitutional amendments "
            "(e.g., do NOT use 14th Amendment substantive due process for a seizure/stop scenario)."
        ),
    },
    "qualified_immunity": {
        "trigger_keywords": [
            "qualified immunity", "clearly established",
        ],
        "cases": [
            "Harlow v. Fitzgerald, 457 U.S. 800 (1982) — Established the qualified immunity defense: officials are shielded unless their conduct violates 'clearly established' law.",
            "Ashcroft v. al-Kidd, 563 U.S. 731 (2011) — The right must be clearly established at the time of the challenged conduct, with sufficient specificity.",
        ],
        "statutes": [],
        "force_classification": None,
    },
    "search_seizure": {
        "trigger_keywords": [
            "search and seizure", "warrantless search", "terry stop",
            "probable cause", "reasonable suspicion",
        ],
        "cases": [
            "Terry v. Ohio, 392 U.S. 1 (1968) — Police may conduct a brief stop if they have reasonable suspicion of criminal activity. Pat-down permitted if officer reasonably believes suspect is armed.",
            "Mapp v. Ohio, 367 U.S. 643 (1961) — Evidence obtained in violation of the Fourth Amendment is inadmissible (exclusionary rule).",
        ],
        "statutes": [],
        "force_classification": None,
    },
    "vehicle_search": {
        "trigger_keywords": [
            "vehicle search", "car search", "automobile exception", "car crash",
            "crashed car", "crashed vehicle", "impound", "inventory search",
            "drugs in car", "weapon in car", "unregistered vehicle",
            "bill of sale", "vehicle found",
        ],
        "cases": [
            "Carroll v. United States, 267 U.S. 132 (1925) — The automobile exception allows warrantless search of a vehicle if there is probable cause to believe it contains contraband or evidence of a crime.",
            "Arizona v. Gant, 556 U.S. 332 (2009) — Vehicle search incident to arrest is limited: police may search only if arrestee is unsecured and within reaching distance, or if it is reasonable to believe the vehicle contains evidence of the offense of arrest.",
            "South Dakota v. Opperman, 428 U.S. 364 (1976) — Inventory searches of lawfully impounded vehicles are permissible without a warrant for caretaking purposes.",
            "Florida v. Wells, 495 U.S. 1 (1990) — Inventory search must follow standardized police department policy; cannot be used as pretext for investigation.",
        ],
        "statutes": [],
        "force_classification": None,
    },
    "home_arrest": {
        "trigger_keywords": [
            "arrested at home", "arrest in home", "arrest at house",
            "arrested in basement", "home entry", "entered home",
            "warrantless arrest home", "knocked on door",
        ],
        "cases": [
            "Payton v. New York, 445 U.S. 573 (1980) — The Fourth Amendment prohibits warrantless, nonconsensual entry into a suspect's home to make a routine felony arrest. Absent exigent circumstances or consent, an arrest warrant is required.",
            "Kirk v. Louisiana, 536 U.S. 635 (2002) — Reaffirmed Payton: police may not enter a home to make a warrantless arrest even with probable cause, absent exigent circumstances.",
            "Welsh v. Wisconsin, 466 U.S. 740 (1984) — When the underlying offense is minor, the exigent circumstances exception to the warrant requirement is especially disfavored for home entry.",
        ],
        "statutes": [],
        "force_classification": None,
    },
    "constructive_possession": {
        "trigger_keywords": [
            "constructive possession", "drugs found in car", "possession",
            "not on person", "found in vehicle", "drugs in glove",
            "paraphernalia", "drug possession",
        ],
        "cases": [
            "Illinois v. Gates, 462 U.S. 213 (1983) — Totality of the circumstances test for evaluating probable cause; replaced rigid two-pronged Aguilar-Spinelli test.",
            "Maryland v. Pringle, 540 U.S. 366 (2003) — Drugs found in a vehicle can support probable cause to arrest all occupants, but officers must articulate a reasonable basis linking the suspect to the contraband.",
        ],
        "statutes": [],
        "force_classification": None,
    },
    "weapon_classification": {
        "trigger_keywords": [
            "dangerous weapon", "weapon in vehicle", "pepper spray",
            "gun-shaped", "imitation weapon", "weapon charge",
            "felony weapon", "53a-217",
        ],
        "cases": [
            "District of Columbia v. Heller, 554 U.S. 570 (2008) — Second Amendment protects individual right to bear arms; relevant to statutory interpretation of 'weapon' definitions.",
        ],
        "statutes": [
            "CT Gen Stat § 53-206 — Carrying of dangerous weapons: defines what constitutes a 'dangerous weapon' under Connecticut law. Pepper spray/mace generally NOT classified as a dangerous weapon if carried for self-defense.",
            "CT Gen Stat § 53a-3(6) — Definition of 'dangerous weapon': any instrument, article or substance which, under the circumstances in which it is used or attempted or threatened to be used, is capable of causing death or serious physical injury.",
            "CT Gen Stat § 29-38 — Weapons in vehicles: prohibits carrying certain weapons in motor vehicles. Applies to pistols, revolvers, and other specific weapon types. Pepper spray may NOT qualify.",
        ],
        "force_classification": None,
    },
}


def detect_anchor_precedents(user_input: str) -> list[dict]:
    """Detect which anchor precedent sets should be injected based on input."""
    lower = user_input.lower()
    matched = []
    for pattern_name, pattern_data in ANCHOR_PRECEDENTS.items():
        if any(kw in lower for kw in pattern_data["trigger_keywords"]):
            matched.append(pattern_data)
    return matched


def build_anchor_injection(anchors: list[dict]) -> str:
    """Build the anchor precedent injection block for the system prompt."""
    if not anchors:
        return ""

    lines = [
        "\n═══════════════════════════════════════════════════════════",
        "MANDATORY ANCHOR PRECEDENTS (MUST INCLUDE IN YOUR RESPONSE)",
        "═══════════════════════════════════════════════════════════\n",
    ]

    for anchor in anchors:
        if anchor["cases"]:
            lines.append("CONTROLLING CASES (you MUST cite these):")
            for case in anchor["cases"]:
                lines.append(f"  - {case}")

        if anchor["statutes"]:
            lines.append("CONTROLLING STATUTES:")
            for statute in anchor["statutes"]:
                lines.append(f"  - {statute}")

        if anchor.get("force_classification"):
            lines.append(f"\n{anchor['force_classification']}")

    lines.append(
        "\nYou MUST include at least ONE of the above cases in your response. "
        "You MUST include at least ONE supporting circuit-level or lower court case. "
        "Do NOT rely solely on the anchor precedents — supplement with jurisdiction-specific authority."
    )

    return "\n".join(lines)


# ── Output Sanitization ───────────────────────────────────────────────────

def sanitize_output(text: str) -> str:
    """
    Strip raw tool call JSON that leaks into LLM output.
    Some models emit tool-call-like JSON blocks in their text responses.
    """
    # Remove JSON blocks that look like tool calls: {"name": "...", "arguments": ...}
    text = re.sub(
        r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"(?:arguments|parameters)"\s*:\s*\{[^}]*\}\s*\}',
        '', text
    )
    # Remove Ollama-style tool blocks: {"function": {"name": ...}}
    text = re.sub(
        r'\{\s*"(?:type"\s*:\s*"function"\s*,\s*")?\s*function"\s*:\s*\{[^}]*\}\s*\}',
        '', text
    )
    # Clean up any resulting empty lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


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
        "LEGAL ISSUES:\n[Numbered list of every distinct legal question raised by the facts]\n\n"
        "CHARGES BREAKDOWN:\n[For each charge: elements required, what prosecution must prove, statutory basis]\n\n"
        "CONSTITUTIONAL ANALYSIS:\n"
        "- Search (vehicle): [Was the search lawful? What exception applies?]\n"
        "- Seizure (evidence): [Chain of custody, nexus to suspect]\n"
        "- Arrest (home entry): [Warrant? Exigent circumstances? Consent?]\n\n"
        "PROSECUTION THEORY:\n[Numbered points with citations — how prosecution builds its case]\n\n"
        "DEFENSE THEORY:\n[Numbered points with citations — strongest defense arguments]\n\n"
        "MULTI-PERSPECTIVE ANALYSIS:\n"
        "- Police perspective: [Their reasoning, what they relied on, potential procedural gaps]\n"
        "- Suspect perspective: [Constitutional rights, plausible alternative narratives]\n"
        "- Defense attorney strategy: [Motions to file, evidence to suppress, arguments to make]\n"
        "- Prosecutor strategy: [How to build circumstantial case, key evidence to emphasize]\n"
        "- Judge considerations: [Evidentiary rulings, constitutional scrutiny, sentencing factors]\n"
        "- Jury perception: [What a reasonable juror would infer, bias risks, emotional weight]\n"
        "- Public/media narrative: [How this plays in the press, reputation impact, public interest]\n\n"
        "KEY PRECEDENTS:\n[Case name — holding — which side it favors]\n\n"
        "CRITICAL WEAKNESSES:\n"
        "- In prosecution case: [Biggest vulnerabilities]\n"
        "- In defense case: [Biggest vulnerabilities]\n\n"
        "EXPANSION POINTS:\n"
        "- Possible additional charges: [What else could be charged?]\n"
        "- Possible suppression arguments: [What evidence could be excluded?]\n"
        "- Civil rights claims (if any): [Potential § 1983 or other claims]\n\n"
        "FOLLOW-UP QUESTIONS:\n[Numbered list of factual questions that would change the analysis]\n\n"
        "LIKELY OUTCOME:\n[Clear directional assessment based on weight of authority]\n\n"
        "Do NOT deviate from this structure. Every claim must cite authority.\n"
        "Do NOT assume police actions are valid — scrutinize them equally.\n"
        "Do NOT use weak placeholder arguments for defense — make them as strong as possible."
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
  Step 1: IDENTIFY the legal issue(s) — state them precisely. List EVERY distinct issue.
  Step 2: MAP relevant law — which statutes and cases apply, and why.
    CRITICAL: Verify statute applicability. Do NOT cite a firearm statute for non-firearm objects.
    Do NOT cite an irrelevant statute just because it appeared in search results.
  Step 3: FACT-TO-LAW COMPARISON — for EACH key fact, explain how it changes the legal analysis:
    Example for excessive force:
    - Resistance level: [none/passive/active] → [how this affects Graham reasonableness]
    - Threat level: [none/minimal/significant/imminent] → [how this affects justification]
    - Crime severity: [minor infraction/misdemeanor/felony] → [how this weighs]
    Example for vehicle/arrest scenario:
    - Nexus: [What links suspect to vehicle? Bill of sale ≠ driver at time of crash]
    - Home entry: [Was there a warrant? Consent? Exigent circumstances?]
    - Constructive possession: [Was suspect in dominion/control of contraband?]
    - Weapon classification: [Does the object meet the statutory definition?]
    You MUST show how changing ONE fact would change the outcome.
  Step 4: ADVERSARIAL ANALYSIS — you MUST analyze from ALL perspectives:
    a) PROSECUTION: Build the strongest possible case. Identify every inference favorable to state.
    b) DEFENSE: Build the strongest possible defense. Challenge EVERY element.
       Defense arguments MUST be as strong and realistic as possible, not placeholders.
    c) POLICE: What was their reasoning? Where are procedural gaps?
    d) JUDGE: What evidentiary rulings are likely? What constitutional issues will be scrutinized?
    e) JURY: What would a reasonable juror infer? What biases might affect perception?
    f) PUBLIC/MEDIA: How does this play in the press? What narrative forms?
  Step 5: SCRUTINIZE — you must aggressively challenge:
    - Probable cause gaps (is the evidence actually sufficient?)
    - Nexus between suspect and evidence (ownership ≠ possession ≠ use)
    - Warrant requirements (especially home entry under Payton v. New York)
    - Misapplied or weak statutes (is the charge actually supported by the statute cited?)
    - Overcharging (are felony charges appropriate for the conduct?)
    You must NOT assume police actions are valid. Challenge both sides equally.
  Step 6: EXPAND — identify what's missing:
    - Additional charges that could apply
    - Additional defenses not yet raised
    - Suppression arguments (4th Amendment violations)
    - Civil rights claims (§ 1983 if applicable)
    - Follow-up factual questions that would change the analysis
  Step 7: CONCLUDE — state your conclusion using this EXACT format:

LIKELY OUTCOME:
[Clear, directional assessment — not "it depends." State which side has the stronger position and WHY.]

CONFIDENCE: High / Medium / Low

REASONING:
- Strength of precedent: [strong/moderate/weak — are there on-point SCOTUS/Circuit cases?]
- Jurisdiction match: [does the authority come from the relevant jurisdiction?]
- Factual similarity: [how closely do cited cases match the current facts?]
- Source conflicts: [do any authorities point in different directions?]

CRITICAL RULES:
- Only include legal doctrines DIRECTLY applicable to the facts.
- Do NOT introduce unrelated constitutional amendments.
- Include at least 1 controlling precedent (SCOTUS) AND 1 supporting case (circuit or lower).
- If deadly force is NOT at issue, do NOT cite Tennessee v. Garner.

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


CONFIDENCE_HARD_REJECT = (
    "**Response rejected: missing confidence scoring.**\n\n"
    "The system requires a structured confidence assessment for all legal responses.\n\n"
    "Expected format:\n"
    "```\n"
    "CONFIDENCE: High / Medium / Low\n"
    "REASONING:\n"
    "- Strength of precedent: ...\n"
    "- Jurisdiction match: ...\n"
    "- Factual similarity: ...\n"
    "- Source conflicts: ...\n"
    "```\n\n"
    "Re-run your query with 'argument mode' or 'analysis mode' for full structured output.\n\n"
    "Note: This is AI-generated legal research, not legal advice. "
    "Verify all citations independently."
)


# ── Confidence Auto-Patch ─────────────────────────────────────────────────

# High-authority cases that indicate strong grounding when cited
HIGH_AUTHORITY_CASES = [
    "Graham v. Connor", "Terry v. Ohio", "Mapp v. Ohio",
    "Harlow v. Fitzgerald", "Tennessee v. Garner", "Payton v. New York",
    "Carroll v. United States", "Miranda v. Arizona",
    "Ashcroft v. al-Kidd", "Pearson v. Callahan",
]


def _auto_patch_confidence(response_text: str) -> str:
    """
    Instead of rejecting a response missing CONFIDENCE/REASONING,
    dynamically compute and append a confidence block based on
    what the response actually contains.

    Scoring:
      - SCOTUS case cited: +2 per case (max 6)
      - Any case citation: +1 per case (max 3)
      - Statute cited: +1 per statute (max 2)
      - Has LIKELY OUTCOME section: +1
      - Has APPLICATION section: +1
      Total >= 8 → High, >= 4 → Medium, else Low
    """
    score = 0
    reasons = {}

    # Count high-authority (SCOTUS) cases
    scotus_hits = sum(1 for c in HIGH_AUTHORITY_CASES if c in response_text)
    score += min(scotus_hits * 2, 6)
    reasons["strength_of_precedent"] = (
        "strong" if scotus_hits >= 2 else "moderate" if scotus_hits >= 1 else "weak"
    )

    # Count any case citations
    all_cases = re.findall(r'[A-Z][a-z]+\s+v\.\s+[A-Z][a-z]+', response_text)
    score += min(len(all_cases), 3)

    # Count statute citations
    statutes = re.findall(
        r'\d+\s*U\.?S\.?C\.?\s*§?\s*\d+|\bSection\s+\d+|CT Gen Stat', response_text
    )
    score += min(len(statutes), 2)

    # Structural completeness
    if re.search(r'LIKELY\s+OUTCOME', response_text, re.IGNORECASE):
        score += 1
    if re.search(r'APPLICATION', response_text, re.IGNORECASE):
        score += 1

    # Determine confidence level
    if score >= 8:
        confidence = "High"
    elif score >= 4:
        confidence = "Medium"
    else:
        confidence = "Low"

    # Jurisdiction reasoning — check for jurisdiction-specific markers
    has_jurisdiction = bool(re.search(
        r'Circuit|District|State|Connecticut|Federal|Supreme Court', response_text
    ))
    reasons["jurisdiction_match"] = "yes" if has_jurisdiction else "partial — verify jurisdiction applicability"

    # Factual similarity — heuristic based on APPLICATION section depth
    app_match = re.search(r'APPLICATION.*?(?=\n[A-Z]{3,}|\Z)', response_text, re.DOTALL | re.IGNORECASE)
    app_length = len(app_match.group()) if app_match else 0
    reasons["factual_similarity"] = (
        "high" if app_length > 500 else "moderate" if app_length > 200 else "low"
    )

    # Source conflicts
    has_conflict_language = bool(re.search(
        r'however|but see|contra|dissent|split|conflict', response_text, re.IGNORECASE
    ))
    reasons["source_conflicts"] = "minor — noted in analysis" if has_conflict_language else "none identified"

    # Build the patch block
    patch = (
        f"\n\nCONFIDENCE: {confidence}\n\n"
        f"REASONING:\n"
        f"- Strength of precedent: {reasons['strength_of_precedent']}\n"
        f"- Jurisdiction match: {reasons['jurisdiction_match']}\n"
        f"- Factual similarity: {reasons['factual_similarity']}\n"
        f"- Source conflicts: {reasons['source_conflicts']}\n"
    )

    return response_text + patch


def validate_legal_response(response, intent: str) -> str:
    """
    Post-processing HARD enforcement for legal responses:
    1. Normalize input (handle str, dict, None, or other types)
    2. Citation check — no citations → hard fail (replace response)
    3. Confidence check — no confidence block → append warning
    4. Disclaimer check — ensure present
    5. Output sanitization — strip leaked tool JSON
    """
    # ── Normalize input type ───────────────────────────────────────
    response_text = None
    if isinstance(response, str):
        response_text = response
    elif isinstance(response, dict):
        response_text = response.get("content", "") or ""
    elif response is None:
        response_text = ""
    else:
        response_text = str(response)

    # ── Non-legal intent: just sanitize and return ─────────────────
    if intent != "legal":
        return sanitize_output(response_text)

    # ── Guard clause: empty or trivially short ─────────────────────
    if not response_text or len(response_text.strip()) < 50:
        return sanitize_output(response_text) if response_text else ""

    # Sanitize first — remove any leaked tool JSON
    response_text = sanitize_output(response_text)

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

    # Check for confidence block — AUTO-PATCH if missing
    has_confidence = bool(re.search(
        r'CONFIDENCE\s*:\s*(?:High|Medium|Low)', response_text, re.IGNORECASE
    ))
    if not has_confidence:
        response_text = _auto_patch_confidence(response_text)

    # Ensure disclaimer is present
    if "not legal advice" not in response_text.lower():
        response_text += "\n\n---\n" + DISCLAIMER

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

Think step by step about which tools to use, then use them.

═══════════════════════════════════════════════════════════
REQUIRED OUTPUT FORMAT (MANDATORY — LEGAL RESPONSES ONLY)
═══════════════════════════════════════════════════════════

For ALL legal responses, you MUST include EVERY section below.
If ANY section is missing, your response is INVALID and will be rejected by the system.

LEGAL ISSUES:
[Numbered list of every distinct legal question raised by the facts]

CHARGES BREAKDOWN:
[For each charge: elements the prosecution must prove, statutory basis, potential issues with the charge]

CONSTITUTIONAL ANALYSIS:
- Search: [Was any search lawful? What exception applies or was violated?]
- Seizure: [Was evidence properly seized? Chain of custody issues?]
- Arrest: [Was the arrest lawful? Warrant? Exigent circumstances? Home entry issues?]

APPLICABLE LAW:
- [Statute citation] — [Short description]
IMPORTANT: Verify each statute actually applies to the facts. Do NOT cite firearm statutes for non-firearms.
Do NOT cite statutes that are tangentially related but do not match the actual conduct.

CASE LAW:
- [Case name], [Volume] [Reporter] [Page] ([Court] [Year]) — [Holding]

PROSECUTION THEORY:
[How the prosecution builds its case — key evidence, inferences, chain of reasoning]

DEFENSE THEORY:
[Strongest defense arguments — constitutional challenges, element failures, alternative explanations]
Defense arguments MUST be as strong and realistic as possible. Do NOT use placeholders.

MULTI-PERSPECTIVE ANALYSIS:
- Police perspective: [Their reasoning, what they relied on, procedural strengths/gaps]
- Suspect perspective: [Constitutional rights at stake, plausible alternative narratives]
- Defense attorney strategy: [Motions to file, evidence to suppress, trial strategy]
- Prosecutor strategy: [How to overcome defense objections, key evidence to emphasize]
- Judge considerations: [Evidentiary rulings, constitutional scrutiny, sentencing factors]
- Jury perception: [What reasonable jurors would infer, bias risks, emotional weight of evidence]
- Public/media narrative: [How this case plays in the press, reputation impact]

KEY PRECEDENTS:
[Case name — holding — which side it favors — why it matters here]

CRITICAL WEAKNESSES:
- In prosecution case: [Specific vulnerabilities — missing evidence, constitutional issues, weak nexus]
- In defense case: [Specific vulnerabilities — damaging evidence, unfavorable inferences]

EXPANSION POINTS:
- Possible additional charges: [What else could be charged based on facts?]
- Possible additional defenses: [What other defenses could be raised?]
- Possible suppression arguments: [What evidence could be excluded and why?]
- Civil rights claims (if any): [Potential § 1983 or other claims against officers]

FOLLOW-UP QUESTIONS:
[Numbered list of factual questions whose answers would change the legal analysis]

LIKELY OUTCOME:
[Clear directional assessment — state which side has the stronger position and WHY]

CONFIDENCE: High / Medium / Low

REASONING:
- Strength of precedent: [strong/moderate/weak]
- Jurisdiction match: [yes/partial/no]
- Factual similarity: [high/moderate/low]
- Source conflicts: [none/minor/significant]

Note: This is AI-generated legal research, not legal advice. Verify all citations independently.

═══════════════════════════════════════════════════════════
ADVERSARIAL ANALYSIS RULES (MANDATORY)
═══════════════════════════════════════════════════════════

You must aggressively scrutinize:
- Probable cause gaps — is the evidence actually sufficient for each charge?
- Nexus between suspect and evidence — ownership ≠ possession ≠ use ≠ driving
- Warrant requirements — especially home entry (Payton v. New York)
- Misapplied or weak statutes — does the charged statute actually cover this conduct?
- Overcharging — are felony charges appropriate or is this charge stacking?

You must NOT assume police actions are valid. Challenge them.
You must NOT assume prosecution theory is correct. Test every element.
You must challenge BOTH prosecution AND defense equally.
You must identify what facts are MISSING and what questions remain unanswered.

DO NOT skip any section. DO NOT merge sections. Output each heading exactly as shown."""


class Agent:
    def __init__(self, config: dict):
        self.config = config
        self.history: list[dict] = []
        self.ollama_host = config["llm"]["ollama_host"]
        self.mode = "general"       # Current agent mode
        self.last_intent = "general" # Intent inheritance for follow-ups
        self.legal_locked = False    # Sticky legal session flag

        # Performance tracking
        self.session_stats = SessionStats()
        self.last_inquiry_stats: InquiryStats | None = None

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
        self.last_intent = "general"
        self.legal_locked = False

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
        """Process user input through the full agent pipeline."""
        process_start = time.perf_counter()

        # Initialize inquiry stats
        inquiry = InquiryStats(
            query=user_input[:200],
            timestamp=datetime.now().isoformat(),
        )

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

        # ── Step 1: Intent detection (with inheritance) ────────────────
        intent = detect_intent(user_input, last_intent=self.last_intent)

        # Sticky legal lock: once a conversation enters legal territory,
        # ALL subsequent queries are treated as legal until 'clear'
        if intent == "legal":
            self.legal_locked = True
        if self.legal_locked and intent == "general":
            intent = "legal"
            print(f"  [Legal Lock] Session is legal-locked — forcing legal intent", flush=True)

        # Force mode escalation: legal intent in general mode → analysis
        if intent == "legal" and self.mode == "general":
            self.mode = "analysis"
            print(f"  [Mode Escalation] legal intent forced mode general → analysis", flush=True)

        # Store for next turn's inheritance
        self.last_intent = intent
        inquiry.intent = intent
        inquiry.mode = self.mode

        if intent != "general":
            print(f"  [Intent: {intent}] [Mode: {self.mode}] [Legal Lock: {self.legal_locked}]", flush=True)

        self.history.append({"role": "user", "content": user_input})

        # Determine which model to use
        model = self._select_model(user_input)
        inquiry.model = model
        print(f"  [Model: {model}]", flush=True)

        # Build message list with system prompt + mode instruction + reasoning framework
        mode_instruction = MODE_INSTRUCTIONS.get(self.mode, "")
        system_content = SYSTEM_PROMPT + mode_instruction

        # Two-pass reasoning framework for legal intents
        if intent == "legal":
            system_content += TWO_PASS_LEGAL_FRAMEWORK

        # ── Step 1.5: Anchor precedent injection ───────────────────────
        anchors = detect_anchor_precedents(user_input)
        # Also check conversation history for anchor triggers (follow-ups)
        if not anchors and len(self.history) >= 2:
            for msg in self.history[-4:]:
                if msg.get("role") in ("user", "assistant"):
                    anchors = detect_anchor_precedents(msg.get("content", ""))
                    if anchors:
                        break

        anchor_injection = build_anchor_injection(anchors)
        if anchor_injection:
            system_content += anchor_injection
            print(f"  [Anchor Precedents] Injecting {len(anchors)} anchor set(s)", flush=True)

        messages = [{"role": "system", "content": system_content}] + self.history

        # ── Step 2: Fact extraction for legal scenarios ────────────────
        fact_timer = StatsTimer()
        fact_timer.start()
        extracted_facts = self._extract_facts(user_input, intent)
        inquiry.fact_extraction_time = fact_timer.stop()
        if extracted_facts:
            inquiry.llm_calls += 1  # fact extraction uses fast model
            messages.append({
                "role": "system",
                "content": (
                    "[STRUCTURED FACTS — extracted from user's scenario]\n"
                    "Use these structured facts to guide your legal analysis. "
                    "Do not re-interpret the facts; work from this extraction.\n\n"
                    + extracted_facts
                )
            })

        # ── Step 3: Forced retrieval (cascading fallback chain) ────────
        retrieval_timer = StatsTimer()
        retrieval_timer.start()
        messages, retrieval_trace = self._force_retrieval(user_input, intent, messages)
        inquiry.retrieval_time = retrieval_timer.stop()

        # ── Step 3.5: Final compliance constraint for legal ─────────────
        if intent == "legal":
            messages.append({
                "role": "system",
                "content": (
                    "FINAL INSTRUCTION (NON-NEGOTIABLE): You MUST end your response with "
                    "a CONFIDENCE and REASONING section. Use this exact format:\n\n"
                    "CONFIDENCE: High / Medium / Low\n\n"
                    "REASONING:\n"
                    "- Strength of precedent: [strong/moderate/weak]\n"
                    "- Jurisdiction match: [yes/partial/no]\n"
                    "- Factual similarity: [high/moderate/low]\n"
                    "- Source conflicts: [none/minor/significant]\n\n"
                    "If you omit this section, your response will be automatically corrected by the system."
                )
            })

        # ── Step 4: Agent loop ─────────────────────────────────────────
        tools_invoked = []  # Track for source trace
        max_iterations = 10
        for i in range(max_iterations):
            print(f"  [Step {i+1}] Sending to LLM...", flush=True)

            try:
                llm_timer = StatsTimer()
                llm_timer.start()
                response = ollama.chat(
                    model=model,
                    messages=messages,
                    tools=self.tools,
                    options={
                        "temperature": self.config["llm"]["temperature"],
                        "num_ctx": self.config["llm"]["context_window"],
                    }
                )
                inquiry.llm_time += llm_timer.stop()
                inquiry.llm_calls += 1
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

                    # Execute the tool with timing
                    handler = self.tool_handlers.get(func_name)
                    tool_timer = StatsTimer()
                    tool_timer.start()
                    if handler:
                        try:
                            result = handler(**func_args)
                            result_str = str(result)
                            if len(result_str) > 200:
                                print(f"    ✓ Got result ({len(result_str)} chars)", flush=True)
                            else:
                                print(f"    ✓ {result_str}", flush=True)
                        except Exception as e:
                            result = f"Tool error: {e}"
                            result_str = str(result)
                            print(f"    ✗ {result}", flush=True)
                    else:
                        result = f"Unknown tool: {func_name}"
                        result_str = str(result)
                        print(f"    ✗ {result}", flush=True)
                    inquiry.tool_time += tool_timer.stop()

                    # Track tool usage stats
                    inquiry.tools_called.append(func_name)
                    inquiry.tool_call_count += 1
                    data_type = SessionStats.classify_tool_data(func_name)
                    if data_type == "online":
                        inquiry.online_data_chars += len(result_str)
                    else:
                        inquiry.offline_data_chars += len(result_str)

                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "content": result_str
                    })
            else:
                # No tool calls — this is the final response
                # Guard: Ollama can return None for content even with default
                final = msg.get("content") or ""
                if not isinstance(final, str):
                    final = str(final)

                # ── Step 5: Citation + confidence validation ───────────
                validated = validate_legal_response(final, intent)

                # ── Step 5b: Retry loop on citation failure ─────────────
                # If the response has NO citations (hard fail), give the LLM
                # one chance to fix it. Confidence is now auto-patched, not rejected.
                is_rejected = (
                    intent == "legal"
                    and validated == HARD_FAIL_RESPONSE
                )
                if is_rejected and not getattr(self, '_retry_attempted', False):
                    self._retry_attempted = True
                    inquiry.retry_triggered = True
                    rejection_reason = "missing legal citations"
                    print(f"  [Validation] Response rejected: {rejection_reason} — retrying...", flush=True)

                    # Add correction instruction and loop back
                    messages.append({
                        "role": "system",
                        "content": (
                            f"[SYSTEM REJECTION] Your previous response was rejected because: {rejection_reason}.\n"
                            "You MUST fix this. Re-generate your response and include ALL required sections:\n"
                            "LEGAL ISSUES, CHARGES BREAKDOWN, CONSTITUTIONAL ANALYSIS, "
                            "PROSECUTION THEORY, DEFENSE THEORY, MULTI-PERSPECTIVE ANALYSIS, "
                            "KEY PRECEDENTS, CRITICAL WEAKNESSES, EXPANSION POINTS, "
                            "FOLLOW-UP QUESTIONS, LIKELY OUTCOME, CONFIDENCE (High/Medium/Low), REASONING.\n"
                            "Use the tools (lookup_statute, search_case_law) if you need sources.\n"
                            "Do NOT skip any section."
                        )
                    })
                    continue  # Go back to top of agent loop for retry

                # Reset retry flag for next query
                self._retry_attempted = False
                final = validated

                # Track validation outcome
                inquiry.validation_passed = validated not in (HARD_FAIL_RESPONSE, CONFIDENCE_HARD_REJECT)

                # ── Step 6: Append source trace for legal responses ────
                if intent == "legal":
                    source_trace = self._build_source_trace(retrieval_trace, tools_invoked)
                    if source_trace:
                        final += source_trace

                print(f"  [Done] Got final response ({len(final)} chars)", flush=True)
                self.history.append({"role": "assistant", "content": final})

                # Trim history to prevent context overflow
                if len(self.history) > 40:
                    self.history = self.history[-30:]

                # ── Record performance stats ──────────────────────────
                inquiry.total_time = time.perf_counter() - process_start
                inquiry.response_chars = len(final)
                self.session_stats.record_inquiry(inquiry)
                self.last_inquiry_stats = inquiry

                return final

        # Max iterations reached — still record stats
        inquiry.total_time = time.perf_counter() - process_start
        self.session_stats.record_inquiry(inquiry)
        self.last_inquiry_stats = inquiry
        return "Reached maximum tool iterations. Please try a simpler request."

    def _select_model(self, user_input: str) -> str:
        """Always use primary model. Fast model routing disabled
        until a larger GPU can handle tool definitions properly."""
        return self.config["llm"]["primary_model"]
