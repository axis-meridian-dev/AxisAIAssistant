"""
Document Writer Tool — generates structured documents from research.

Creates essays, articles, legal briefs, research reports, and debate prep
documents. Pulls from the knowledge base and legal research library to
support claims with real references.
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Callable

from tools.base import BaseTool


OUTPUT_DIR = Path.home() / "LegalResearch" / "documents"


class DocumentWriterTool(BaseTool):
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.output_dir = Path(config.get("documents", {}).get(
            "output_dir", str(OUTPUT_DIR)
        ))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def write_document(self, title: str, content: str, 
                       doc_type: str = "article",
                       filename: str = None) -> str:
        """
        Write a formatted document to file.
        Types: article, essay, brief, memo, report, letter, debate_prep
        """
        if not filename:
            safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in title)
            safe_title = safe_title[:50].strip().replace(" ", "_")
            date = datetime.now().strftime("%Y%m%d")
            filename = f"{doc_type}_{safe_title}_{date}.md"
        
        filepath = self.output_dir / filename
        
        # Add document header
        header = (
            f"---\n"
            f"title: \"{title}\"\n"
            f"type: {doc_type}\n"
            f"date: {datetime.now().strftime('%Y-%m-%d')}\n"
            f"author: AI Research Assistant\n"
            f"status: draft\n"
            f"---\n\n"
        )
        
        with open(filepath, "w") as f:
            f.write(header)
            f.write(content)
        
        size = filepath.stat().st_size
        return (
            f"Document written: {filepath}\n"
            f"Type: {doc_type}\n"
            f"Size: {size / 1024:.1f} KB\n"
            f"Lines: {content.count(chr(10)) + 1}"
        )
    
    def write_debate_prep(self, topic: str, position: str,
                          arguments: str, counterarguments: str = "",
                          evidence: str = "", filename: str = None) -> str:
        """
        Create a structured debate/trial preparation document.
        Organizes arguments, counterarguments, evidence, and talking points.
        """
        if not filename:
            safe = "".join(c if c.isalnum() or c in " -_" else "" for c in topic)[:40]
            date = datetime.now().strftime("%Y%m%d")
            filename = f"debate_prep_{safe.replace(' ', '_')}_{date}.md"
        
        filepath = self.output_dir / filename
        
        doc = f"""---
title: "Debate Prep: {topic}"
type: debate_prep
date: {datetime.now().strftime('%Y-%m-%d')}
position: "{position}"
status: draft
---

# Debate Preparation: {topic}

## Position
{position}

## Core Arguments

{arguments}

## Anticipated Counterarguments & Rebuttals

{counterarguments if counterarguments else "[Add counterarguments and your rebuttals here]"}

## Supporting Evidence

{evidence if evidence else "[Add statistical data, case citations, and sources here]"}

## Key Talking Points

[Distill your strongest points into 3-5 concise statements]

1. 
2. 
3. 

## Opening Statement Framework

[Draft your opening here]

## Closing Statement Framework

[Draft your closing here]

## Quick Reference — Statutes & Cases

| Citation | Key Point |
|----------|-----------|
| | |

## Notes

[Additional preparation notes]

---
*Generated {datetime.now().strftime('%B %d, %Y')} — AI Research Assistant*
"""
        
        with open(filepath, "w") as f:
            f.write(doc)
        
        return f"Debate prep document created: {filepath}"
    
    def append_to_document(self, filepath: str, content: str,
                           section_header: str = None) -> str:
        """Append content to an existing document."""
        target = Path(filepath).expanduser().resolve()
        
        if not target.exists():
            return f"File not found: {target}"
        
        with open(target, "a") as f:
            if section_header:
                f.write(f"\n\n## {section_header}\n\n")
            f.write(content)
            f.write("\n")
        
        return f"Appended {len(content)} chars to {target}"
    
    def list_documents(self) -> str:
        """List all generated documents."""
        files = sorted(self.output_dir.glob("*.md"))
        
        if not files:
            return f"No documents yet. Directory: {self.output_dir}"
        
        lines = [f"Documents in {self.output_dir}:\n"]
        for f in files:
            size = f.stat().st_size / 1024
            mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            lines.append(f"  📄 {f.name}  ({size:.0f}KB, {mtime})")
        
        return "\n".join(lines)
    
    def get_tool_definitions(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "write_document",
                    "description": "Write a formatted document (article, essay, brief, memo, report, letter, debate_prep). Saves as markdown file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "Document title"},
                            "content": {"type": "string", "description": "Full document content in markdown format"},
                            "doc_type": {"type": "string", "enum": ["article", "essay", "brief", "memo", "report", "letter", "debate_prep"]},
                            "filename": {"type": "string", "description": "Custom filename (optional)"}
                        },
                        "required": ["title", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_debate_prep",
                    "description": "Create a structured debate or trial preparation document with sections for arguments, counterarguments, evidence, talking points, and opening/closing statements.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string", "description": "The debate/trial topic"},
                            "position": {"type": "string", "description": "Your position/thesis"},
                            "arguments": {"type": "string", "description": "Main arguments supporting the position"},
                            "counterarguments": {"type": "string", "description": "Anticipated opposing arguments with rebuttals"},
                            "evidence": {"type": "string", "description": "Supporting evidence, statistics, case citations"},
                            "filename": {"type": "string", "description": "Custom filename (optional)"}
                        },
                        "required": ["topic", "position", "arguments"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "append_to_document",
                    "description": "Add content to an existing document. Useful for building up research documents incrementally.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filepath": {"type": "string", "description": "Path to existing document"},
                            "content": {"type": "string", "description": "Content to append"},
                            "section_header": {"type": "string", "description": "Optional section header to add before content"}
                        },
                        "required": ["filepath", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_documents",
                    "description": "List all generated documents in the output directory.",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]
    
    def get_handlers(self) -> dict[str, Callable]:
        return {
            "write_document": self.write_document,
            "write_debate_prep": self.write_debate_prep,
            "append_to_document": self.append_to_document,
            "list_documents": self.list_documents,
        }