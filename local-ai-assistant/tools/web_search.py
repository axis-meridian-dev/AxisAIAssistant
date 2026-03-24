"""
Web Search Tool — internet research capability.

Uses SearXNG (local) with DuckDuckGo fallback.
Can also fetch and extract content from web pages.
"""

import json
import subprocess
from typing import Callable
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup

from tools.base import BaseTool


class WebSearchTool(BaseTool):
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.searxng_url = config.get("search", {}).get("searxng_url", "http://localhost:8888")
        self.fallback_ddg = config.get("search", {}).get("fallback_to_ddg", True)
        self.max_results = config.get("search", {}).get("max_results", 5)
    
    def web_search(self, query: str, max_results: int = 5) -> str:
        """Search the web using SearXNG or DuckDuckGo."""
        
        # Try SearXNG first
        try:
            result = self._searxng_search(query, max_results)
            if result:
                return result
        except Exception:
            pass
        
        # Fallback to DuckDuckGo
        if self.fallback_ddg:
            try:
                return self._ddg_search(query, max_results)
            except Exception as e:
                return f"Search failed: {e}"
        
        return "Search unavailable. SearXNG not running and DuckDuckGo fallback disabled."
    
    def fetch_webpage(self, url: str, extract_text: bool = True) -> str:
        """Fetch a webpage and extract its text content."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"
            }
            response = httpx.get(url, headers=headers, follow_redirects=True, timeout=15)
            response.raise_for_status()
            
            if not extract_text:
                return response.text[:10000]
            
            soup = BeautifulSoup(response.text, "lxml")
            
            # Remove script, style, nav, footer elements
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                tag.decompose()
            
            # Try to find main content
            main = soup.find("main") or soup.find("article") or soup.find("div", {"role": "main"})
            if main:
                text = main.get_text(separator="\n", strip=True)
            else:
                text = soup.get_text(separator="\n", strip=True)
            
            # Clean up excessive whitespace
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            text = "\n".join(lines)
            
            # Truncate if too long
            if len(text) > 8000:
                text = text[:8000] + "\n\n... [content truncated]"
            
            title = soup.title.string if soup.title else "No title"
            return f"Title: {title}\nURL: {url}\n{'─' * 40}\n{text}"
            
        except httpx.TimeoutException:
            return f"Timeout fetching: {url}"
        except Exception as e:
            return f"Error fetching {url}: {e}"
    
    def _searxng_search(self, query: str, max_results: int) -> str | None:
        """Search using local SearXNG instance."""
        url = f"{self.searxng_url}/search"
        params = {
            "q": query,
            "format": "json",
            "categories": "general",
        }
        
        response = httpx.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        results = data.get("results", [])[:max_results]
        if not results:
            return None
        
        lines = [f"Search results for: '{query}'\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            url = r.get("url", "")
            snippet = r.get("content", "No description")
            lines.append(f"{i}. **{title}**\n   {url}\n   {snippet}\n")
        
        return "\n".join(lines)
    
    def _ddg_search(self, query: str, max_results: int) -> str:
        """Search using DuckDuckGo (no API key needed)."""
        try:
            from duckduckgo_search import DDGS
            
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            
            if not results:
                return f"No results found for: '{query}'"
            
            lines = [f"Search results for: '{query}'\n"]
            for i, r in enumerate(results, 1):
                title = r.get("title", "No title")
                url = r.get("href", "")
                snippet = r.get("body", "No description")
                lines.append(f"{i}. **{title}**\n   {url}\n   {snippet}\n")
            
            return "\n".join(lines)
        except ImportError:
            return "DuckDuckGo search library not installed. Run: pip install duckduckgo-search"
    
    # ── Tool definitions ────────────────────────────────────────────────────
    
    def get_tool_definitions(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the internet for information. Returns titles, URLs, and snippets from top results. Use for current events, technical questions, research.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "max_results": {"type": "integer", "description": "Number of results (default: 5)"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_webpage",
                    "description": "Fetch a webpage URL and extract its text content. Use to read articles, documentation, or any web page in detail.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "Full URL to fetch"},
                            "extract_text": {"type": "boolean", "description": "Extract clean text (true) or raw HTML (false)"}
                        },
                        "required": ["url"]
                    }
                }
            }
        ]
    
    def get_handlers(self) -> dict[str, Callable]:
        return {
            "web_search": self.web_search,
            "fetch_webpage": self.fetch_webpage,
        }
