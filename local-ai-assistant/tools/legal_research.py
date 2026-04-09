"""
Legal Research Tool — federal/state law lookup, case law search,
statistics retrieval, and news monitoring.

Capabilities:
- Download and search federal statutes (US Code via House.gov XML)
- Download and search state statutes (Connecticut General Statutes)
- Search case law via CourtListener API (free, no key needed for basic)
- Search legal encyclopedias via Cornell LII
- Pull DOJ/FBI crime statistics and reports
- Monitor news on civil rights, police, and legal topics
- Build structured research briefs from multiple sources

All data is stored locally and ingested into the knowledge base.
"""

import os
import json
import re
import time
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Callable
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup

from tools.base import BaseTool


# ── Constants ───────────────────────────────────────────────────────────────

LEGAL_DATA_DIR = Path.home() / "LegalResearch"

FEDERAL_SOURCES = {
    "us_code": {
        "name": "United States Code",
        "base_url": "https://uscode.house.gov",
        "xml_bulk": "https://uscode.house.gov/download/download.shtml",
    },
    "cornell_lii": {
        "name": "Cornell Legal Information Institute",
        "base_url": "https://www.law.cornell.edu",
    },
    "courtlistener": {
        "name": "CourtListener (Free Law Project)",
        "api_url": "https://www.courtlistener.com/api/rest/v4",
        "search_url": "https://www.courtlistener.com",
    }
}

# US Code Title numbers and names for reference
US_CODE_TITLES = {
    1: "General Provisions", 2: "The Congress", 4: "Flag and Seal",
    5: "Government Organization", 6: "Domestic Security",
    7: "Agriculture", 8: "Aliens and Nationality", 9: "Arbitration",
    10: "Armed Forces", 11: "Bankruptcy", 12: "Banks and Banking",
    13: "Census", 14: "Coast Guard", 15: "Commerce and Trade",
    16: "Conservation", 17: "Copyrights", 18: "Crimes and Criminal Procedure",
    19: "Customs Duties", 20: "Education", 21: "Food and Drugs",
    22: "Foreign Relations", 23: "Highways", 24: "Hospitals",
    25: "Indians", 26: "Internal Revenue Code", 27: "Intoxicating Liquors",
    28: "Judiciary and Judicial Procedure", 29: "Labor",
    30: "Mineral Lands and Mining", 31: "Money and Finance",
    32: "National Guard", 33: "Navigation and Navigable Waters",
    34: "Crime Control", 35: "Patents", 36: "Patriotic Societies",
    37: "Pay and Allowances", 38: "Veterans' Benefits",
    39: "Postal Service", 40: "Public Buildings",
    41: "Public Contracts", 42: "The Public Health and Welfare",
    43: "Public Lands", 44: "Public Printing",
    45: "Railroads", 46: "Shipping", 47: "Telecommunications",
    48: "Territories", 49: "Transportation", 50: "War and National Defense",
    51: "National and Commercial Space Programs",
    52: "Voting and Elections", 54: "National Park Service",
}

# Key civil rights statutes for quick reference
CIVIL_RIGHTS_STATUTES = {
    "42 USC 1983": "Civil action for deprivation of rights (Section 1983)",
    "42 USC 1981": "Equal rights under the law",
    "42 USC 1982": "Property rights of citizens",
    "42 USC 1985": "Conspiracy to interfere with civil rights",
    "42 USC 1986": "Action for neglect to prevent conspiracy",
    "42 USC 1988": "Attorney's fees in civil rights cases",
    "42 USC 2000e": "Title VII — Employment discrimination",
    "42 USC 2000a": "Title II — Public accommodations",
    "42 USC 3601-3619": "Fair Housing Act",
    "42 USC 12101": "Americans with Disabilities Act",
    "18 USC 241": "Conspiracy against rights (criminal)",
    "18 USC 242": "Deprivation of rights under color of law (criminal)",
    "18 USC 245": "Federally protected activities",
    "18 USC 249": "Matthew Shepard Hate Crimes Act",
    "28 USC 1331": "Federal question jurisdiction",
    "28 USC 1343": "Civil rights jurisdiction",
    "28 USC 2254": "Habeas corpus — state custody",
    "28 USC 2255": "Habeas corpus — federal custody",
    "4th Amendment": "Unreasonable search and seizure",
    "5th Amendment": "Due process, self-incrimination, double jeopardy",
    "6th Amendment": "Right to counsel, speedy trial, confrontation",
    "8th Amendment": "Cruel and unusual punishment, excessive bail",
    "14th Amendment": "Equal protection, due process (state actors)",
}

# Connecticut-specific references
CT_STATUTES = {
    "ct_general_statutes": {
        "name": "Connecticut General Statutes",
        "url": "https://www.cga.ct.gov/current/pub/titles.htm",
        "search_url": "https://search.cga.state.ct.us/r/statute/",
    },
    "ct_practice_book": {
        "name": "Connecticut Practice Book (Court Rules)",
        "url": "https://www.jud.ct.gov/pb.htm",
    }
}


class LegalResearchTool(BaseTool):
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.data_dir = Path(config.get("legal", {}).get(
            "data_dir", str(LEGAL_DATA_DIR)
        ))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories
        (self.data_dir / "federal_statutes").mkdir(exist_ok=True)
        (self.data_dir / "state_statutes").mkdir(exist_ok=True)
        (self.data_dir / "case_law").mkdir(exist_ok=True)
        (self.data_dir / "statistics").mkdir(exist_ok=True)
        (self.data_dir / "news_clips").mkdir(exist_ok=True)
        (self.data_dir / "research_briefs").mkdir(exist_ok=True)
        
        self.client = httpx.Client(
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"
            },
            follow_redirects=True,
            timeout=30
        )
        
        # Shared web search tool instance (lazy-loaded)
        self._web_search = None
    
    @property
    def web_search(self):
        """Reuse a single WebSearchTool instance instead of creating new ones."""
        if self._web_search is None:
            from tools.web_search import WebSearchTool
            self._web_search = WebSearchTool(self.config)
        return self._web_search
    
    # ── Federal Statute Lookup ──────────────────────────────────────────────
    
    def lookup_statute(self, citation: str) -> str:
        """
        Look up a specific federal or state statute by citation.
        Examples: '42 USC 1983', '18 USC 242', 'CT Gen Stat 53a-217'
        """
        citation = citation.strip()
        
        # Parse federal citation (e.g., "42 USC 1983", "18 U.S.C. § 242")
        federal_match = re.match(
            r'(\d+)\s*(?:USC|U\.?S\.?C\.?)\s*§?\s*(\d+\w*)',
            citation, re.IGNORECASE
        )
        
        if federal_match:
            title = federal_match.group(1)
            section = federal_match.group(2)
            return self._fetch_federal_statute(title, section)
        
        # Parse CT citation (e.g., "CT Gen Stat 53a-217")
        ct_match = re.match(
            r'(?:CT|Conn\.?)\s*(?:Gen\.?\s*Stat\.?|CGS)\s*§?\s*([\d\w\-]+)',
            citation, re.IGNORECASE
        )
        
        if ct_match:
            section = ct_match.group(1)
            return self._fetch_ct_statute(section)
        
        # Try as a general search
        return self._search_statutes_online(citation)
    
    def _fetch_federal_statute(self, title: str, section: str) -> str:
        """Fetch a specific section of the US Code from Cornell LII."""
        url = f"https://www.law.cornell.edu/uscode/text/{title}/{section}"
        
        try:
            response = self.client.get(url)
            
            if response.status_code == 404:
                return f"Statute not found: {title} USC § {section}. Try searching instead."
            
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "lxml")
            
            # Extract the statute content
            content_div = soup.find("div", class_="tab-pane") or soup.find("div", id="content")
            
            if content_div:
                # Get title/heading
                heading = soup.find("h1") or soup.find("h2")
                heading_text = heading.get_text(strip=True) if heading else f"{title} USC § {section}"
                
                # Get statute text
                text = content_div.get_text(separator="\n", strip=True)
                
                # Clean up
                lines = [l.strip() for l in text.split("\n") if l.strip()]
                text = "\n".join(lines)
                
                if len(text) > 8000:
                    text = text[:8000] + "\n\n... [truncated — full text saved to file]"
                
                # Save locally
                save_path = self.data_dir / "federal_statutes" / f"{title}_USC_{section}.txt"
                with open(save_path, "w") as f:
                    f.write(f"# {heading_text}\n")
                    f.write(f"# Source: {url}\n")
                    f.write(f"# Retrieved: {datetime.now().isoformat()}\n\n")
                    f.write(text)
                
                result = (
                    f"**{heading_text}**\n"
                    f"Source: {url}\n"
                    f"{'─' * 60}\n\n"
                    f"{text}\n\n"
                    f"Saved to: {save_path}"
                )
                return result
            
            return f"Could not parse statute content from {url}"
            
        except Exception as e:
            return f"Error fetching statute: {e}\nURL: {url}"
    
    def _fetch_ct_statute(self, section: str) -> str:
        """Fetch a Connecticut statute by section number (e.g., '53a-22', '46a-58')."""
        # Normalize: "53a-22" → "sec_53a-22"
        sec_id = section.strip().lower()
        if not sec_id.startswith("sec_"):
            sec_id = f"sec_{sec_id}"

        # Try direct CGA URL (e.g., https://www.cga.ct.gov/current/pub/sec_53a-22.htm)
        url = f"https://www.cga.ct.gov/current/pub/{sec_id}.htm"

        try:
            response = self.client.get(url)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "lxml")

                # Extract heading
                heading = soup.find("h1") or soup.find("h2")
                heading_text = heading.get_text(strip=True) if heading else f"CGS § {section}"

                # Extract statute content
                content_div = (
                    soup.find("div", id="content") or
                    soup.find("main") or
                    soup.find("article") or
                    soup
                )
                for tag in content_div(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()

                text = content_div.get_text(separator="\n", strip=True)
                lines = [l.strip() for l in text.split("\n") if l.strip()]
                text = "\n".join(lines)

                if len(text) > 8000:
                    text = text[:8000] + "\n\n... [truncated — full text saved to file]"

                # Save locally
                safe_sec = section.replace("-", "_").replace(".", "_")
                save_path = self.data_dir / "state_statutes" / "connecticut" / f"CGS_{safe_sec}.txt"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "w") as f:
                    f.write(f"# {heading_text}\n")
                    f.write(f"# Source: {url}\n")
                    f.write(f"# Retrieved: {datetime.now().isoformat()}\n\n")
                    f.write(text)

                return (
                    f"**{heading_text}**\n"
                    f"Source: {url}\n"
                    f"{'─' * 60}\n\n"
                    f"{text}\n\n"
                    f"Saved to: {save_path}"
                )

            # If direct URL fails, try the search
            return self._search_statutes_online(f"Connecticut General Statutes section {section}")

        except Exception as e:
            # Fallback to web search
            return self._search_statutes_online(f"Connecticut General Statutes section {section}")
    
    def _search_statutes_online(self, query: str) -> str:
        """Search for statutes using web search."""
        ws = self.web_search
        
        # Add legal context to search
        legal_query = f"{query} site:law.cornell.edu OR site:cga.ct.gov OR site:law.justia.com"
        return ws.web_search(legal_query, max_results=5)
    
    # ── Case Law Search ─────────────────────────────────────────────────────
    
    def search_case_law(self, query: str, jurisdiction: str = "all",
                        max_results: int = 10) -> str:
        """
        Search court opinions via CourtListener (Free Law Project).
        Covers federal and state courts.
        """
        api_url = "https://www.courtlistener.com/api/rest/v4/search/"
        
        params = {
            "q": query,
            "type": "o",  # opinions
            "order_by": "score desc",
            "page_size": min(max_results, 20),
        }
        
        if jurisdiction != "all":
            # Map common names to CourtListener court codes
            court_map = {
                "scotus": "scotus",
                "supreme_court": "scotus",
                "2nd_circuit": "ca2",
                "ct": "ctd",  # CT District
                "ct_supreme": "conn",
                "ct_appellate": "connappct",
            }
            if jurisdiction.lower() in court_map:
                params["court"] = court_map[jurisdiction.lower()]
        
        try:
            response = self.client.get(api_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = data.get("results", [])
            
            if not results:
                return f"No cases found for: '{query}'"
            
            lines = [f"Case law results for: '{query}'\n"]
            
            for i, case in enumerate(results, 1):
                case_name = case.get("caseName", "Unknown")
                court = case.get("court", "Unknown court")
                date = case.get("dateFiled", "Unknown date")
                citation = case.get("citation", [])
                cite_str = ", ".join(citation) if citation else "No citation"
                snippet = case.get("snippet", "")
                # Clean HTML from snippet
                snippet = re.sub(r'<[^>]+>', '', snippet).strip()
                
                abs_url = case.get("absolute_url", "")
                url = f"https://www.courtlistener.com{abs_url}" if abs_url else ""
                
                lines.append(
                    f"{i}. **{case_name}**\n"
                    f"   Court: {court} | Date: {date}\n"
                    f"   Citation: {cite_str}\n"
                    f"   {snippet[:300]}\n"
                    f"   {url}\n"
                )
            
            return "\n".join(lines)
            
        except Exception as e:
            # Fallback to web search
            return self._search_statutes_online(f"case law {query}")
    
    def fetch_court_opinion(self, url: str) -> str:
        """Fetch and save the full text of a court opinion."""
        try:
            response = self.client.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "lxml")
            
            # Try to find opinion text
            opinion = (
                soup.find("div", id="opinion-content") or
                soup.find("article") or
                soup.find("div", class_="opinion") or
                soup.find("pre")  # Some opinions are in pre tags
            )
            
            if opinion:
                text = opinion.get_text(separator="\n", strip=True)
            else:
                text = soup.get_text(separator="\n", strip=True)
            
            # Clean up
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            text = "\n".join(lines)
            
            # Save locally
            safe_name = re.sub(r'[^\w\-]', '_', url.split("/")[-2] or "opinion")[:80]
            save_path = self.data_dir / "case_law" / f"{safe_name}.txt"
            
            with open(save_path, "w") as f:
                f.write(f"# Source: {url}\n")
                f.write(f"# Retrieved: {datetime.now().isoformat()}\n\n")
                f.write(text)
            
            if len(text) > 10000:
                text = text[:10000] + "\n\n... [full text saved to file]"
            
            return f"Opinion saved to: {save_path}\n\n{text}"
            
        except Exception as e:
            return f"Error fetching opinion: {e}"
    
    # ── Statistics / Data Downloads ─────────────────────────────────────────
    
    def search_legal_statistics(self, topic: str) -> str:
        """
        Search for legal/crime statistics from DOJ, FBI UCR, BJS, 
        and other government sources.
        """
        ws = self.web_search
        
        # Build targeted queries for statistical sources
        queries = [
            f"{topic} statistics site:bjs.ojp.gov OR site:ucr.fbi.gov OR site:fbi.gov",
            f"{topic} data report site:justice.gov OR site:civilrights.gov",
        ]
        
        results = []
        for q in queries:
            r = ws.web_search(q, max_results=3)
            if r and "No results" not in r:
                results.append(r)
        
        if not results:
            # Broader search
            results.append(ws.web_search(f"{topic} crime statistics government data", max_results=5))
        
        return "\n\n".join(results)
    
    def download_resource(self, url: str, filename: str = None,
                          category: str = "research_briefs") -> str:
        """
        Download a document (PDF, HTML, text) from a URL and save it
        to the legal research directory.
        """
        valid_categories = [
            "federal_statutes", "state_statutes", "case_law",
            "statistics", "news_clips", "research_briefs"
        ]
        if category not in valid_categories:
            category = "research_briefs"
        
        save_dir = self.data_dir / category
        
        try:
            response = self.client.get(url)
            response.raise_for_status()
            
            # Determine filename
            if not filename:
                # Extract from URL or content-disposition
                cd = response.headers.get("content-disposition", "")
                if "filename=" in cd:
                    filename = cd.split("filename=")[-1].strip('"')
                else:
                    filename = url.split("/")[-1].split("?")[0]
                    if not filename or "." not in filename:
                        filename = f"download_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Detect content type and save
            content_type = response.headers.get("content-type", "")
            
            if "pdf" in content_type or filename.endswith(".pdf"):
                if not filename.endswith(".pdf"):
                    filename += ".pdf"
                save_path = save_dir / filename
                with open(save_path, "wb") as f:
                    f.write(response.content)
            else:
                # Treat as text/HTML — extract and save as text
                if not any(filename.endswith(ext) for ext in [".txt", ".md", ".html"]):
                    filename += ".txt"
                save_path = save_dir / filename
                
                soup = BeautifulSoup(response.text, "lxml")
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                
                text = soup.get_text(separator="\n", strip=True)
                lines = [l.strip() for l in text.split("\n") if l.strip()]
                
                with open(save_path, "w") as f:
                    f.write(f"# Source: {url}\n")
                    f.write(f"# Downloaded: {datetime.now().isoformat()}\n\n")
                    f.write("\n".join(lines))
            
            size = save_path.stat().st_size
            return (
                f"Downloaded: {save_path}\n"
                f"Size: {size / 1024:.1f} KB\n"
                f"Category: {category}\n\n"
                f"To add to knowledge base: ingest this file at {save_path}"
            )
            
        except Exception as e:
            return f"Download error: {e}"
    
    # ── News Monitoring ─────────────────────────────────────────────────────
    
    def search_legal_news(self, topic: str, days_back: int = 30) -> str:
        """
        Search recent news on legal topics — civil rights, police,
        court decisions, legislation.
        """
        ws = self.web_search
        
        # Build news-focused queries
        queries = [
            f"{topic} news {datetime.now().year}",
        ]
        
        if "police" in topic.lower() or "civil rights" in topic.lower():
            queries.append(f"{topic} lawsuit settlement verdict {datetime.now().year}")
        
        results = []
        for q in queries:
            r = ws.web_search(q, max_results=5)
            if r and "No results" not in r:
                results.append(r)
        
        return "\n\n".join(results) if results else f"No recent news found for: '{topic}'"
    
    def clip_article(self, url: str, tags: str = "") -> str:
        """
        Save a news article to the local research library.
        Extracts clean text and stores with metadata.
        """
        try:
            response = self.client.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "lxml")
            
            # Extract metadata
            title = (soup.find("meta", {"property": "og:title"}) or {}).get("content", "")
            if not title:
                title = soup.title.string if soup.title else "Untitled"
            
            pub_date = (soup.find("meta", {"property": "article:published_time"}) or {}).get("content", "")
            author = (soup.find("meta", {"name": "author"}) or {}).get("content", "")
            
            # Extract article text
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                tag.decompose()
            
            article = (
                soup.find("article") or
                soup.find("div", class_=re.compile(r"article|story|content|post")) or
                soup.find("main")
            )
            
            if article:
                text = article.get_text(separator="\n", strip=True)
            else:
                text = soup.get_text(separator="\n", strip=True)
            
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            text = "\n".join(lines)
            
            # Save
            safe_title = re.sub(r'[^\w\s\-]', '', title)[:60].strip().replace(" ", "_")
            date_str = datetime.now().strftime("%Y%m%d")
            filename = f"{date_str}_{safe_title}.txt"
            save_path = self.data_dir / "news_clips" / filename
            
            with open(save_path, "w") as f:
                f.write(f"# {title}\n")
                f.write(f"# Source: {url}\n")
                f.write(f"# Author: {author}\n")
                f.write(f"# Published: {pub_date}\n")
                f.write(f"# Clipped: {datetime.now().isoformat()}\n")
                if tags:
                    f.write(f"# Tags: {tags}\n")
                f.write(f"\n{'─' * 60}\n\n")
                f.write(text)
            
            return (
                f"Article clipped: {title}\n"
                f"Saved to: {save_path}\n"
                f"Length: {len(text)} chars\n\n"
                f"To add to knowledge base, ingest this file."
            )
            
        except Exception as e:
            return f"Error clipping article: {e}"
    
    # ── Research Brief Generation ───────────────────────────────────────────
    
    def generate_research_brief(self, topic: str, output_file: str = None) -> str:
        """
        Create a structured research brief template on a legal topic.
        Populates with relevant statutes, case references, and questions.
        """
        if not output_file:
            safe_topic = re.sub(r'[^\w\s\-]', '', topic)[:40].strip().replace(" ", "_")
            output_file = str(
                self.data_dir / "research_briefs" / 
                f"brief_{safe_topic}_{datetime.now().strftime('%Y%m%d')}.md"
            )
        
        brief = f"""# Research Brief: {topic}
**Date:** {datetime.now().strftime('%B %d, %Y')}
**Status:** Draft

---

## I. Issue Statement

[State the legal question(s) to be answered]

## II. Relevant Statutes

### Federal
| Citation | Description | Relevance |
|----------|-------------|-----------|
| 42 USC § 1983 | Civil action for deprivation of rights | |
| | | |
| | | |

### State (Connecticut)
| Citation | Description | Relevance |
|----------|-------------|-----------|
| | | |

## III. Key Case Law

| Case | Court | Year | Holding | Relevance |
|------|-------|------|---------|-----------|
| | | | | |

## IV. Facts / Factual Background

[Relevant facts of the situation]

## V. Legal Analysis

### A. Elements to Prove
1. 
2. 
3. 

### B. Arguments For
- 

### C. Arguments Against
- 

### D. Standards of Review
- 

## VI. Statistical Evidence

[Relevant data, studies, reports]

| Source | Finding | Date |
|--------|---------|------|
| | | |

## VII. Related News / Current Events

[Recent articles, cases, or developments]

## VIII. Conclusions / Recommendations

[Summary of research findings and recommended course of action]

## IX. Sources Consulted

1. 
2. 
3. 

---
*Generated by Local AI Assistant — {datetime.now().isoformat()}*
"""
        
        with open(output_file, "w") as f:
            f.write(brief)
        
        return (
            f"Research brief template created: {output_file}\n\n"
            f"To populate it with actual research, ask me to:\n"
            f"1. Look up specific statutes related to '{topic}'\n"
            f"2. Search case law for '{topic}'\n"
            f"3. Find statistics on '{topic}'\n"
            f"4. Search recent news about '{topic}'\n"
            f"I'll fill in the brief with real data."
        )
    
    # ── Bulk Download Helpers ───────────────────────────────────────────────
    
    def download_statute_collection(self, collection: str = "civil_rights") -> str:
        """
        Download a curated collection of statutes on a topic.
        Available: civil_rights, criminal_procedure, evidence,
                   police_accountability, ct_criminal
        """
        collections = {
            "civil_rights": {
                "title": "Civil Rights Statutes",
                "citations": [
                    ("42", "1981"), ("42", "1982"), ("42", "1983"),
                    ("42", "1985"), ("42", "1986"), ("42", "1988"),
                    ("18", "241"), ("18", "242"), ("18", "245"), ("18", "249"),
                ]
            },
            "criminal_procedure": {
                "title": "Criminal Procedure",
                "citations": [
                    ("18", "3161"),  # Speedy Trial Act
                    ("18", "3142"),  # Bail Reform Act
                    ("18", "3553"),  # Sentencing factors
                    ("18", "3582"),  # Sentence modification
                    ("28", "2241"),  # Habeas corpus
                    ("28", "2254"),  # Habeas — state
                    ("28", "2255"),  # Habeas — federal
                ]
            },
            "evidence": {
                "title": "Federal Rules of Evidence (key sections)",
                "citations": [
                    ("28", "1731"),  # Handwriting evidence
                    ("28", "1732"),  # Record of regularly conducted activity
                    ("28", "1733"),  # Government records
                    ("28", "1734"),  # Court record lost
                    ("28", "1735"),  # Court record from another district
                ]
            },
            "police_accountability": {
                "title": "Police Accountability & Excessive Force",
                "citations": [
                    ("42", "1983"),  # Section 1983
                    ("42", "14141"), # Pattern or practice
                    ("18", "242"),   # Deprivation under color of law
                    ("34", "12601"), # Investigations of law enforcement
                    ("42", "3789d"), # Civil rights compliance
                ]
            }
        }
        
        if collection not in collections:
            avail = ", ".join(collections.keys())
            return f"Unknown collection: '{collection}'\nAvailable: {avail}"
        
        coll = collections[collection]
        results = [f"Downloading: {coll['title']}\n"]
        
        for title, section in coll["citations"]:
            result = self._fetch_federal_statute(title, section)
            status = "✅" if "Error" not in result and "not found" not in result.lower() else "❌"
            results.append(f"  {status} {title} USC § {section}")
            time.sleep(1)  # Be polite to Cornell LII
        
        results.append(f"\nFiles saved to: {self.data_dir / 'federal_statutes'}")
        results.append(
            f"\nTo ingest into knowledge base:\n"
            f"  > ingest directory {self.data_dir / 'federal_statutes'}"
        )
        
        return "\n".join(results)
    
    def list_civil_rights_statutes(self) -> str:
        """Quick reference list of key civil rights statutes."""
        lines = ["Key Civil Rights Statutes:\n"]
        for citation, description in CIVIL_RIGHTS_STATUTES.items():
            lines.append(f"  {citation:.<30} {description}")
        return "\n".join(lines)
    
    def list_research_files(self, category: str = "all") -> str:
        """List all files in the legal research directory."""
        if category == "all":
            dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        else:
            target = self.data_dir / category
            if not target.exists():
                return f"Category not found: {category}"
            dirs = [target]
        
        lines = [f"Legal Research Library: {self.data_dir}\n"]
        
        total = 0
        for d in sorted(dirs):
            files = sorted(d.glob("*"))
            files = [f for f in files if f.is_file()]
            if files:
                lines.append(f"\n📂 {d.name}/ ({len(files)} files)")
                for f in files[:20]:
                    size = f.stat().st_size
                    lines.append(f"  📄 {f.name} ({size/1024:.0f}KB)")
                if len(files) > 20:
                    lines.append(f"  ... and {len(files) - 20} more")
                total += len(files)
        
        lines.append(f"\nTotal: {total} files")
        return "\n".join(lines)
    
    # ── Case Comparison Engine ──────────────────────────────────────────────
    
    def compare_cases(self, case1: str, case2: str) -> str:
        """
        Fetch two cases and generate a structured comparison.
        Input can be case names, citations, or CourtListener URLs.
        """
        # Fetch both cases
        results1 = self.search_case_law(case1, max_results=1)
        results2 = self.search_case_law(case2, max_results=1)
        
        comparison = f"""# Case Comparison

## Case 1: {case1}
{results1}

## Case 2: {case2}
{results2}

## Comparison Framework

| Element | Case 1 | Case 2 |
|---------|--------|--------|
| Facts | [extracted from above] | [extracted from above] |
| Legal Issue | | |
| Holding | | |
| Standard Applied | | |
| Key Reasoning | | |
| Disposition | | |

## Key Differences

[The LLM should fill this in based on the retrieved case data]

## How These Cases Relate

[Analysis of how these cases interact — does one extend, limit, or distinguish the other?]

---
*Note: This comparison is based on available summaries. Read the full opinions for complete analysis.*
"""
        
        # Save comparison
        safe1 = re.sub(r'[^\w]', '_', case1)[:20]
        safe2 = re.sub(r'[^\w]', '_', case2)[:20]
        save_path = self.data_dir / "research_briefs" / f"comparison_{safe1}_vs_{safe2}.md"
        
        with open(save_path, "w") as f:
            f.write(comparison)
        
        return comparison + f"\nSaved to: {save_path}"
    
    # ── Project Management ──────────────────────────────────────────────────
    
    def create_project(self, name: str, description: str = "",
                       facts: str = "", relevant_laws: str = "") -> str:
        """
        Create a legal research project. Projects organize ongoing case work
        with facts, relevant laws, notes, and research files.
        """
        projects_dir = self.data_dir / "projects"
        projects_dir.mkdir(exist_ok=True)
        
        safe_name = re.sub(r'[^\w\-]', '_', name).lower()
        project_dir = projects_dir / safe_name
        project_dir.mkdir(exist_ok=True)
        
        project = {
            "name": name,
            "created": datetime.now().isoformat(),
            "updated": datetime.now().isoformat(),
            "description": description,
            "facts": facts,
            "relevant_laws": [l.strip() for l in relevant_laws.split(",") if l.strip()] if relevant_laws else [],
            "notes": [],
            "status": "active",
        }
        
        project_file = project_dir / "project.json"
        with open(project_file, "w") as f:
            json.dump(project, f, indent=2)
        
        # Create notes file
        notes_file = project_dir / "notes.md"
        with open(notes_file, "w") as f:
            f.write(f"# {name} — Research Notes\n\n")
            f.write(f"Created: {datetime.now().strftime('%B %d, %Y')}\n\n")
            if description:
                f.write(f"## Description\n{description}\n\n")
            if facts:
                f.write(f"## Facts\n{facts}\n\n")
            f.write("## Notes\n\n")
        
        return (
            f"Project created: {name}\n"
            f"Directory: {project_dir}\n"
            f"Files: project.json, notes.md\n\n"
            f"Add research to this project by saving files to {project_dir}\n"
            f"Then ingest the project directory into the knowledge base."
        )
    
    def update_project(self, name: str, note: str = None,
                       add_law: str = None, facts: str = None,
                       status: str = None) -> str:
        """Update a legal research project with new notes, laws, or facts."""
        projects_dir = self.data_dir / "projects"
        safe_name = re.sub(r'[^\w\-]', '_', name).lower()
        project_dir = projects_dir / safe_name
        project_file = project_dir / "project.json"
        
        if not project_file.exists():
            return f"Project not found: {name}. Create it first with create_project."
        
        with open(project_file) as f:
            project = json.load(f)
        
        if note:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            project["notes"].append({"date": timestamp, "text": note})
            # Also append to notes.md
            notes_file = project_dir / "notes.md"
            with open(notes_file, "a") as f:
                f.write(f"\n### {timestamp}\n{note}\n")
        
        if add_law:
            if add_law not in project["relevant_laws"]:
                project["relevant_laws"].append(add_law)
        
        if facts:
            project["facts"] = facts
        
        if status:
            project["status"] = status
        
        project["updated"] = datetime.now().isoformat()
        
        with open(project_file, "w") as f:
            json.dump(project, f, indent=2)
        
        return f"Project '{name}' updated."
    
    def list_projects(self) -> str:
        """List all legal research projects."""
        projects_dir = self.data_dir / "projects"
        if not projects_dir.exists():
            return "No projects yet. Create one with create_project."
        
        dirs = [d for d in projects_dir.iterdir() if d.is_dir()]
        if not dirs:
            return "No projects yet."
        
        lines = ["Legal Research Projects:\n"]
        for d in sorted(dirs):
            pf = d / "project.json"
            if pf.exists():
                with open(pf) as f:
                    p = json.load(f)
                status = p.get("status", "active")
                laws = ", ".join(p.get("relevant_laws", [])[:3])
                notes_count = len(p.get("notes", []))
                lines.append(
                    f"  📋 {p['name']} [{status}]\n"
                    f"     Laws: {laws or 'none yet'}\n"
                    f"     Notes: {notes_count} | Updated: {p.get('updated', '?')[:10]}"
                )
        
        return "\n".join(lines)
    
    def get_project(self, name: str) -> str:
        """Get full details of a legal research project."""
        projects_dir = self.data_dir / "projects"
        safe_name = re.sub(r'[^\w\-]', '_', name).lower()
        project_file = projects_dir / safe_name / "project.json"
        
        if not project_file.exists():
            return f"Project not found: {name}"
        
        with open(project_file) as f:
            p = json.load(f)
        
        lines = [
            f"# Project: {p['name']}",
            f"Status: {p.get('status', 'active')}",
            f"Created: {p.get('created', '?')[:10]}",
            f"Updated: {p.get('updated', '?')[:10]}",
        ]
        
        if p.get("description"):
            lines.append(f"\n## Description\n{p['description']}")
        
        if p.get("facts"):
            lines.append(f"\n## Facts\n{p['facts']}")
        
        if p.get("relevant_laws"):
            lines.append(f"\n## Relevant Laws\n" + "\n".join(f"  - {l}" for l in p["relevant_laws"]))
        
        if p.get("notes"):
            lines.append(f"\n## Notes ({len(p['notes'])} entries)")
            for n in p["notes"][-5:]:  # Show last 5
                lines.append(f"\n### {n['date']}\n{n['text']}")
        
        return "\n".join(lines)
    
    # ── Tool Definitions ────────────────────────────────────────────────────
    
    def get_tool_definitions(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "lookup_statute",
                    "description": "Look up a specific federal or Connecticut state statute by citation. Examples: '42 USC 1983', '18 USC 242', 'CT Gen Stat 53a-217'. Downloads the full text and saves locally.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "citation": {"type": "string", "description": "Legal citation (e.g., '42 USC 1983', '18 USC 242')"}
                        },
                        "required": ["citation"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_case_law",
                    "description": "Search court opinions and case law. Covers federal and state courts. Returns case names, citations, dates, and snippets.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query (legal issue, case name, topic)"},
                            "jurisdiction": {"type": "string", "description": "Court filter: 'all', 'scotus', '2nd_circuit', 'ct', 'ct_supreme', 'ct_appellate'"},
                            "max_results": {"type": "integer", "description": "Number of results (default: 10)"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_court_opinion",
                    "description": "Download the full text of a court opinion from a URL. Saves it locally for reference.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "URL of the court opinion"}
                        },
                        "required": ["url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_legal_statistics",
                    "description": "Search for crime statistics, DOJ reports, FBI UCR data, and government studies on a legal topic.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string", "description": "Topic to find statistics on (e.g., 'police use of force', 'racial profiling', 'wrongful conviction rates')"}
                        },
                        "required": ["topic"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_legal_news",
                    "description": "Search recent news articles on legal topics — civil rights, police, court decisions, new legislation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string", "description": "News topic to search"},
                            "days_back": {"type": "integer", "description": "How many days back to search (default: 30)"}
                        },
                        "required": ["topic"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "clip_article",
                    "description": "Save a news or research article from a URL to the local legal research library. Extracts clean text with metadata.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "Article URL to clip"},
                            "tags": {"type": "string", "description": "Comma-separated tags (e.g., 'civil-rights, police, ct')"}
                        },
                        "required": ["url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "download_resource",
                    "description": "Download a document (PDF, webpage, report) from a URL and save to the legal research library.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "URL to download"},
                            "filename": {"type": "string", "description": "Custom filename (optional)"},
                            "category": {"type": "string", "enum": ["federal_statutes", "state_statutes", "case_law", "statistics", "news_clips", "research_briefs"]}
                        },
                        "required": ["url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "download_statute_collection",
                    "description": "Download a curated set of statutes on a topic. Available: 'civil_rights', 'criminal_procedure', 'evidence', 'police_accountability'",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "collection": {"type": "string", "description": "Collection name"}
                        },
                        "required": ["collection"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_research_brief",
                    "description": "Create a structured research brief template on a legal topic with sections for statutes, case law, facts, analysis, and sources.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string", "description": "Legal topic or issue"},
                            "output_file": {"type": "string", "description": "Custom output path (optional)"}
                        },
                        "required": ["topic"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_civil_rights_statutes",
                    "description": "Quick reference list of key civil rights statutes (Section 1983, Title VII, constitutional amendments, etc.)",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_research_files",
                    "description": "List all files in the legal research library, optionally filtered by category.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "category": {"type": "string", "description": "'all', 'federal_statutes', 'state_statutes', 'case_law', 'statistics', 'news_clips', 'research_briefs'"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "compare_cases",
                    "description": "Compare two court cases side by side. Searches for both cases and generates a structured comparison of facts, legal issues, holdings, reasoning, and key differences.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "case1": {"type": "string", "description": "First case (name, citation, or search query)"},
                            "case2": {"type": "string", "description": "Second case (name, citation, or search query)"}
                        },
                        "required": ["case1", "case2"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_project",
                    "description": "Create a legal research project. Organizes ongoing case work with facts, relevant laws, notes, and research files in ~/LegalResearch/projects/.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Project name"},
                            "description": {"type": "string", "description": "Brief description of the case or research topic"},
                            "facts": {"type": "string", "description": "Key facts of the situation"},
                            "relevant_laws": {"type": "string", "description": "Comma-separated list of relevant statutes (e.g., '42 USC 1983, 18 USC 242')"}
                        },
                        "required": ["name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "update_project",
                    "description": "Update a legal research project with new notes, relevant laws, facts, or status changes.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Project name"},
                            "note": {"type": "string", "description": "New note to add"},
                            "add_law": {"type": "string", "description": "Add a relevant statute citation"},
                            "facts": {"type": "string", "description": "Updated facts"},
                            "status": {"type": "string", "description": "New status: 'active', 'on_hold', 'closed', 'research_complete'"}
                        },
                        "required": ["name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_projects",
                    "description": "List all legal research projects with their status, relevant laws, and note counts.",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_project",
                    "description": "Get full details of a legal research project including facts, laws, and recent notes.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Project name"}
                        },
                        "required": ["name"]
                    }
                }
            }
        ]
    
    def get_handlers(self) -> dict[str, Callable]:
        return {
            "lookup_statute": self.lookup_statute,
            "search_case_law": self.search_case_law,
            "fetch_court_opinion": self.fetch_court_opinion,
            "search_legal_statistics": self.search_legal_statistics,
            "search_legal_news": self.search_legal_news,
            "clip_article": self.clip_article,
            "download_resource": self.download_resource,
            "download_statute_collection": self.download_statute_collection,
            "generate_research_brief": self.generate_research_brief,
            "list_civil_rights_statutes": self.list_civil_rights_statutes,
            "list_research_files": self.list_research_files,
            "compare_cases": self.compare_cases,
            "create_project": self.create_project,
            "update_project": self.update_project,
            "list_projects": self.list_projects,
            "get_project": self.get_project,
        }
