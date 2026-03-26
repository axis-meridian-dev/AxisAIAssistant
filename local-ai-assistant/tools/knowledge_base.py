"""
Knowledge Base Tool — Local RAG (Retrieval Augmented Generation).

Ingests your files into a local vector database (ChromaDB) so the LLM can
search and reference your documents, code, notes, and research when answering.

Supports: .txt, .md, .py, .js, .ts, .json, .yaml, .yml, .toml, .cfg, .ini,
          .html, .css, .sh, .bash, .rs, .go, .java, .c, .cpp, .h, .hpp,
          .pdf, .csv, .log, .xml, .sql, .env, .conf, .rst, .tex

Flow:
  1. Ingest: scan directories → chunk files → embed with local model → store in ChromaDB
  2. Query: embed question → find similar chunks → inject into LLM context
  3. The LLM sees your actual file content alongside the question

Everything runs locally. No data leaves your machine.
"""

import os
import re
import hashlib
import json
import time
from pathlib import Path
from typing import Callable
from datetime import datetime

from tools.base import BaseTool

# ── File type handlers ──────────────────────────────────────────────────────

TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".jsx", ".tsx", ".json", ".yaml",
    ".yml", ".toml", ".cfg", ".ini", ".html", ".css", ".sh", ".bash",
    ".rs", ".go", ".java", ".c", ".cpp", ".h", ".hpp", ".sql", ".env",
    ".conf", ".rst", ".tex", ".log", ".xml", ".csv", ".r", ".rb",
    ".php", ".swift", ".kt", ".scala", ".lua", ".vim", ".dockerfile",
    ".makefile", ".cmake", ".gradle", ".properties", ".gitignore",
    ".editorconfig", ".prettierrc", ".eslintrc",
}

PDF_SUPPORT = False
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    pass


def read_text_file(path: Path, max_chars: int = 100_000) -> str | None:
    """Read a text file, return None if binary/unreadable."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(max_chars)
    except Exception:
        return None


def read_pdf_file(path: Path) -> str | None:
    """Extract text from PDF using PyMuPDF."""
    if not PDF_SUPPORT:
        return None
    try:
        doc = fitz.open(str(path))
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return text.strip() if text.strip() else None
    except Exception:
        return None


def read_file_content(path: Path) -> str | None:
    """Read file content based on extension."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return read_pdf_file(path)
    elif ext in TEXT_EXTENSIONS or ext == "":
        return read_text_file(path)
    return None


# ── Chunking ────────────────────────────────────────────────────────────────

def detect_document_type(text: str, filepath: Path = None) -> str:
    """Classify document type for smarter chunking and metadata."""
    text_lower = text[:2000].lower()
    name = filepath.name.lower() if filepath else ""
    
    # Statute detection
    if any(p in text_lower for p in [
        "united states code", "u.s.c.", "usc §", "§ ", "section ",
        "public law", "stat.", "enacted by", "an act to"
    ]) or "USC" in name:
        return "statute"
    
    # Case law detection
    if any(p in text_lower for p in [
        "plaintiff", "defendant", "appellant", "appellee", "petitioner",
        "respondent", "opinion of the court", "syllabus", "certiorari",
        "reversed and remanded", "affirmed", "dissenting", "concurring",
        " v. ", " vs. "
    ]):
        return "case_law"
    
    # Legal brief/memo
    if any(p in text_lower for p in [
        "memorandum of law", "brief in support", "motion to",
        "comes now", "argument", "statement of facts", "prayer for relief"
    ]):
        return "legal_brief"
    
    # News article
    if any(p in text_lower for p in [
        "associated press", "reuters", "(ap)", "reporting by",
        "published:", "updated:", "breaking news"
    ]) or filepath and "news_clips" in str(filepath):
        return "news_article"
    
    # Statistics/data
    if any(p in text_lower for p in [
        "bureau of justice", "fbi.gov", "census", "survey",
        "table 1", "figure 1", "percentage", "per capita"
    ]) or filepath and "statistics" in str(filepath):
        return "statistics"
    
    # Code
    if filepath and filepath.suffix in {".py", ".js", ".ts", ".rs", ".go", ".java", ".c", ".cpp"}:
        return "code"
    
    return "document"


def detect_legal_topics(text: str) -> list[str]:
    """Auto-detect legal topic tags from content."""
    text_lower = text.lower()
    tags = []
    
    topic_patterns = {
        "civil_rights": ["civil rights", "1983", "equal protection", "discrimination", "14th amendment"],
        "excessive_force": ["excessive force", "use of force", "deadly force", "police brutality", "taser", "chokehold"],
        "qualified_immunity": ["qualified immunity", "clearly established", "reasonable officer"],
        "search_seizure": ["search and seizure", "4th amendment", "fourth amendment", "warrant", "probable cause", "terry stop"],
        "due_process": ["due process", "5th amendment", "fifth amendment", "14th amendment"],
        "traffic_stop": ["traffic stop", "motor vehicle", "terry stop", "reasonable suspicion"],
        "false_arrest": ["false arrest", "false imprisonment", "unlawful detention", "wrongful arrest"],
        "employment": ["title vii", "employment discrimination", "hostile work environment", "retaliation"],
        "housing": ["fair housing", "housing discrimination", "section 8", "redlining"],
        "criminal_procedure": ["miranda", "right to counsel", "speedy trial", "plea", "sentencing"],
        "habeas_corpus": ["habeas corpus", "2254", "2255", "post-conviction"],
        "first_amendment": ["free speech", "1st amendment", "first amendment", "freedom of expression", "press"],
        "eighth_amendment": ["cruel and unusual", "8th amendment", "eighth amendment", "excessive bail"],
    }
    
    for tag, patterns in topic_patterns.items():
        if any(p in text_lower for p in patterns):
            tags.append(tag)
    
    return tags if tags else ["general"]


def detect_jurisdiction(text: str) -> str:
    """Detect jurisdiction from content."""
    text_lower = text.lower()
    
    if any(p in text_lower for p in ["supreme court of the united states", "scotus", "u.s. supreme"]):
        return "scotus"
    if any(p in text_lower for p in ["united states code", "u.s.c.", "federal"]):
        return "federal"
    if any(p in text_lower for p in ["connecticut", "conn.", "ct gen stat", "ct."]):
        return "connecticut"
    if "circuit" in text_lower:
        return "federal_appellate"
    if "district court" in text_lower:
        return "federal_district"
    return "unknown"


def extract_citations(text: str) -> list[str]:
    """Extract legal citations from text."""
    citations = []
    
    # Federal statute: 42 U.S.C. § 1983
    citations.extend(re.findall(
        r'\d+\s*U\.?S\.?C\.?\s*§?\s*\d+\w*', text
    ))
    
    # Case citations: 490 U.S. 386
    citations.extend(re.findall(
        r'\d+\s+(?:U\.S\.|S\.Ct\.|L\.Ed|F\.\d+[a-z]*|F\.Supp)\s*\d+', text
    ))
    
    # Named cases: Graham v. Connor
    citations.extend(re.findall(
        r'[A-Z][a-z]+\s+v\.\s+[A-Z][a-z]+', text
    ))
    
    return list(set(citations))[:20]  # Dedupe, cap at 20


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200,
               doc_type: str = "document") -> list[str]:
    """
    Split text into overlapping chunks.
    Legal-aware: splits on section (§) boundaries for statutes,
    headnote boundaries for case law, paragraph boundaries for everything else.
    """
    if len(text) <= chunk_size:
        return [text]
    
    # Legal-aware splitting
    if doc_type == "statute":
        # Split on section markers
        section_pattern = r'(?=§\s*\d|Section\s+\d|(?:^\([a-z]\))|(?:^\(\d+\)))'
        sections = re.split(section_pattern, text, flags=re.MULTILINE)
        sections = [s.strip() for s in sections if s.strip()]
        if len(sections) > 1:
            return _merge_chunks(sections, chunk_size, overlap)
    
    elif doc_type == "case_law":
        # Split on opinion structure markers
        markers = r'(?=(?:^|\n)(?:I{1,3}V?\.?\s|OPINION|DISSENT|CONCUR|BACKGROUND|FACTS|ANALYSIS|CONCLUSION|DISCUSSION))'
        sections = re.split(markers, text, flags=re.MULTILINE | re.IGNORECASE)
        sections = [s.strip() for s in sections if s.strip()]
        if len(sections) > 1:
            return _merge_chunks(sections, chunk_size, overlap)
    
    # Default: paragraph-based splitting
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            if len(para) > chunk_size:
                lines = para.split("\n")
                current_chunk = ""
                for line in lines:
                    if len(current_chunk) + len(line) + 1 <= chunk_size:
                        current_chunk += line + "\n"
                    else:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        current_chunk = line + "\n"
            else:
                current_chunk = para + "\n\n"
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Add overlap between chunks
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_end = chunks[i - 1][-overlap:]
            overlapped.append(prev_end + "\n" + chunks[i])
        chunks = overlapped
    
    return chunks


def _merge_chunks(sections: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Merge small sections into chunks of target size."""
    chunks = []
    current = ""
    
    for section in sections:
        if len(current) + len(section) + 2 <= chunk_size:
            current += section + "\n\n"
        else:
            if current.strip():
                chunks.append(current.strip())
            if len(section) > chunk_size:
                # Section too big — split by paragraphs
                sub_chunks = chunk_text(section, chunk_size, overlap, "document")
                chunks.extend(sub_chunks)
                current = ""
            else:
                current = section + "\n\n"
    
    if current.strip():
        chunks.append(current.strip())
    
    return chunks


def file_hash(path: Path) -> str:
    """Quick hash of file for change detection."""
    stat = path.stat()
    key = f"{path}:{stat.st_size}:{stat.st_mtime}"
    return hashlib.md5(key.encode()).hexdigest()


# ── Knowledge Base Tool ─────────────────────────────────────────────────────

class KnowledgeBaseTool(BaseTool):
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.kb_config = config.get("knowledge_base", {})
        self.db_path = Path(self.kb_config.get(
            "db_path",
            str(Path.home() / ".local" / "share" / "ai-assistant" / "knowledge_db")
        ))
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.chunk_size = self.kb_config.get("chunk_size", 1000)
        self.chunk_overlap = self.kb_config.get("chunk_overlap", 200)
        self.embed_model = self.kb_config.get("embed_model", "nomic-embed-text")
        self.max_results = self.kb_config.get("max_results", 5)
        
        # Track ingested files
        self.manifest_path = self.db_path / "manifest.json"
        self.manifest = self._load_manifest()
        
        # ChromaDB — lazy init
        self._chroma_client = None
        self._collection = None
    
    @property
    def collection(self):
        """Lazy-load ChromaDB collection."""
        if self._collection is None:
            import chromadb
            from chromadb.config import Settings
            
            self._chroma_client = chromadb.PersistentClient(
                path=str(self.db_path / "chroma"),
                settings=Settings(anonymized_telemetry=False)
            )
            self._collection = self._chroma_client.get_or_create_collection(
                name="knowledge_base",
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection
    
    def _load_manifest(self) -> dict:
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)
        return {"files": {}, "stats": {"total_chunks": 0, "total_files": 0}}
    
    def _save_manifest(self):
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)
    
    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using Ollama's embedding model."""
        import ollama
        
        embeddings = []
        # Batch to avoid OOM
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = ollama.embed(model=self.embed_model, input=batch)
            embeddings.extend(response["embeddings"])
        return embeddings
    
    # ── Public methods (exposed as tools) ───────────────────────────────────
    
    def ingest_directory(self, path: str = "~", recursive: bool = True,
                         file_types: str = "all") -> str:
        """
        Scan a directory and ingest all supported files into the knowledge base.
        This makes them searchable by the AI.
        """
        target = Path(path).expanduser().resolve()
        
        if not target.exists():
            return f"Directory not found: {target}"
        if not target.is_dir():
            return f"Not a directory: {target}"
        
        # Determine which extensions to include
        if file_types == "all":
            valid_ext = TEXT_EXTENSIONS | {".pdf"}
        elif file_types == "code":
            valid_ext = {".py", ".js", ".ts", ".jsx", ".tsx", ".rs", ".go",
                        ".java", ".c", ".cpp", ".h", ".hpp", ".rb", ".php",
                        ".sh", ".bash", ".sql", ".lua"}
        elif file_types == "docs":
            valid_ext = {".txt", ".md", ".rst", ".tex", ".pdf", ".html", ".xml"}
        else:
            valid_ext = set(f".{e.strip('.')}" for e in file_types.split(","))
        
        excluded_dirs = {".git", "node_modules", "__pycache__", ".cache",
                        ".venv", "venv", ".tox", "dist", "build", ".eggs",
                        ".mypy_cache", ".pytest_cache", "target"}
        
        # Collect files
        if recursive:
            files = [f for f in target.rglob("*") if f.is_file()]
        else:
            files = [f for f in target.iterdir() if f.is_file()]
        
        # Filter
        files = [
            f for f in files
            if f.suffix.lower() in valid_ext
            and not any(ex in f.parts for ex in excluded_dirs)
            and f.stat().st_size < 50 * 1024 * 1024  # 50MB max
        ]
        
        if not files:
            return f"No supported files found in {target}"
        
        print(f"\n  Found {len(files)} files to process...\n")
        
        ingested = 0
        skipped = 0
        errors = 0
        total_chunks = 0
        
        for file_idx, filepath in enumerate(files):
            try:
                fhash = file_hash(filepath)
                str_path = str(filepath)
                
                # Progress output every file
                print(f"  [{file_idx+1}/{len(files)}] {filepath.name}...", end=" ", flush=True)
                
                # Skip if already ingested and unchanged
                if str_path in self.manifest["files"]:
                    if self.manifest["files"][str_path]["hash"] == fhash:
                        skipped += 1
                        print("(unchanged, skipped)")
                        continue
                    else:
                        # File changed — remove old chunks and re-ingest
                        self._remove_file_chunks(str_path)
                
                # Read file
                content = read_file_content(filepath)
                if not content or len(content.strip()) < 10:
                    skipped += 1
                    print("(empty/unreadable, skipped)")
                    continue
                
                # Detect document type for smart chunking
                doc_type = detect_document_type(content, filepath)
                topics = detect_legal_topics(content)
                jurisdiction = detect_jurisdiction(content) if doc_type in ("statute", "case_law", "legal_brief") else "n/a"
                citations_found = extract_citations(content) if doc_type in ("statute", "case_law", "legal_brief") else []
                
                # Chunk with type awareness
                chunks = chunk_text(content, self.chunk_size, self.chunk_overlap, doc_type)
                
                # Create metadata for each chunk
                ids = []
                documents = []
                metadatas = []
                
                for i, chunk in enumerate(chunks):
                    chunk_id = hashlib.md5(
                        f"{str_path}:chunk:{i}:{fhash}".encode()
                    ).hexdigest()
                    
                    ids.append(chunk_id)
                    documents.append(chunk)
                    metadatas.append({
                        "source": str_path,
                        "filename": filepath.name,
                        "extension": filepath.suffix,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "file_hash": fhash,
                        "ingested_at": datetime.now().isoformat(),
                        "file_size": filepath.stat().st_size,
                        "directory": str(filepath.parent),
                        "doc_type": doc_type,
                        "topics": ",".join(topics),
                        "jurisdiction": jurisdiction,
                        "citations": ",".join(citations_found[:5]),
                    })
                
                # Embed and store
                embeddings = self._embed(documents)
                
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )
                
                # Update manifest
                self.manifest["files"][str_path] = {
                    "hash": fhash,
                    "chunks": len(chunks),
                    "ingested_at": datetime.now().isoformat(),
                    "size": filepath.stat().st_size,
                }
                
                ingested += 1
                total_chunks += len(chunks)
                print(f"OK ({len(chunks)} chunks)")
                
                # Save manifest every 25 files so progress isn't lost on crash
                if ingested % 25 == 0:
                    self.manifest["stats"]["total_files"] = len(self.manifest["files"])
                    self.manifest["stats"]["total_chunks"] = self.collection.count()
                    self._save_manifest()
                    print(f"  --- checkpoint: {ingested} files ingested so far ---")
                
            except Exception as e:
                errors += 1
                print(f"ERROR: {e}")
        
        # Update stats
        self.manifest["stats"]["total_files"] = len(self.manifest["files"])
        self.manifest["stats"]["total_chunks"] = self.collection.count()
        self._save_manifest()
        
        return (
            f"Ingestion complete:\n"
            f"  📥 Ingested: {ingested} files ({total_chunks} chunks)\n"
            f"  ⏭️  Skipped (unchanged): {skipped}\n"
            f"  ❌ Errors: {errors}\n"
            f"  📊 Total in knowledge base: {self.manifest['stats']['total_files']} files, "
            f"{self.manifest['stats']['total_chunks']} chunks\n"
            f"  📁 Database: {self.db_path}"
        )
    
    def ingest_file(self, path: str) -> str:
        """Ingest a single file into the knowledge base with full metadata extraction."""
        filepath = Path(path).expanduser().resolve()

        if not filepath.exists():
            return f"File not found: {filepath}"
        if not filepath.is_file():
            return f"Not a file: {filepath}"

        content = read_file_content(filepath)
        if not content:
            return f"Could not read file (unsupported format or binary): {filepath}"

        fhash = file_hash(filepath)
        str_path = str(filepath)

        # Remove old version if exists
        if str_path in self.manifest["files"]:
            self._remove_file_chunks(str_path)

        # Detect document type and extract metadata (same as ingest_directory)
        doc_type = detect_document_type(content, filepath)
        topics = detect_legal_topics(content)
        jurisdiction = detect_jurisdiction(content) if doc_type in ("statute", "case_law", "legal_brief") else "n/a"
        citations_found = extract_citations(content) if doc_type in ("statute", "case_law", "legal_brief") else []

        # Chunk with type awareness
        chunks = chunk_text(content, self.chunk_size, self.chunk_overlap, doc_type)

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(
                f"{str_path}:chunk:{i}:{fhash}".encode()
            ).hexdigest()

            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "source": str_path,
                "filename": filepath.name,
                "extension": filepath.suffix,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "file_hash": fhash,
                "ingested_at": datetime.now().isoformat(),
                "file_size": filepath.stat().st_size,
                "directory": str(filepath.parent),
                "doc_type": doc_type,
                "topics": ",".join(topics),
                "jurisdiction": jurisdiction,
                "citations": ",".join(citations_found[:5]),
            })

        embeddings = self._embed(documents)

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        self.manifest["files"][str_path] = {
            "hash": fhash,
            "chunks": len(chunks),
            "ingested_at": datetime.now().isoformat(),
            "size": filepath.stat().st_size,
        }
        self.manifest["stats"]["total_files"] = len(self.manifest["files"])
        self.manifest["stats"]["total_chunks"] = self.collection.count()
        self._save_manifest()

        meta_info = f"  Type: {doc_type}"
        if topics != ["general"]:
            meta_info += f"\n  Topics: {', '.join(topics)}"
        if jurisdiction != "n/a":
            meta_info += f"\n  Jurisdiction: {jurisdiction}"
        if citations_found:
            meta_info += f"\n  Citations found: {len(citations_found)}"

        return (
            f"Ingested: {filepath.name}\n"
            f"  Chunks: {len(chunks)}\n"
            f"  Size: {filepath.stat().st_size} bytes\n"
            f"{meta_info}"
        )
    
    def query_knowledge(self, query: str, max_results: int = 5,
                        filter_extension: str = None,
                        filter_directory: str = None,
                        filter_doc_type: str = None,
                        filter_topic: str = None,
                        filter_jurisdiction: str = None,
                        rerank: bool = False) -> str:
        """
        Search the knowledge base for content relevant to a query.
        Returns the most relevant chunks from your ingested files.
        
        Filters:
          filter_doc_type: statute, case_law, legal_brief, news_article, statistics, code, document
          filter_topic: civil_rights, excessive_force, qualified_immunity, search_seizure, etc.
          filter_jurisdiction: federal, scotus, connecticut, federal_appellate, federal_district
          rerank: if True, uses LLM to re-rank top results for better precision
        """
        if self.collection.count() == 0:
            return "Knowledge base is empty. Use ingest_directory or ingest_file first."
        
        # Build ChromaDB where filter
        where_clauses = []
        if filter_extension:
            where_clauses.append({"extension": filter_extension})
        if filter_directory:
            where_clauses.append({"directory": filter_directory})
        if filter_doc_type:
            where_clauses.append({"doc_type": filter_doc_type})
        if filter_jurisdiction:
            where_clauses.append({"jurisdiction": filter_jurisdiction})
        if filter_topic:
            where_clauses.append({"topics": {"$contains": filter_topic}})
        
        where = None
        if len(where_clauses) == 1:
            where = where_clauses[0]
        elif len(where_clauses) > 1:
            where = {"$and": where_clauses}
        
        # Fetch more results if we're going to re-rank
        fetch_n = min(max_results * 3 if rerank else max_results, self.collection.count())
        
        # Embed the query
        query_embedding = self._embed([query])[0]
        
        # Search
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=fetch_n,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            # If filter fails, retry without it
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=fetch_n,
                include=["documents", "metadatas", "distances"]
            )
        
        if not results["documents"][0]:
            return f"No relevant results found for: '{query}'"
        
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        
        # Re-rank using LLM if requested
        if rerank and len(docs) > max_results:
            docs, metas, dists = self._rerank(query, docs, metas, dists, max_results)
        else:
            docs = docs[:max_results]
            metas = metas[:max_results]
            dists = dists[:max_results]
        
        # Format results
        lines = [f"Knowledge base results for: '{query}'\n"]
        
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
            similarity = max(0, 1 - dist)
            source = meta.get("filename", "unknown")
            chunk_idx = meta.get("chunk_index", "?")
            total = meta.get("total_chunks", "?")
            doc_type = meta.get("doc_type", "document")
            topics = meta.get("topics", "")
            jurisdiction = meta.get("jurisdiction", "")
            citations = meta.get("citations", "")
            
            header = f"── Result {i+1} (similarity: {similarity:.2%}, type: {doc_type}) ──"
            meta_line = f"Source: {meta.get('source', 'unknown')} (chunk {chunk_idx}/{total})"
            
            tag_parts = []
            if topics and topics != "general":
                tag_parts.append(f"Topics: {topics}")
            if jurisdiction and jurisdiction != "n/a" and jurisdiction != "unknown":
                tag_parts.append(f"Jurisdiction: {jurisdiction}")
            if citations:
                tag_parts.append(f"Citations: {citations}")
            tag_line = " | ".join(tag_parts)
            
            lines.append(header)
            lines.append(meta_line)
            if tag_line:
                lines.append(tag_line)
            lines.append(f"\n{doc}\n")
        
        return "\n".join(lines)
    
    def _rerank(self, query: str, docs: list, metas: list, dists: list,
                top_n: int) -> tuple:
        """Re-rank results using LLM relevance scoring."""
        try:
            import ollama
            
            # Build a concise summary of each doc for ranking
            summaries = []
            for i, (doc, meta) in enumerate(zip(docs, metas)):
                snippet = doc[:200].replace("\n", " ")
                source = meta.get("filename", "?")
                doc_type = meta.get("doc_type", "?")
                summaries.append(f"{i+1}. [{doc_type}] {source}: {snippet}...")
            
            prompt = (
                f"Rank these search results by relevance to the question: \"{query}\"\n\n"
                + "\n".join(summaries) + "\n\n"
                f"Return ONLY the numbers of the top {top_n} most relevant results, "
                f"in order from most to least relevant. Format: 1,3,7"
            )
            
            fast_model = self.config.get("llm", {}).get("fast_model", "llama3.1:8b")
            response = ollama.chat(
                model=fast_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0, "num_ctx": 4096}
            )
            
            # Parse rankings
            ranking_text = response["message"]["content"].strip()
            indices = []
            for num in re.findall(r'\d+', ranking_text):
                idx = int(num) - 1  # Convert to 0-indexed
                if 0 <= idx < len(docs) and idx not in indices:
                    indices.append(idx)
                    if len(indices) >= top_n:
                        break
            
            if indices:
                return (
                    [docs[i] for i in indices],
                    [metas[i] for i in indices],
                    [dists[i] for i in indices]
                )
        except Exception:
            pass
        
        # Fallback: return top N by vector similarity
        return docs[:top_n], metas[:top_n], dists[:top_n]
    
    def knowledge_stats(self) -> str:
        """Get statistics about the knowledge base."""
        stats = self.manifest["stats"]
        files = self.manifest["files"]
        
        if not files:
            return "Knowledge base is empty. Use ingest_directory to add files."
        
        # Count by extension
        ext_counts = {}
        total_size = 0
        for path, info in files.items():
            ext = Path(path).suffix.lower() or "(no ext)"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
            total_size += info.get("size", 0)
        
        # Count by directory (top-level grouping)
        dir_counts = {}
        home = str(Path.home())
        for path in files:
            rel = path.replace(home, "~")
            parts = Path(rel).parts
            top_dir = "/".join(parts[:3]) if len(parts) > 2 else "/".join(parts[:2])
            dir_counts[top_dir] = dir_counts.get(top_dir, 0) + 1
        
        lines = [
            "Knowledge Base Statistics:\n",
            f"  Total files:  {stats['total_files']}",
            f"  Total chunks: {stats['total_chunks']}",
            f"  Total size:   {total_size / 1024 / 1024:.1f} MB",
            f"  Database:     {self.db_path}\n",
            "  By file type:"
        ]
        for ext, count in sorted(ext_counts.items(), key=lambda x: -x[1]):
            lines.append(f"    {ext}: {count}")
        
        lines.append("\n  Top directories:")
        for d, count in sorted(dir_counts.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"    {d}: {count} files")
        
        return "\n".join(lines)
    
    def remove_source(self, path: str) -> str:
        """Remove a file or directory from the knowledge base."""
        target = Path(path).expanduser().resolve()
        str_target = str(target)
        
        removed = 0
        to_remove = []
        
        for fpath in list(self.manifest["files"].keys()):
            if fpath.startswith(str_target):
                to_remove.append(fpath)
        
        for fpath in to_remove:
            self._remove_file_chunks(fpath)
            del self.manifest["files"][fpath]
            removed += 1
        
        self.manifest["stats"]["total_files"] = len(self.manifest["files"])
        self.manifest["stats"]["total_chunks"] = self.collection.count()
        self._save_manifest()
        
        if removed == 0:
            return f"No entries found for: {target}"
        return f"Removed {removed} files from knowledge base."
    
    def list_sources(self, path_filter: str = None) -> str:
        """List all files currently in the knowledge base."""
        files = self.manifest["files"]
        
        if not files:
            return "Knowledge base is empty."
        
        if path_filter:
            files = {
                k: v for k, v in files.items()
                if path_filter.lower() in k.lower()
            }
        
        if not files:
            return f"No files matching '{path_filter}' in knowledge base."
        
        home = str(Path.home())
        lines = [f"Knowledge base sources ({len(files)} files):\n"]
        
        for fpath, info in sorted(files.items()):
            display = fpath.replace(home, "~")
            chunks = info.get("chunks", "?")
            ingested = info.get("ingested_at", "?")[:10]
            lines.append(f"  📄 {display}  ({chunks} chunks, ingested {ingested})")
        
        if len(lines) > 52:
            lines = lines[:50]
            lines.append(f"\n  ... and {len(files) - 49} more files")
        
        return "\n".join(lines)
    
    def _remove_file_chunks(self, filepath: str):
        """Remove all chunks for a specific file from ChromaDB."""
        try:
            # Find chunk IDs for this file
            results = self.collection.get(
                where={"source": filepath},
                include=[]
            )
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
        except Exception:
            pass
    
    # ── Tool definitions ────────────────────────────────────────────────────
    
    def get_tool_definitions(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "ingest_directory",
                    "description": "Scan a directory and add all supported files to the knowledge base. This makes them searchable by the AI. Supports text files, code, markdown, PDFs, configs, etc. Skips already-ingested unchanged files.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Directory path to ingest (default: ~)"},
                            "recursive": {"type": "boolean", "description": "Scan subdirectories (default: true)"},
                            "file_types": {"type": "string", "description": "'all', 'code', 'docs', or comma-separated extensions like '.py,.md,.txt'"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "ingest_file",
                    "description": "Add a single file to the knowledge base.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path to ingest"}
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "query_knowledge",
                    "description": "Search the knowledge base for information relevant to a question. Returns the most relevant chunks from ingested files. USE THIS to answer questions about the user's projects, code, notes, research, statutes, or case law. Supports filtering by document type, legal topic, and jurisdiction.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Natural language query or keywords"},
                            "max_results": {"type": "integer", "description": "Number of results (default: 5)"},
                            "filter_extension": {"type": "string", "description": "Filter by file extension (e.g., '.py')"},
                            "filter_directory": {"type": "string", "description": "Filter by directory path"},
                            "filter_doc_type": {"type": "string", "description": "Filter by document type: 'statute', 'case_law', 'legal_brief', 'news_article', 'statistics', 'code', 'document'"},
                            "filter_topic": {"type": "string", "description": "Filter by legal topic: 'civil_rights', 'excessive_force', 'qualified_immunity', 'search_seizure', 'due_process', 'traffic_stop', 'false_arrest', 'criminal_procedure', 'first_amendment', 'eighth_amendment'"},
                            "filter_jurisdiction": {"type": "string", "description": "Filter by jurisdiction: 'federal', 'scotus', 'connecticut', 'federal_appellate', 'federal_district'"},
                            "rerank": {"type": "boolean", "description": "If true, uses LLM to re-rank results for better accuracy (slower but more precise). Recommended for legal queries."}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "knowledge_stats",
                    "description": "Get statistics about the knowledge base — file counts, types, sizes, and directories.",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "remove_source",
                    "description": "Remove a file or all files under a directory from the knowledge base.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File or directory path to remove"}
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_sources",
                    "description": "List all files currently in the knowledge base, optionally filtered by path.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path_filter": {"type": "string", "description": "Filter results by path substring"}
                        }
                    }
                }
            }
        ]
    
    def get_handlers(self) -> dict[str, Callable]:
        return {
            "ingest_directory": self.ingest_directory,
            "ingest_file": self.ingest_file,
            "query_knowledge": self.query_knowledge,
            "knowledge_stats": self.knowledge_stats,
            "remove_source": self.remove_source,
            "list_sources": self.list_sources,
        }