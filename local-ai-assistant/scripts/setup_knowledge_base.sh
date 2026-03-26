#!/usr/bin/env bash
# ============================================================================
# Knowledge Base Setup — adds RAG capabilities
# Run AFTER setup.sh or after pip installing base dependencies
# ============================================================================
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[+]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# Activate venv if not already
if [ -z "${VIRTUAL_ENV:-}" ]; then
    source .venv/bin/activate
fi

info "Installing knowledge base dependencies..."
pip install chromadb PyMuPDF watchdog

# Pull the embedding model
info "Pulling embedding model (nomic-embed-text)..."
if command -v ollama &>/dev/null; then
    ollama pull nomic-embed-text
else
    warn "Ollama not found. Install it first, then run: ollama pull nomic-embed-text"
fi

# Create directories
mkdir -p ~/.local/share/ai-assistant/knowledge_db
mkdir -p ~/LegalResearch/{federal_statutes,state_statutes,case_law,statistics,news_clips,research_briefs,projects,documents}

echo ""
info "Knowledge Base ready!"
echo ""
info "Quick start (inside the assistant CLI):"
echo "  > ingest directory ~/Documents"
echo "  > ingest directory ~/LegalResearch"
echo "  > what do my files say about [topic]?"
echo ""
echo "  Auto-watch (separate terminal):"
echo "  python watcher.py ~/Documents ~/LegalResearch"
echo ""
