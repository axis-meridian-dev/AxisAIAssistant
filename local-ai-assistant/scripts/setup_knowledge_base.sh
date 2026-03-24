#!/usr/bin/env bash
# ============================================================================
# Knowledge Base Setup — adds RAG capabilities to the local AI assistant
# Run this AFTER the main setup.sh or after pip installing base dependencies
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

pip install \
    chromadb \
    PyMuPDF \
    watchdog

# Pull the embedding model via Ollama
info "Pulling embedding model (nomic-embed-text)..."
if command -v ollama &>/dev/null; then
    ollama pull nomic-embed-text
else
    warn "Ollama not found. Install it first, then run: ollama pull nomic-embed-text"
fi

# Create the database directory
mkdir -p ~/.local/share/ai-assistant/knowledge_db

echo ""
info "============================================"
info "  Knowledge Base ready!"
info "============================================"
echo ""
info "Quick start:"
echo ""
echo "  # In the assistant CLI, just say:"
echo "  > ingest my Documents folder"
echo "  > ingest my projects directory"
echo "  > what do my notes say about [topic]?"
echo ""
echo "  # Or ingest everything at once:"
echo "  > ingest all files under my home directory"
echo ""
echo "  # Auto-watch for changes (separate terminal):"
echo "  python watcher.py ~/Documents ~/Projects"
echo ""
info "The embedding model (nomic-embed-text) is ~275MB and runs on CPU."
info "ChromaDB stores vectors locally at ~/.local/share/ai-assistant/knowledge_db"
