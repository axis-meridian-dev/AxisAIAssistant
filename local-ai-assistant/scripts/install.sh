#!/usr/bin/env bash
# ============================================================================
# INSTALL SCRIPT — Place downloaded files into correct project structure
#
# Usage:
#   1. Download all files from Claude into a folder (e.g. ~/Downloads/)
#   2. Run this script from that folder:
#      cd ~/Downloads
#      chmod +x install.sh && ./install.sh
#
# This creates the full project at:
#   ~/Desktop/AxisAIAssistant/local-ai-assistant/
# ============================================================================
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
info()  { echo -e "${GREEN}[+]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }

# Where to install
PROJECT="$HOME/Desktop/AxisAIAssistant/local-ai-assistant"

# Where the downloaded files are (same dir as this script)
SRC="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "============================================"
echo "  Installing Local AI Assistant"
echo "  Target: $PROJECT"
echo "============================================"
echo ""

# Create full directory structure
info "Creating directory structure..."
mkdir -p "$PROJECT/tools"
mkdir -p "$PROJECT/config"
mkdir -p "$PROJECT/scripts"
mkdir -p "$PROJECT/searxng"

# ─── Root-level Python files ──────────────────────────────────────────────
info "Copying root files..."
for f in main.py agent.py config.py watcher.py voice.py; do
    if [ -f "$SRC/$f" ]; then
        cp "$SRC/$f" "$PROJECT/$f"
        echo "  ✓ $f"
    else
        warn "  ✗ $f not found in $SRC"
    fi
done

# ─── Tools directory ──────────────────────────────────────────────────────
info "Copying tool modules..."
for f in __init__.py base.py file_manager.py web_search.py desktop_control.py \
         system_info.py knowledge_base.py legal_research.py document_writer.py; do
    if [ -f "$SRC/$f" ]; then
        cp "$SRC/$f" "$PROJECT/tools/$f"
        echo "  ✓ tools/$f"
    else
        warn "  ✗ $f not found in $SRC"
    fi
done

# ─── Config ───────────────────────────────────────────────────────────────
info "Copying config..."
if [ -f "$SRC/settings.json" ]; then
    # Don't overwrite existing config
    if [ ! -f "$PROJECT/config/settings.json" ]; then
        cp "$SRC/settings.json" "$PROJECT/config/settings.json"
        echo "  ✓ config/settings.json"
    else
        echo "  ⏭  config/settings.json already exists (not overwriting)"
    fi
fi

# ─── Scripts ──────────────────────────────────────────────────────────────
info "Copying scripts..."
if [ -f "$SRC/setup.sh" ]; then
    cp "$SRC/setup.sh" "$PROJECT/scripts/setup.sh"
    chmod +x "$PROJECT/scripts/setup.sh"
    echo "  ✓ scripts/setup.sh"
fi
if [ -f "$SRC/setup_knowledge_base.sh" ]; then
    cp "$SRC/setup_knowledge_base.sh" "$PROJECT/scripts/setup_knowledge_base.sh"
    chmod +x "$PROJECT/scripts/setup_knowledge_base.sh"
    echo "  ✓ scripts/setup_knowledge_base.sh"
fi
if [ -f "$SRC/local-ai-assistant.service" ]; then
    cp "$SRC/local-ai-assistant.service" "$PROJECT/scripts/local-ai-assistant.service"
    echo "  ✓ scripts/local-ai-assistant.service"
fi

# ─── SearXNG ──────────────────────────────────────────────────────────────
info "Copying SearXNG config..."
if [ -f "$SRC/searxng-docker-compose.yml" ]; then
    cp "$SRC/searxng-docker-compose.yml" "$PROJECT/searxng/docker-compose.yml"
    echo "  ✓ searxng/docker-compose.yml"
fi
if [ -f "$SRC/searxng-settings.yml" ]; then
    cp "$SRC/searxng-settings.yml" "$PROJECT/searxng/settings.yml"
    echo "  ✓ searxng/settings.yml"
fi

# ─── README ───────────────────────────────────────────────────────────────
if [ -f "$SRC/README.md" ]; then
    cp "$SRC/README.md" "$PROJECT/README.md"
    echo "  ✓ README.md"
fi

# ─── Create data directories ─────────────────────────────────────────────
info "Creating data directories..."
mkdir -p ~/.local/share/ai-assistant/knowledge_db
mkdir -p ~/LegalResearch/{federal_statutes,state_statutes,case_law,statistics,news_clips,research_briefs,projects,documents}

echo ""
echo "============================================"
info "  Files installed!"
echo "============================================"
echo ""
info "Project structure:"
find "$PROJECT" -type f | sort | head -30
echo ""

info "Next steps:"
echo ""
echo "  # 1. Go to the project"
echo "  cd $PROJECT"
echo ""
echo "  # 2. Run full setup (installs Ollama, models, Python packages)"
echo "  chmod +x scripts/setup.sh"
echo "  ./scripts/setup.sh"
echo ""
echo "  # 3. Or if you already have .venv and packages, just run:"
echo "  source .venv/bin/activate"
echo "  python main.py"
echo ""
