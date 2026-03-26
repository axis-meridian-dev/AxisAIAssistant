#!/usr/bin/env bash
# ============================================================================
# LOCAL AI ASSISTANT — Full Setup Script
# Installs: System deps, Ollama, LLM models, Python venv + packages,
#           SearXNG (optional), knowledge base, systemd service
# Tested on: Ubuntu 22.04+ / Debian 12+ / Fedora 38+
# ============================================================================
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[+]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
error() { echo -e "${RED}[✗]${NC} $1"; }

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo ""
echo "============================================"
echo "  LOCAL AI ASSISTANT — Full Setup"
echo "============================================"
echo ""
info "Project directory: $PROJECT_DIR"
echo ""

# ─── Detect package manager ────────────────────────────────────────────────
if command -v apt-get &>/dev/null; then
    PKG="apt"
    INSTALL="sudo apt-get install -y"
    UPDATE="sudo apt-get update --fix-missing"
elif command -v dnf &>/dev/null; then
    PKG="dnf"
    INSTALL="sudo dnf install -y"
    UPDATE="sudo dnf check-update || true"
elif command -v pacman &>/dev/null; then
    PKG="pacman"
    INSTALL="sudo pacman -S --noconfirm"
    UPDATE="sudo pacman -Sy"
else
    error "Unsupported package manager. Install dependencies manually."
    exit 1
fi

# ─── System dependencies ───────────────────────────────────────────────────
info "Installing system dependencies..."
$UPDATE || warn "apt update had errors (mirror sync issue — continuing anyway)"

if [ "$PKG" = "apt" ]; then
    $INSTALL curl wget git python3 python3-pip python3-venv \
        xdotool wmctrl xclip xdg-utils \
        portaudio19-dev libsndfile1 ffmpeg scrot espeak \
        2>/dev/null || warn "Some packages failed — continuing"
elif [ "$PKG" = "dnf" ]; then
    $INSTALL curl wget git python3 python3-pip \
        xdotool wmctrl xclip xdg-utils \
        portaudio-devel libsndfile ffmpeg scrot espeak \
        2>/dev/null || warn "Some packages failed — continuing"
elif [ "$PKG" = "pacman" ]; then
    $INSTALL curl wget git python python-pip \
        xdotool wmctrl xclip xdg-utils \
        portaudio libsndfile ffmpeg scrot espeak \
        2>/dev/null || warn "Some packages failed — continuing"
fi

info "System dependencies done."

# ─── Install Ollama ────────────────────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
    info "Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
else
    info "Ollama already installed: $(ollama --version 2>/dev/null || echo 'unknown version')"
fi

# Start Ollama service
info "Starting Ollama service..."
if systemctl is-active --quiet ollama 2>/dev/null; then
    info "Ollama service already running."
else
    sudo systemctl enable ollama 2>/dev/null || true
    sudo systemctl start ollama 2>/dev/null || {
        warn "systemctl failed — starting ollama manually in background"
        nohup ollama serve &>/dev/null &
        sleep 3
    }
fi

# ─── Pull LLM models ──────────────────────────────────────────────────────
info "Pulling LLM models..."

# Fast model for routing + simple tasks
info "Pulling llama3.1:8b (~4GB)..."
ollama pull llama3.1:8b || warn "Failed to pull llama3.1:8b — check connection"

# Embedding model for knowledge base
info "Pulling nomic-embed-text (~275MB)..."
ollama pull nomic-embed-text || warn "Failed to pull nomic-embed-text"

echo ""
info "Models available:"
ollama list 2>/dev/null || warn "Could not list models"
echo ""

warn "To pull the full 70B model later (requires 24GB+ VRAM, ~40GB download):"
echo "  ollama pull llama3.1:70b"
echo "  Then edit config/settings.json → primary_model to 'llama3.1:70b'"
echo ""

# ─── Python virtual environment ───────────────────────────────────────────
info "Setting up Python environment..."
cd "$PROJECT_DIR"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    info "Created virtual environment"
else
    info "Virtual environment already exists"
fi

source .venv/bin/activate

pip install --upgrade pip

info "Installing Python packages..."
pip install \
    ollama \
    httpx \
    rich \
    pydantic \
    watchdog \
    duckduckgo-search \
    readability-lxml \
    beautifulsoup4 \
    lxml \
    pynput \
    pyperclip \
    psutil \
    python-xlib \
    sounddevice \
    numpy \
    chromadb \
    PyMuPDF \
    platformdirs

info "Python packages installed."

# ─── Create directory structure ────────────────────────────────────────────
info "Creating data directories..."
mkdir -p "$PROJECT_DIR/config"
mkdir -p ~/.local/share/ai-assistant/knowledge_db
mkdir -p ~/LegalResearch/{federal_statutes,state_statutes,case_law,statistics,news_clips,research_briefs,projects,documents}

# Copy default config if not present
if [ ! -f "$PROJECT_DIR/config/settings.json" ]; then
    cp "$PROJECT_DIR/settings.json" "$PROJECT_DIR/config/settings.json" 2>/dev/null || {
        warn "settings.json not found in project root — using defaults from config.py"
    }
fi

# ─── Systemd service (optional) ───────────────────────────────────────────
info "Creating systemd service file..."
cat > "$PROJECT_DIR/scripts/local-ai-assistant.service" << SERVICE
[Unit]
Description=Local AI Assistant
After=network.target ollama.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$PROJECT_DIR/.venv/bin/python $PROJECT_DIR/main.py
Restart=on-failure
RestartSec=5
Environment=DISPLAY=:0
Environment=XAUTHORITY=/home/$USER/.Xauthority

[Install]
WantedBy=default.target
SERVICE

echo ""
echo "============================================"
info "  Setup complete!"
echo "============================================"
echo ""
info "To run the assistant:"
echo "  cd $PROJECT_DIR"
echo "  source .venv/bin/activate"
echo "  python main.py"
echo ""
info "To run with voice:"
echo "  pip install faster-whisper piper-tts openwakeword"
echo "  python main.py --voice"
echo ""
info "To run the file watcher (separate terminal):"
echo "  source .venv/bin/activate"
echo "  python watcher.py ~/Documents ~/Desktop ~/LegalResearch"
echo ""
info "To install as a service:"
echo "  sudo cp scripts/local-ai-assistant.service /etc/systemd/system/"
echo "  sudo systemctl enable --now local-ai-assistant"
echo ""
info "First steps inside the assistant:"
echo "  > download the civil rights statute collection"
echo "  > ingest directory ~/LegalResearch"
echo "  > search case law for excessive force traffic stop"
echo ""
