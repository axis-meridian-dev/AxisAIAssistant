#!/usr/bin/env bash
# ============================================================================
# LOCAL AI ASSISTANT — Setup Script
# Installs: Ollama, LLM models, Python deps, SearXNG, system control tools
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

# ─── Detect package manager ────────────────────────────────────────────────
if command -v apt-get &>/dev/null; then
    PKG="apt"
    INSTALL="sudo apt-get install -y"
    UPDATE="sudo apt-get update"
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
$UPDATE

if [ "$PKG" = "apt" ]; then
    $INSTALL curl wget git python3 python3-pip python3-venv \
        xdotool wmctrl xclip xdg-utils playerctl \
        portaudio19-dev libsndfile1 ffmpeg \
        docker.io docker-compose
elif [ "$PKG" = "dnf" ]; then
    $INSTALL curl wget git python3 python3-pip \
        xdotool wmctrl xclip xdg-utils playerctl \
        portaudio-devel libsndfile ffmpeg \
        docker docker-compose
elif [ "$PKG" = "pacman" ]; then
    $INSTALL curl wget git python python-pip \
        xdotool wmctrl xclip xdg-utils playerctl \
        portaudio libsndfile ffmpeg \
        docker docker-compose
fi

# ─── Install Ollama ────────────────────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
    info "Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
else
    info "Ollama already installed: $(ollama --version)"
fi

# Start Ollama service
info "Starting Ollama service..."
if systemctl is-active --quiet ollama 2>/dev/null; then
    info "Ollama service already running."
else
    sudo systemctl enable ollama 2>/dev/null || true
    sudo systemctl start ollama 2>/dev/null || ollama serve &>/dev/null &
    sleep 3
fi

# ─── Pull LLM models ──────────────────────────────────────────────────────
info "Pulling LLM models (this will take a while on first run)..."

# Primary reasoning model — 70B for your 24GB+ setup
# Uses Q4 quantization to fit in VRAM
info "Pulling llama3.1:70b (primary model — ~40GB download)..."
ollama pull llama3.1:70b || {
    warn "70B model failed — falling back to 8B for now"
    ollama pull llama3.1:8b
}

# Fast model for simple tasks (routing, classification)
info "Pulling llama3.1:8b (fast router model)..."
ollama pull llama3.1:8b

info "Models ready:"
ollama list

# ─── Python virtual environment ───────────────────────────────────────────
info "Setting up Python environment..."
cd "$PROJECT_DIR"

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip

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
    i3ipc \
    faster-whisper \
    sounddevice \
    numpy \
    piper-tts \
    openwakeword \
    platformdirs

# ─── SearXNG (self-hosted search) ─────────────────────────────────────────
info "Setting up SearXNG (local search engine)..."

SEARXNG_DIR="$PROJECT_DIR/searxng"
mkdir -p "$SEARXNG_DIR"

cat > "$SEARXNG_DIR/docker-compose.yml" << 'YAML'
version: '3.7'
services:
  searxng:
    image: searxng/searxng:latest
    container_name: searxng
    ports:
      - "8888:8080"
    volumes:
      - ./settings.yml:/etc/searxng/settings.yml:ro
    environment:
      - SEARXNG_BASE_URL=http://localhost:8888/
    restart: unless-stopped
YAML

cat > "$SEARXNG_DIR/settings.yml" << 'YAML'
use_default_settings: true
server:
  secret_key: "$(openssl rand -hex 32)"
  bind_address: "0.0.0.0"
  port: 8080
search:
  formats:
    - html
    - json
engines:
  - name: google
    engine: google
    shortcut: g
  - name: duckduckgo
    engine: duckduckgo
    shortcut: ddg
  - name: wikipedia
    engine: wikipedia
    shortcut: wp
  - name: github
    engine: github
    shortcut: gh
YAML

# Start SearXNG
cd "$SEARXNG_DIR"
sudo docker-compose up -d 2>/dev/null || {
    warn "Docker not available or failed. Web search will fall back to DuckDuckGo API."
    warn "To enable SearXNG later: cd $SEARXNG_DIR && docker-compose up -d"
}
cd "$PROJECT_DIR"

# ─── Create config ─────────────────────────────────────────────────────────
info "Creating default config..."
cat > "$PROJECT_DIR/config/settings.json" << JSON
{
    "llm": {
        "primary_model": "llama3.1:70b",
        "fast_model": "llama3.1:8b",
        "ollama_host": "http://localhost:11434",
        "temperature": 0.3,
        "context_window": 8192
    },
    "search": {
        "searxng_url": "http://localhost:8888",
        "fallback_to_ddg": true,
        "max_results": 5
    },
    "voice": {
        "enabled": false,
        "wake_word": "computer",
        "stt_model": "base.en",
        "tts_voice": "en_US-lessac-medium",
        "push_to_talk_key": "ctrl+space"
    },
    "files": {
        "allowed_roots": ["~"],
        "excluded_dirs": [".git", "node_modules", "__pycache__", ".cache"],
        "max_file_size_mb": 50
    },
    "desktop": {
        "screenshot_enabled": true,
        "app_launch_enabled": true,
        "clipboard_enabled": true
    }
}
JSON

# ─── Systemd service (optional) ───────────────────────────────────────────
info "Creating systemd service file (install with: sudo cp ... /etc/systemd/system/)..."
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
info "============================================"
info "  Setup complete!"
info "============================================"
echo ""
info "To run the assistant:"
echo "  cd $PROJECT_DIR"
echo "  source .venv/bin/activate"
echo "  python main.py"
echo ""
info "To install as a service:"
echo "  sudo cp scripts/local-ai-assistant.service /etc/systemd/system/"
echo "  sudo systemctl enable --now local-ai-assistant"
echo ""
warn "Note: 70B model needs ~40GB disk + 24GB+ VRAM."
warn "If it's slow, edit config/settings.json → primary_model to 'llama3.1:8b'"
