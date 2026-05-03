#!/bin/bash
# scripts/setup.sh — One-command setup for YouTube Knowledge Engine
# Usage: chmod +x scripts/setup.sh && ./scripts/setup.sh

set -e  # Exit on error

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_step() { echo -e "\n${CYAN}${BOLD}[STEP]${NC} $1"; }
print_ok()   { echo -e "${GREEN}✓${NC} $1"; }
print_warn() { echo -e "${YELLOW}⚠${NC}  $1"; }
print_err()  { echo -e "${RED}✗${NC} $1"; }

echo -e "${BOLD}"
echo "╔══════════════════════════════════════════════════╗"
echo "║   YouTube Knowledge Engine — Setup Script        ║"
echo "║   Production RAG System                          ║"
echo "╚══════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── 1. Check Python version ───────────────────────────────────────────────────
print_step "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED="3.10"
if python3 -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)"; then
    print_ok "Python $PYTHON_VERSION found"
else
    print_err "Python 3.10+ required. Found: $PYTHON_VERSION"
    exit 1
fi

# ── 2. Check FFmpeg ───────────────────────────────────────────────────────────
print_step "Checking FFmpeg..."
if command -v ffmpeg &> /dev/null; then
    FFMPEG_VER=$(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')
    print_ok "FFmpeg $FFMPEG_VER found"
else
    print_warn "FFmpeg not found. Installing..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update -q && sudo apt-get install -y ffmpeg
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install ffmpeg
    else
        print_err "Please install FFmpeg manually: https://ffmpeg.org/download.html"
        exit 1
    fi
fi

# ── 3. Create virtual environment ─────────────────────────────────────────────
print_step "Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_ok "Virtual environment created at ./venv"
else
    print_ok "Virtual environment already exists"
fi

source venv/bin/activate

# ── 4. Upgrade pip ────────────────────────────────────────────────────────────
print_step "Upgrading pip..."
pip install --upgrade pip --quiet
print_ok "pip upgraded"

# ── 5. Install dependencies ───────────────────────────────────────────────────
print_step "Installing Python dependencies (this may take 3-5 minutes)..."
pip install -r requirements.txt --quiet
print_ok "Dependencies installed"

# ── 6. Setup .env ─────────────────────────────────────────────────────────────
print_step "Setting up environment configuration..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    print_ok ".env file created from template"
    echo ""
    print_warn "IMPORTANT: You must set your GROQ_API_KEY in .env before running!"
    echo -e "  ${CYAN}Get a free key at: https://console.groq.com${NC}"
    echo ""
else
    print_ok ".env already exists"
fi

# ── 7. Create data directories ────────────────────────────────────────────────
print_step "Creating data directories..."
mkdir -p data/chroma_db data/logs data/cache/embeddings
print_ok "Data directories created"

# ── 8. Verify GROQ_API_KEY ────────────────────────────────────────────────────
print_step "Checking GROQ_API_KEY..."
if grep -q "your_groq_api_key_here" .env; then
    print_warn "GROQ_API_KEY not set in .env — please add it before querying"
else
    print_ok "GROQ_API_KEY is configured"
fi

# ── 9. Run unit tests ─────────────────────────────────────────────────────────
print_step "Running unit tests..."
if python -m pytest tests/unit/ -q --tb=short 2>&1; then
    print_ok "All unit tests passed"
else
    print_warn "Some tests failed — check output above"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  Setup complete! 🎉${NC}"
echo -e "${GREEN}${BOLD}════════════════════════════════════════${NC}"
echo ""
echo -e "Next steps:"
echo -e "  1. ${CYAN}Edit .env${NC} → add your GROQ_API_KEY"
echo -e "  2. ${CYAN}source venv/bin/activate${NC}"
echo -e "  3. Start backend:  ${CYAN}uvicorn backend.main:app --reload --port 8000${NC}"
echo -e "  4. Start frontend: ${CYAN}streamlit run frontend/app.py${NC}"
echo -e "  5. Or use Docker:  ${CYAN}docker-compose up --build${NC}"
echo ""
echo -e "  API docs: ${CYAN}http://localhost:8000/docs${NC}"
echo -e "  Frontend: ${CYAN}http://localhost:8501${NC}"
echo ""
