#!/usr/bin/env bash
# filepath: /Users/anirudhnishtala/Desktop/Brand-Genome-Engine/start.sh
# ─────────────────────────────────────────────────────────────
#  Brand Genome Engine — single-command startup
#  Launches FastAPI backend (:8000) and Vite frontend (:5173)
#  Usage:  ./start.sh          (start both)
#          ./start.sh --stop   (kill both)
# ─────────────────────────────────────────────────────────────
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIDFILE="$ROOT_DIR/.dev-pids"

# ── Colours ───────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

# ── Stop helper ───────────────────────────────────────────────
kill_tree() {
    # Kill a process and all its children (entire process group)
    local pid=$1
    if kill -0 "$pid" 2>/dev/null; then
        # Kill the entire process group rooted at this PID
        pkill -P "$pid" 2>/dev/null          # kill children first
        kill "$pid" 2>/dev/null               # then the parent
        sleep 0.3
        # Force-kill anything still alive
        pkill -9 -P "$pid" 2>/dev/null
        kill -9 "$pid" 2>/dev/null
        echo -e "  Killed PID $pid (and children)"
    fi
}

stop_services() {
    if [[ -f "$PIDFILE" ]]; then
        echo -e "${CYAN}Stopping services...${NC}"
        while IFS= read -r pid; do
            kill_tree "$pid"
        done < "$PIDFILE"
        rm -f "$PIDFILE"
    fi
    # Also kill any leftover uvicorn / vite on the expected ports
    lsof -ti:8000 2>/dev/null | xargs kill -9 2>/dev/null || true
    lsof -ti:5173 2>/dev/null | xargs kill -9 2>/dev/null || true
    echo -e "${GREEN}All services stopped.${NC}"
}

if [[ "${1:-}" == "--stop" ]]; then
    stop_services
    exit 0
fi

# ── Pre-flight checks ────────────────────────────────────────
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" &>/dev/null; then
    echo -e "${RED}Error: $PYTHON not found. Install Python 3.9+ or set PYTHON=...${NC}"
    exit 1
fi
if ! command -v node &>/dev/null; then
    echo -e "${RED}Error: node not found. Install Node.js 18+.${NC}"
    exit 1
fi

# ── Install deps if needed ───────────────────────────────────
if ! "$PYTHON" -c "import uvicorn" 2>/dev/null; then
    echo -e "${CYAN}Installing Python dependencies...${NC}"
    "$PYTHON" -m pip install -r "$ROOT_DIR/requirements.txt" --quiet
fi

if [[ ! -d "$ROOT_DIR/frontend/node_modules" ]]; then
    echo -e "${CYAN}Installing frontend dependencies...${NC}"
    (cd "$ROOT_DIR/frontend" && npm install --silent)
fi

# ── Kill any previous run ────────────────────────────────────
stop_services 2>/dev/null || true

# ── Start backend ─────────────────────────────────────────────
echo -e "${BOLD}${GREEN}Starting FastAPI backend on http://localhost:8000 ...${NC}"
(cd "$ROOT_DIR" && "$PYTHON" -m uvicorn src.api.main:app \
    --host 0.0.0.0 --port 8000 --reload \
    --log-level info) &
BACKEND_PID=$!

# ── Start frontend ────────────────────────────────────────────
echo -e "${BOLD}${GREEN}Starting Vite frontend on http://localhost:5173 ...${NC}"
(cd "$ROOT_DIR/frontend" && npx vite --host 127.0.0.1 --port 5173) &
FRONTEND_PID=$!

# ── Save PIDs for --stop ─────────────────────────────────────
echo "$BACKEND_PID" >  "$PIDFILE"
echo "$FRONTEND_PID" >> "$PIDFILE"

echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║  Brand Genome Engine is running                  ║${NC}"
echo -e "${BOLD}${CYAN}║                                                  ║${NC}"
echo -e "${BOLD}${CYAN}║  Frontend  →  http://localhost:5173              ║${NC}"
echo -e "${BOLD}${CYAN}║  Backend   →  http://localhost:8000/docs         ║${NC}"
echo -e "${BOLD}${CYAN}║                                                  ║${NC}"
echo -e "${BOLD}${CYAN}║  Press Ctrl+C to stop both services              ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# ── Trap Ctrl+C to clean up ──────────────────────────────────
cleanup() {
    echo ""
    echo -e "${CYAN}Shutting down...${NC}"
    kill_tree "$BACKEND_PID"
    kill_tree "$FRONTEND_PID"
    # Fallback: force-free the ports in case child processes survived
    lsof -ti:8000 2>/dev/null | xargs kill -9 2>/dev/null || true
    lsof -ti:5173 2>/dev/null | xargs kill -9 2>/dev/null || true
    rm -f "$PIDFILE"
    echo -e "${GREEN}Done.${NC}"
    exit 0
}
trap cleanup SIGINT SIGTERM

# Wait for both — if either exits, shut down the other
wait -n "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null
cleanup
