#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  Brand Genome Engine — single-command startup
#  Launches FastAPI backend (:8000) and Vite frontend (:5173)
#  Usage:  ./start.sh          (start both)
#          ./start.sh --stop   (kill both)
# ─────────────────────────────────────────────────────────────

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIDFILE="$ROOT_DIR/.dev-pids"
BACKEND_PID=""
FRONTEND_PID=""

# ── Colours ───────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

# ── Kill helpers ──────────────────────────────────────────────
kill_port() {
    # Force-kill every process listening on a given port.
    local port="$1"
    local pids
    pids=$(lsof -ti:"$port" 2>/dev/null) || true
    if [[ -n "$pids" ]]; then
        echo "$pids" | xargs kill -9 2>/dev/null || true
        echo -e "  Freed port $port"
    fi
}

nuke_services() {
    # Kill by process name — the most reliable method on macOS.
    # Catches orphaned / reparented processes that PID-tracking misses.
    pkill -f "uvicorn src.api.main" 2>/dev/null || true
    pkill -f "node.*vite"           2>/dev/null || true
    sleep 0.5
    pkill -9 -f "uvicorn src.api.main" 2>/dev/null || true
    pkill -9 -f "node.*vite"           2>/dev/null || true

    # Final fallback: free the ports directly
    kill_port 8000
    kill_port 5173
}

stop_services() {
    echo -e "${CYAN}Stopping services...${NC}"

    # 1. Kill by saved PIDs (if file exists from a prior run)
    if [[ -f "$PIDFILE" ]]; then
        while IFS= read -r pid; do
            if [[ -n "$pid" ]]; then
                kill "$pid" 2>/dev/null || true
                kill -9 "$pid" 2>/dev/null || true
            fi
        done < "$PIDFILE"
        rm -f "$PIDFILE"
    fi

    # 2. Kill by process name + free ports (always runs)
    nuke_services

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
# Launch uvicorn directly (no subshell wrapper) so $! is the real PID.
echo -e "${BOLD}${GREEN}Starting FastAPI backend on http://localhost:8000 ...${NC}"
cd "$ROOT_DIR"
"$PYTHON" -m uvicorn src.api.main:app \
    --host 0.0.0.0 --port 8000 --reload \
    --log-level info &
BACKEND_PID=$!

# ── Start frontend ────────────────────────────────────────────
# Use node to run vite directly (not via npx wrapper which can exit early).
echo -e "${BOLD}${GREEN}Starting Vite frontend on http://localhost:5173 ...${NC}"
cd "$ROOT_DIR/frontend"
node node_modules/.bin/vite --host 127.0.0.1 --port 5173 &
FRONTEND_PID=$!

cd "$ROOT_DIR"

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

# ── Trap Ctrl+C / SIGTERM to clean up ────────────────────────
SHUTTING_DOWN=0
cleanup() {
    # Prevent the trap from firing more than once
    [[ "$SHUTTING_DOWN" -eq 1 ]] && return
    SHUTTING_DOWN=1
    trap - SIGINT SIGTERM

    echo ""
    echo -e "${CYAN}Shutting down...${NC}"

    # Kill the direct PIDs we launched
    [[ -n "$BACKEND_PID"  ]] && kill "$BACKEND_PID"  2>/dev/null || true
    [[ -n "$FRONTEND_PID" ]] && kill "$FRONTEND_PID" 2>/dev/null || true
    sleep 0.5
    [[ -n "$BACKEND_PID"  ]] && kill -9 "$BACKEND_PID"  2>/dev/null || true
    [[ -n "$FRONTEND_PID" ]] && kill -9 "$FRONTEND_PID" 2>/dev/null || true

    # Kill any spawned workers / watchers by name + free ports
    nuke_services

    rm -f "$PIDFILE"
    echo -e "${GREEN}Done.${NC}"
    exit 0
}
trap cleanup SIGINT SIGTERM

# ── Keep the script alive until both processes exit ───────────
# Poll every 2 seconds. If either backend or frontend dies, clean up.
while true; do
    # Check if both are still running
    backend_alive=0
    frontend_alive=0
    kill -0 "$BACKEND_PID"  2>/dev/null && backend_alive=1
    kill -0 "$FRONTEND_PID" 2>/dev/null && frontend_alive=1

    # Also check if something is actually listening on the ports
    # (covers the case where PID exists but process hasn't bound yet)
    if [[ "$backend_alive" -eq 0 ]] && [[ "$frontend_alive" -eq 0 ]]; then
        echo -e "${RED}Both services exited unexpectedly.${NC}"
        cleanup
    fi

    sleep 2
done
