#!/usr/bin/env bash
# ==============================================================================
# MessMate - Live Physical Smartphone & Dev Backend Server Runner
# ==============================================================================
# Binds FastAPI backend to 0.0.0.0:8000 for local network access (Wi-Fi),
# configures automatic USB adb reverse tunneling for wired phones, and
# displays clean connection URLs.
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"

# Formatting Colors
BOLD='\033[1m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}${BOLD}[*] Initializing MessMate Phone Server Runner...${NC}"

# 1. Resolve local Wi-Fi LAN IP
LAN_IP=""
if command -v ipconfig &>/dev/null; then
    LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)
fi
if [ -z "$LAN_IP" ] && command -v hostname &>/dev/null; then
    LAN_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || true)
fi
if [ -z "$LAN_IP" ]; then
    LAN_IP="10.5.138.91"
fi

# 2. Locate ADB binary
ADB_BIN=""
if command -v adb &>/dev/null; then
    ADB_BIN="$(command -v adb)"
elif [ -x "$HOME/Library/Android/sdk/platform-tools/adb" ]; then
    ADB_BIN="$HOME/Library/Android/sdk/platform-tools/adb"
elif [ -n "$ANDROID_HOME" ] && [ -x "$ANDROID_HOME/platform-tools/adb" ]; then
    ADB_BIN="$ANDROID_HOME/platform-tools/adb"
elif [ -n "$ANDROID_SDK_ROOT" ] && [ -x "$ANDROID_SDK_ROOT/platform-tools/adb" ]; then
    ADB_BIN="$ANDROID_SDK_ROOT/platform-tools/adb"
fi

# 3. Check for USB-connected Android phone & configure reverse tunnel
if [ -n "$ADB_BIN" ]; then
    CONNECTED_DEVICES=$("$ADB_BIN" devices 2>/dev/null | awk 'NR>1 && $2=="device" {print $1}')
    if [ -n "$CONNECTED_DEVICES" ]; then
        echo -e "${GREEN}[+] Detected USB Android device(s):${NC} $CONNECTED_DEVICES"
        if "$ADB_BIN" reverse tcp:8000 tcp:8000 2>/dev/null; then
            echo -e "${GREEN}[+] ADB reverse tunnel established: tcp:8000 -> tcp:8000${NC}"
        else
            echo -e "${YELLOW}[!] Warning: 'adb reverse tcp:8000 tcp:8000' returned non-zero.${NC}"
        fi
    else
        echo -e "${YELLOW}[i] No USB Android device currently attached.${NC}"
        echo -e "    Phone can connect over Wi-Fi LAN (http://${LAN_IP}:8000)"
        echo -e "    or plug in USB cable with USB Debugging enabled for direct tunnel."
    fi
else
    echo -e "${YELLOW}[i] ADB not found on PATH or Android SDK platform-tools.${NC}"
fi

# 4. Resolve Python / Uvicorn executable
UVICORN_BIN=""
if [ -x "$BACKEND_DIR/.venv/bin/uvicorn" ]; then
    UVICORN_BIN="$BACKEND_DIR/.venv/bin/uvicorn"
elif command -v uvicorn &>/dev/null; then
    UVICORN_BIN="$(command -v uvicorn)"
elif [ -x "/Library/Frameworks/Python.framework/Versions/3.11/bin/uvicorn" ]; then
    UVICORN_BIN="/Library/Frameworks/Python.framework/Versions/3.11/bin/uvicorn"
fi

if [ -z "$UVICORN_BIN" ]; then
    echo -e "${RED}[!] Error: Could not locate uvicorn executable.${NC}"
    echo -e "    Ensure backend virtual environment exists at: $BACKEND_DIR/.venv"
    exit 1
fi

# 5. Clean up any existing process on port 8000 if needed
EXISTING_PID=$(lsof -ti :8000 2>/dev/null || true)
if [ -n "$EXISTING_PID" ]; then
    echo -e "${YELLOW}[!] Port 8000 is currently occupied by PID(s): $EXISTING_PID${NC}"
    echo -e "${YELLOW}[*] Gracefully stopping previous instance to bind cleanly to 0.0.0.0...${NC}"
    kill -15 $EXISTING_PID 2>/dev/null || kill -9 $EXISTING_PID 2>/dev/null || true
    sleep 1
fi

# 6. Terminal Banner
echo ""
echo -e "${CYAN}======================================================================${NC}"
echo -e "${GREEN}${BOLD}     MESSMATE LIVE PHONE BACKEND SERVER RUNNER${NC}"
echo -e "${CYAN}======================================================================${NC}"
echo -e "  ${BOLD}Host Binding:${NC}       0.0.0.0 (All Network Interfaces)"
echo -e "  ${BOLD}Port:${NC}               8000"
echo -e "  ${BOLD}Mode:${NC}               Live Production Backend (Zero Mock Interceptors)"
echo ""
echo -e "  ${BOLD}EXACT CONNECTION ENDPOINTS FOR CLIENTS:${NC}"
echo -e "  ------------------------------------------------------------------"
echo -e "  ${GREEN}${BOLD}* Phone on Wi-Fi LAN:${NC}   http://${LAN_IP}:8000"
echo -e "  ${GREEN}${BOLD}* Phone on USB Cable:${NC}   http://127.0.0.1:8000   (adb reverse)"
echo -e "  ${GREEN}${BOLD}* Web / Simulator:${NC}      http://localhost:8000"
echo -e "  ${CYAN}* API Health Check:${NC}     http://${LAN_IP}:8000/v1/health"
echo -e "  ${CYAN}* Interactive Docs:${NC}     http://${LAN_IP}:8000/docs"
echo ""
echo -e "  ${BOLD}MOBILE APP INSTRUCTIONS:${NC}"
echo -e "  1. Launch the Flutter app on your phone."
echo -e "  2. Go to the ${BOLD}Profile & Settings${NC} tab."
echo -e "  3. In the ${BOLD}Backend Server Connection${NC} card:"
echo -e "     - Tap ${GREEN}'Wi-Fi LAN (${LAN_IP})'${NC} if connected over wireless."
echo -e "     - Tap ${GREEN}'USB Reverse (127.0.0.1)'${NC} if connected over USB cable."
echo -e "     - Or tap ${CYAN}'Auto-Detect Best Route'${NC} for automatic discovery."
echo -e "${CYAN}======================================================================${NC}"
echo ""

# 7. Start FastAPI Uvicorn Server
cd "$BACKEND_DIR"
exec "$UVICORN_BIN" app.main:app --host 0.0.0.0 --port 8000
