#!/usr/bin/env bash
# setup.sh - create a Python venv and install dependencies for the fitness AI project on Windows (Git Bash)
# Run this from your project root (D:\python fitness ai trial) in Git Bash:
#   chmod +x setup.sh
#   ./setup.sh
#
# This script:
# - creates a venv (.venv)
# - upgrades pip/setuptools/wheel
# - pins protobuf to a mediapipe-compatible version
# - installs OpenCV
# - tries to install mediapipe (0.10.21). If a direct pip wheel cannot be installed,
#   it downloads the appropriate wheel from PyPI for your Python version and installs it.
# - sets temporary pip config & Windows appdata env vars to avoid platformdirs registry issues
#
# Notes:
# - You must run this in Git Bash on Windows (it works in mingw/msys). If you use PowerShell or cmd,
#   activation paths differ: use .venv\Scripts\Activate.ps1 or .venv\Scripts\activate.bat respectively.
# - If mediapipe imports fail with a missing DLL (MSVCP/VCRUNTIME), install the Microsoft Visual C++ Redistributable (x64)
#   https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist and reboot.

set -euo pipefail

echo
echo "=== setup.sh: Starting environment setup ==="
echo

# 1) Ensure python exists
if ! command -v python >/dev/null 2>&1; then
  echo "ERROR: python not found in PATH. Please install Python or run this script from an environment where python is available."
  exit 1
fi

PYTHON_BIN="$(command -v python)"
echo "Using python: $PYTHON_BIN"
PY_VER=$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version detected: $PY_VER"

# 2) Create venv
VENV_DIR=".venv"
if [ -d "$VENV_DIR" ]; then
  echo "Virtualenv $VENV_DIR already exists — reusing it."
else
  echo "Creating virtual environment in $VENV_DIR ..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# 3) Activate venv (works for Git Bash)
# On Windows Git Bash the activate script is under .venv/Scripts/activate
if [ -f "$VENV_DIR/Scripts/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/Scripts/activate"
elif [ -f "$VENV_DIR/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
else
  echo "ERROR: cannot find venv activate script. Looked for $VENV_DIR/Scripts/activate and $VENV_DIR/bin/activate"
  exit 1
fi

echo "Virtualenv activated. Python is: $(which python)"
python --version

# 4) Upgrade pip, setuptools, wheel
echo
echo "Upgrading pip, setuptools, wheel..."
python -m pip install --upgrade pip setuptools wheel

# 5) Create an empty pip config file to avoid platformdirs/registry lookups (session only)
# Use TEMP or /tmp depending on environment
PIP_CONF="$TEMP/pip.conf"
if [ -z "${TEMP-}" ] && [ -d "/tmp" ]; then
  PIP_CONF="/tmp/pip.conf"
fi
echo "" > "$PIP_CONF"
export PIP_CONFIG_FILE="$PIP_CONF"

# 6) Set Windows APPDATA/PROGRAMDATA/LOCALAPPDATA env vars for this shell session to avoid registry lookup errors
# These only affect this shell session
export PROGRAMDATA="C:\\ProgramData"
# NOTE: USERPROFILE exists in Git Bash environment; ensure it's set
USERPROFILE_WIN="${USERPROFILE:-$HOME}"
export APPDATA="${USERPROFILE_WIN}\\AppData\\Roaming"
export LOCALAPPDATA="${USERPROFILE_WIN}\\AppData\\Local"
echo "Temporary pip config: $PIP_CONFIG_FILE"
echo "Set APPDATA/LOCALAPPDATA/PROGRAMDATA for this session."

# 7) Install protobuf (compat) and opencv
echo
echo "Installing protobuf==3.20.3 and opencv-python..."
# Try normal isolated pip install (avoid reading config files):
python -m pip --isolated install --no-cache-dir protobuf==3.20.3 opencv-python || {
  echo "WARNING: pip install protobuf/opencv failed. Retrying without --isolated..."
  python -m pip install --no-cache-dir protobuf==3.20.3 opencv-python
}

# 8) Try installing mediapipe==0.10.21 via pip wheels
MEDIAPIPE_VERSION="0.10.21"
echo
echo "Attempting to install mediapipe==$MEDIAPIPE_VERSION via pip (binary wheels preferred)..."
if python -m pip --isolated install --no-cache-dir --only-binary=:all: "mediapipe==${MEDIAPIPE_VERSION}"; then
  echo "mediapipe installed successfully via pip wheels."
else
  echo
  echo "mediapipe wheel install via pip failed. Attempting to find a matching wheel on PyPI and install it manually."
  echo "Querying PyPI for a compatible mediapipe wheel for this Python..."

  # 9) Query PyPI JSON to find a matching wheel for win_amd64 and cp{major}{minor}
  PY_TAG=$($PYTHON_BIN - <<PY
import sys
v = sys.version_info
print(f"cp{v.major}{v.minor}-cp{v.major}{v.minor}")
PY
)

  echo "Looking for wheel matching tag: $PY_TAG and platform win_amd64"

  WHEEL_URL=$($PYTHON_BIN - <<PY
import json,urllib.request,sys
pkg="mediapipe"
ver="$MEDIAPIPE_VERSION"
url=f"https://pypi.org/pypi/{pkg}/{ver}/json"
data=json.load(urllib.request.urlopen(url))
files=data.get("releases",{}).get(ver,[])
# prefer manylinux windows wheel for cp tag & win_amd64
py_tag = sys.argv[1]
for f in files:
    fname=f.get("filename","")
    if py_tag in fname and ("win_amd64" in fname or "win32" in fname):
        print(f.get("url"))
        sys.exit(0)
# fallback: any wheel with win_amd64
for f in files:
    fname=f.get("filename","")
    if "win_amd64" in fname:
        print(f.get("url"))
        sys.exit(0)
# fallback: print nothing
print("")
PY
  "$PY_TAG"
)

  if [ -z "$WHEEL_URL" ]; then
    echo "Could not find a suitable mediapipe wheel URL on PyPI for your Python. You can:
  - Manually download the wheel for your Python (cp${PY_VER//./}), win_amd64 from:
    https://pypi.org/project/mediapipe/${MEDIAPIPE_VERSION}/#files
  - Or try installing mediapipe with a different Python 3.10 environment."
    echo "Exiting with failure for mediapipe install."
    exit 0
  fi

  echo "Found wheel URL: $WHEEL_URL"
  WHEEL_FILE="./mediapipe-${MEDIAPIPE_VERSION}.whl"
  echo "Downloading wheel to $WHEEL_FILE ..."
  # Use curl or wget
  if command -v curl >/dev/null 2>&1; then
    curl -L -o "$WHEEL_FILE" "$WHEEL_URL"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$WHEEL_FILE" "$WHEEL_URL"
  else
    echo "ERROR: neither curl nor wget available to download the wheel. Please download it manually and re-run this script."
    echo "Wheel URL: $WHEEL_URL"
    exit 1
  fi

  echo "Installing downloaded wheel..."
  python -m pip --isolated install --no-cache-dir "$WHEEL_FILE" || {
    echo "ERROR: installing downloaded wheel failed. You may be missing the Microsoft Visual C++ Redistributable (x64)."
    echo "Please install it from: https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist and reboot, then re-run."
    exit 1
  }
fi

# 10) Test imports
echo
echo "Testing imports: mediapipe and cv2..."
python - <<PY
import sys
try:
    import mediapipe as mp
    import cv2
    print("OK: mediapipe", getattr(mp,"__version__","unknown"), "opencv", cv2.__version__)
except Exception as e:
    import traceback
    print("IMPORT ERROR: ", file=sys.stderr)
    traceback.print_exc()
    sys.exit(2)
PY

echo
echo "=== setup.sh: finished. If import test printed OK, your venv and packages are ready. ==="
echo
echo "Next (recommended) commands to run from project root (while venv is activated):"
echo "  python scripts/extract_landmarks_from_videos.py --dir data/raw --fps_reduce 2"
echo "  python scripts/build_features.py"
echo "  python scripts/auto_label_from_videos.py"
echo "  python -c \"import pandas as pd; df=pd.read_csv('data/features/features.csv'); print(df.head().to_string(index=False)); print('\\nLabel counts:\\n', df['label'].value_counts())\""
echo
echo "If mediapipe import failed due to missing DLLs (msvcp/vcruntime), install the Microsoft Visual C++ Redistributable (x64) and reboot:"
echo "  https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist"
echo
echo "If you need help after running this script, copy & paste the full output here and I'll guide the next step."