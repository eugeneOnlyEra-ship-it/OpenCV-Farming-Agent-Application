#!/usr/bin/env bash
# Launcher for the interactive Farm Agent app (Fedora / any Linux).
# Creates a local virtualenv (.venv) on first run and keeps its packages in sync
# with requirements.txt (OpenCV 5) on every launch.
set -e
cd "$(dirname "$0")"

if ! python3 -c "import tkinter" 2>/dev/null; then
    echo "Tkinter is missing. On Fedora install it with:"
    echo "    sudo dnf install python3-tkinter"
    exit 1
fi

if [ ! -d .venv ]; then
    echo "Creating virtual environment (.venv) ..."
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
fi
.venv/bin/pip install -q -r requirements.txt
echo "OpenCV $(.venv/bin/python -c 'import cv2; print(cv2.__version__)')"

exec .venv/bin/python app_ui.py "$@"
