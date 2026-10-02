#!/usr/bin/env bash
# Launcher for the interactive Farm Agent app (Fedora / any Linux).
# First run creates a local virtualenv in .venv and installs the Python deps.
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
    .venv/bin/pip install -r requirements.txt
fi

exec .venv/bin/python app_ui.py "$@"
