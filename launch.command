#!/bin/bash
# Launch the Strava Run Analytics dashboard.
#
# Double-click this file in Finder, or run ./launch.command from a terminal.
# The first run builds a virtual environment and installs the requirements;
# later runs skip straight to opening the app at http://localhost:8501.
#
# The venv lives in ~/.venvs/strava rather than the project's .venv because the
# Desktop is synced by iCloud, which offloads venv files and makes Python hang.

set -euo pipefail

cd "$(dirname "$0")"

VENV="$HOME/.venvs/strava"
STAMP="$VENV/.requirements.sha"

if [ ! -x "$VENV/bin/python" ]; then
    echo "Creating virtual environment in $VENV ..."
    mkdir -p "$HOME/.venvs"
    python3 -m venv "$VENV"
fi

# Reinstall only when requirements.txt has changed since the last install.
REQ_SHA="$(shasum requirements.txt | cut -d' ' -f1)"
if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$REQ_SHA" ]; then
    echo "Installing requirements ..."
    "$VENV/bin/pip" install --upgrade pip --quiet
    "$VENV/bin/pip" install -r requirements.txt
    echo "$REQ_SHA" > "$STAMP"
fi

if [ ! -f data/sample/activities.csv ]; then
    echo "Generating sample data ..."
    "$VENV/bin/python" -m src.generate_sample
fi

echo "Starting dashboard at http://localhost:8501  (Ctrl+C to stop)"
exec "$VENV/bin/streamlit" run app.py
