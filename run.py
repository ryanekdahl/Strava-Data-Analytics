"""Open this file and press Run to start the dashboard.

Hands off to launch.command, which sets up the virtual environment in
~/.venvs/strava (outside iCloud), installs requirements if needed and runs
`streamlit run app.py`. Uses only the standard library, so any Python
interpreter VS Code has selected will work. Stop the app with Ctrl+C.
"""

import subprocess
import sys
from pathlib import Path

launcher = Path(__file__).resolve().parent / "launch.command"

try:
    sys.exit(subprocess.call(["bash", str(launcher)]))
except KeyboardInterrupt:
    pass
