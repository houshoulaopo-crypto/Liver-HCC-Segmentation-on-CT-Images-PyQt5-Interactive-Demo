#!/usr/bin/env python3
"""Launch the PyQt5 demo."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    main_py = ROOT / "app" / "main.py"
    subprocess.run([sys.executable, str(main_py)], cwd=ROOT, check=False)


if __name__ == "__main__":
    main()
