#!/usr/bin/env python3
"""Launch the unified W-state experiment control UI."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_root_on_path()

from system_control.experiment_control.experiment_ui import main


if __name__ == "__main__":
    main()
