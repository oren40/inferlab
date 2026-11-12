"""Allow `python -m inferlab` without installing anything.

A fresh clone has no `inferlab` executable on PATH -- that only appears after
`pip install -e .`. This module closes that gap, so the README's quickstart
works for someone who cloned the repo and does not want to install first.
"""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
