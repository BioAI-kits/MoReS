"""Run MoReS commands directly from a source checkout.

Examples:
    python main.py --help
    python main.py train --data data/blca --config configs/method.json --output runs/train

Use a Python environment with the dependencies declared in pyproject.toml.
"""
from pathlib import Path
import sys


def main(argv=None):
    # Resolve relative to this file so invocation also works from another cwd.
    source = str(Path(__file__).resolve().parent / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from mores.cli import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    main()
