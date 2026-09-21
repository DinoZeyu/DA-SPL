"""Repository-local entrypoint, no editable package installation required."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_method.runner import main

if __name__ == '__main__':
    main()
