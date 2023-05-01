from pathlib import Path

import open_fairprune

pre_commit = Path(open_fairprune.__path__[0]).parents[1] / ".git/hooks/pre-commit"
if not pre_commit.exists():
    raise IOError("Please run `pip install pre-commit && pre-commit install` to enable pre-commit hooks")
