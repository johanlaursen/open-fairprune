from pathlib import Path

import mlflow

import open_fairprune
from open_fairprune.data_util import DATA_PATH

mlflow.set_tracking_uri(uri=f"file://{DATA_PATH}\mlruns")

pre_commit = Path(open_fairprune.__path__[0]).parents[1] / ".git/hooks/pre-commit"
if not pre_commit.exists():
    raise IOError("Please run `pip install pre-commit && pre-commit install` to enable pre-commit hooks")
