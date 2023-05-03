import random
from pathlib import Path

import mlflow
import numpy as np
import torch

import open_fairprune
from open_fairprune.data_util import DATA_PATH

random.seed(0)
np.random.seed(0)
torch.manual_seed(42)

mlflow.set_tracking_uri(uri=f"file://{DATA_PATH}\mlruns")

pre_commit = Path(open_fairprune.__path__[0]).parents[1] / ".git/hooks/pre-commit"
if not pre_commit.exists():
    raise IOError("Please run `pip install pre-commit && pre-commit install` to enable pre-commit hooks")
