import torch
import torch.nn as nn

from open_fairprune.data_util import INPUT_SIZE

MODEL_NAME = "SimpleNNv3"

model = torch.nn.Sequential(
    nn.Linear(INPUT_SIZE, 500, bias=False),
    nn.ReLU(),
    nn.Dropout(0.50),
    nn.Linear(500, 500, bias=False),
    nn.ReLU(),
    nn.Dropout(0.50),
    nn.Linear(500, 500, bias=False),
    nn.ReLU(),
    nn.Dropout(0.50),
    nn.Linear(500, 2, bias=False),
)
