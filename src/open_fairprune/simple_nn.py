import torch
import torch.nn as nn

from open_fairprune.data_util import INPUT_SIZE

MODEL_NAME = "SimpleNNv3"

model = torch.nn.Sequential(
    nn.Linear(INPUT_SIZE, 1000, bias=False),
    nn.ReLU(),
    nn.Dropout(0.20),
    nn.Linear(1000, 1000, bias=False),
    nn.ReLU(),
    nn.Dropout(0.20),
    nn.Linear(1000, 1000, bias=False),
    nn.ReLU(),
    nn.Dropout(0.20),
    nn.Linear(1000, 2, bias=False),
)
