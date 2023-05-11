import torch
import torch.nn as nn

from open_fairprune.data_util import INPUT_SIZE

MODEL_NAME = "SimpleNNv3"

model = torch.nn.Sequential(
    nn.Linear(INPUT_SIZE, 1000),
    nn.ReLU(),
    nn.Dropout(0.20),
    nn.Linear(1000, 100),
    nn.ReLU(),
    nn.Dropout(0.20),
    nn.Linear(100, 10),
    nn.ReLU(),
    nn.Dropout(0.20),
    nn.Linear(10, 2),
)
