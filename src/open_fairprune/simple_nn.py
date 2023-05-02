import torch
import torch.nn as nn

from open_fairprune.data_util import INPUT_SIZE

MODEL_NAME = "SimpleNNv2"

model = torch.nn.Sequential(
    nn.Linear(INPUT_SIZE, 1000),
    nn.ReLU(),
    nn.Dropout(0.25),
    nn.Linear(1000, 2),
)
