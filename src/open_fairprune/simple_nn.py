import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from open_fairprune.data_util import INPUT_SIZE, LoanDataset

MODEL_NAME = "SimpleNNv2"


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.seq1 = nn.Sequential(  # new seq to allow concat embedding
            nn.Linear(INPUT_SIZE, 1000),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(1000, 2),
            # nn.Sigmoid(),
        )

    def forward(self, segment):
        x = self.seq1(segment)
        return x.squeeze()


model = torch.nn.Sequential(
    nn.Linear(INPUT_SIZE, 1000),
    nn.ReLU(),
    nn.Dropout(0.25),
    nn.Linear(1000, 2),
)
