"""Small projection modules shared by the MaDiS architecture."""

import torch.nn as nn


class MLPConnector(nn.Module):
    """Two-layer SiLU projection used for MoP and auxiliary objectives."""

    def __init__(self, input_dim=1024, ff_dim=3072, output_dim=512, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, output_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, inputs):
        return self.dropout2(
            self.linear2(self.dropout1(self.act(self.linear1(inputs)))))
