import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_features=2048, bottleneck_dim=512, dropout=0.4):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(in_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = self.bottleneck(x)
        return F.normalize(x, p=2, dim=1)
