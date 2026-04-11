import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    """
    多层 MLP Bottleneck，增强非线性表达能力
    """
    def __init__(self, in_features=2048, bottleneck_dim=512, dropout=0.4, use_mlp=True):
        super().__init__()
        self.use_mlp = use_mlp
        
        if use_mlp:
            # 多层 MLP: 2048 -> 1024 -> 512
            mid_dim = (in_features + bottleneck_dim) // 2  # 1024
            self.bottleneck = nn.Sequential(
                nn.Linear(in_features, mid_dim),
                nn.BatchNorm1d(mid_dim),
                nn.ReLU(inplace=True),
                nn.Linear(mid_dim, bottleneck_dim),
                nn.BatchNorm1d(bottleneck_dim),
                nn.Dropout(dropout)
            )
        else:
            # 原始单层（向后兼容）
            self.bottleneck = nn.Sequential(
                nn.Linear(in_features, bottleneck_dim),
                nn.BatchNorm1d(bottleneck_dim),
                nn.Dropout(dropout)
            )

    def forward(self, x):
        x = self.bottleneck(x)
        return F.normalize(x, p=2, dim=1)
