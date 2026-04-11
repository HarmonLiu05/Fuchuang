import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFace(nn.Module):
    def __init__(self, in_features=512, num_classes=8, s=30.0, m=0.35):
        super().__init__()
        self.s = s
        self.m = m
        self.in_features = in_features
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x, label):
        # 权重矩阵归一化
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        
        # 特征归一化（双重保险）
        x_norm = F.normalize(x, p=2, dim=1)
        cosine = torch.mm(x_norm, weight_norm.t())
        
        # ArcFace: 角度空间加 margin
        theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
        phi = torch.cos(theta + self.m)
        
        # one-hot 编码
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        
        # 选择正确的 logit
        output = one_hot * phi + (1 - one_hot) * cosine
        output *= self.s
        
        return output
    
    def update_margin(self, new_m):
        self.m = new_m
