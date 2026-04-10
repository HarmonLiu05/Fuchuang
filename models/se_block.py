"""
Squeeze-and-Excitation (SE) Block
用于增强 ResNet 通道注意力
"""
import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        """
        Args:
            channels: 输入通道数
            reduction: 压缩比例（越大参数量越少）
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def add_se_to_resnet_layer(layer, reduction=16):
    """
    给 ResNet 的 layer 添加 SE-Block
    遍历所有 BasicBlock/Bottleneck，在末尾添加 SE
    """
    for module in layer.modules():
        if hasattr(module, 'bn3') and not hasattr(module, 'se'):
            # Bottleneck block (ResNet50/101/152)
            module.se = SEBlock(module.bn3.num_features, reduction=reduction)
            # 修改 forward传播
            module.forward = _bottleneck_forward_with_se.__get__(module)
        elif hasattr(module, 'bn2') and not hasattr(module, 'se'):
            # BasicBlock (ResNet18/34)
            module.se = SEBlock(module.bn2.num_features, reduction=reduction)
            module.forward = _basicblock_forward_with_se.__get__(module)
    return layer


def _bottleneck_forward_with_se(self, x):
    """带 SE 的 Bottleneck 前向传播"""
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    # SE-Block
    if hasattr(self, 'se'):
        out = self.se(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out


def _basicblock_forward_with_se(self, x):
    """带 SE 的 BasicBlock 前向传播"""
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    # SE-Block
    if hasattr(self, 'se'):
        out = self.se(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out
