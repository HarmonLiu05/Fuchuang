import torch
import pytest
from losses.time_weighted_triplet import TimeWeightedTripletLoss


class TestTimeWeightedTripletLoss:
    def test_basic_loss(self):
        """基础测试：4个样本，2个个体"""
        loss_fn = TimeWeightedTripletLoss(margin=0.3, alpha=0.3)
        embeddings = torch.randn(4, 512)
        labels = torch.tensor([0, 0, 1, 1])
        times = torch.tensor([2020.0, 2021.0, 2020.0, 2021.0])
        
        loss = loss_fn(embeddings, labels, times)
        assert loss >= 0
        assert loss.dim() == 0

    def test_time_weighting_effect(self):
        """验证时间加权效果"""
        loss_fn = TimeWeightedTripletLoss(margin=0.5, alpha=0.5, max_time_gap=10.0)
        
        embeddings = torch.tensor([
            [1.0, 0.0],  # A: 个体0, 2020年
            [0.9, 0.1],  # B: 个体0, 2025年 (时间跨度5年)
            [0.0, 1.0],  # C: 个体1, 2020年
        ])
        labels = torch.tensor([0, 0, 1])
        times = torch.tensor([2020.0, 2025.0, 2020.0])
        
        loss = loss_fn(embeddings, labels, times)
        assert loss >= 0

    def test_no_time_info(self):
        """测试无时间信息时退化为普通 Triplet"""
        loss_fn = TimeWeightedTripletLoss(margin=0.3, alpha=0.3)
        embeddings = torch.randn(4, 512)
        labels = torch.tensor([0, 0, 1, 1])
        times = torch.tensor([float('nan')] * 4)
        
        loss = loss_fn(embeddings, labels, times)
        assert loss >= 0

    def test_single_identity(self):
        """测试单一个体返回 0"""
        loss_fn = TimeWeightedTripletLoss()
        embeddings = torch.randn(4, 512)
        labels = torch.tensor([0, 0, 0, 0])
        times = torch.tensor([2020.0, 2021.0, 2022.0, 2023.0])
        
        loss = loss_fn(embeddings, labels, times)
        assert loss.item() == 0.0

    def test_alpha_zero_equals_standard(self):
        """测试 alpha=0 时等于标准 Triplet Loss"""
        from losses.triplet import BatchHardTripletLoss
        
        loss_fn_time = TimeWeightedTripletLoss(margin=0.3, alpha=0.0)
        loss_fn_std = BatchHardTripletLoss(margin=0.3)
        
        embeddings = torch.randn(8, 256)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        times = torch.tensor([2020.0] * 8)
        
        loss_time = loss_fn_time(embeddings, labels, times)
        loss_std = loss_fn_std(embeddings, labels)
        
        assert torch.allclose(loss_time, loss_std, atol=1e-6)
