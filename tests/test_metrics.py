import pytest
import torch
import numpy as np
from utils.metrics import compute_accuracy, compute_tar_at_far, compute_all_metrics


class TestComputeAccuracy:
    def test_compute_accuracy_perfect(self):
        # 完美分类场景：同一identity的样本相似度最高
        # 构造特征：同一identity的样本特征接近，不同identity的样本特征差异大
        features = torch.tensor([
            [1.0, 0.0, 0.0],  # identity 0
            [0.9, 0.1, 0.0],  # identity 0 (与样本0相似)
            [0.0, 0.0, 1.0],  # identity 1
            [0.0, 0.1, 0.9],  # identity 1 (与样本2相似)
        ], dtype=torch.float32)
        
        # L2 归一化
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        labels = torch.tensor([0, 0, 1, 1])
        
        similarity = torch.mm(features, features.t())
        accuracy = compute_accuracy(similarity, labels)
        
        # 样本0的最近邻是样本1（同一identity），样本1的最近邻是样本0
        # 样本2的最近邻是样本3，样本3的最近邻是样本2
        assert accuracy == 1.0, f"Expected 1.0, got {accuracy}"
    
    def test_compute_accuracy_wrong(self):
        # 全错分类场景：每个样本的最近邻都是不同identity
        # 构造相似度矩阵使得 cross-identity 相似度最高
        similarity = torch.tensor([
            [-1.0, 0.1, 0.9, 0.8],  # 样本0最近邻是样本2（不同identity）
            [0.1, -1.0, 0.8, 0.9],  # 样本1最近邻是样本3（不同identity）
            [0.9, 0.8, -1.0, 0.1],  # 样本2最近邻是样本0（不同identity）
            [0.8, 0.9, 0.1, -1.0],  # 样本3最近邻是样本1（不同identity）
        ])
        labels = torch.tensor([0, 0, 1, 1])
        
        accuracy = compute_accuracy(similarity, labels)
        assert accuracy == 0.0, f"Expected 0.0, got {accuracy}"


class TestTarAtFar:
    def test_compute_tar_at_far_synthetic(self):
        # 合成数据：前2个样本是同一identity，后2个是不同identity
        features = torch.tensor([
            [1.0, 0.0],
            [0.9, 0.1],  # 与样本0相似
            [0.0, 1.0],
            [0.1, 0.9],  # 与样本2相似
        ], dtype=torch.float32)
        
        # L2归一化
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        labels = torch.tensor([0, 0, 1, 1])
        
        tar, threshold = compute_tar_at_far(features, labels, target_far=0.001)
        
        assert 0.0 <= tar <= 1.0, f"TAR should be in [0, 1], got {tar}"
        assert -1.0 <= threshold <= 1.0, f"Threshold should be in [-1, 1], got {threshold}"


class TestComputeAllMetrics:
    def test_compute_all_metrics(self):
        features = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.1, 0.9],
        ], dtype=torch.float32)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        labels = torch.tensor([0, 0, 1, 1])
        
        metrics = compute_all_metrics(features, labels)
        
        assert 'accuracy' in metrics
        assert 'tar_at_far_0.1' in metrics
        assert 'threshold' in metrics
        
        assert metrics['accuracy'] == 1.0
        assert 0.0 <= metrics['tar_at_far_0.1'] <= 1.0
        assert -1.0 <= metrics['threshold'] <= 1.0
