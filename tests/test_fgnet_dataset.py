"""
FGNET 数据集加载器单元测试
"""
import os
import pytest
import torch
from torch.utils.data import DataLoader

from data.prepare_fgnet import FGNETDataset, prepare_fgnet_dataloaders

# 标记为集成测试 (需要真实数据集)
pytestmark = pytest.mark.integration

# FGNET 数据路径 - 不存在则跳过所有测试
FGNET_ROOT = r'E:\fuchuang\FGNET\FGNET'


@pytest.fixture
def config():
    """测试配置"""
    # 检查 FGNET 数据是否存在,不存在则跳过测试
    if not os.path.exists(FGNET_ROOT):
        pytest.skip(f"FGNET 数据集不存在: {FGNET_ROOT}")

    return {
        'data': {
            'fgnet_root': FGNET_ROOT,
            'age_annotation_path': 'age_annotations/kara2015_ageannotations/age_groundtruth.csv',
            'min_samples_per_identity': 3,
            'num_workers': 0,
            'image_size': 224,
        },
        'training': {
            'batch_size': 8,
        },
        'augmentation': {
            'random_horizontal_flip': 0.5,
            'random_affine_degrees': 15,
            'color_jitter_brightness': 0.3,
            'color_jitter_contrast': 0.3,
            'color_jitter_saturation': 0.2,
            'color_jitter_hue': 0.1,
            'gaussian_blur_kernel': 3,
            'gaussian_blur_sigma': [0.1, 2.0],
            'random_erasing_p': 0.5,
            'random_erasing_scale': [0.02, 0.15],
            'random_erasing_ratio': [0.3, 3.3],
        }
    }


def test_fgnet_dataset_creation(config):
    """测试数据集能否正常加载"""
    dataset = FGNETDataset(config, split='train', return_time=False)
    assert len(dataset) > 0, "训练集不应为空"
    assert 80 <= dataset.num_identities <= 85, f"个体数应在 80-85 范围内,实际 {dataset.num_identities}"


def test_fgnet_dataset_getitem(config):
    """测试 __getitem__ 返回值"""
    from data.dataset import get_train_transform
    
    dataset = FGNETDataset(config, split='train', transform=get_train_transform(config), return_time=False)
    image, label = dataset[0]
    assert isinstance(image, torch.Tensor), "图片应为 Tensor"
    assert image.shape[0] == 3, "图片应为 3 通道"
    assert isinstance(label, int), "标签应为整数"


def test_fgnet_dataset_with_time(config):
    """测试返回时间信息"""
    dataset = FGNETDataset(config, split='train', return_time=True)
    image, label, age = dataset[0]
    assert isinstance(age, float), "年龄应为浮点数"
    assert 0 <= age <= 69, f"年龄应在 0-69 范围内,实际 {age}"


def test_train_test_split(config):
    """测试训练/测试集划分"""
    train_dataset = FGNETDataset(config, split='train', return_time=False)
    test_dataset = FGNETDataset(config, split='test', return_time=False)

    total_samples = len(train_dataset) + len(test_dataset)
    # 由于 041A13.JPG 无年龄标注,实际有效样本约 1001
    assert 990 <= total_samples <= 1002, f"总样本数应在 990-1002 范围内,实际 {total_samples}"

    # 训练集约占 70%
    train_ratio = len(train_dataset) / total_samples
    assert 0.65 <= train_ratio <= 0.75, f"训练集比例应在 65-75% 之间,实际 {train_ratio:.2%}"


def test_prepare_dataloaders(config):
    """测试 DataLoader 创建"""
    train_loader, test_loader, num_identities, train_dataset = prepare_fgnet_dataloaders(
        config, return_time=False
    )

    assert 80 <= num_identities <= 85, f"个体数应在 80-85 范围内,实际 {num_identities}"
    assert len(train_loader) > 0
    assert len(test_loader) > 0

    # 测试 batch 数据
    batch = next(iter(train_loader))
    assert len(batch) == 2  # (image, label)
    assert batch[0].shape[0] == config['training']['batch_size']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
