"""
龟类数据集数据准备函数
创建 DataLoader 实例
"""
import torch
from torch.utils.data import DataLoader

from data.turtle_dataset import TurtleDataset
from data.dataset import get_train_transform, get_val_transform


def prepare_turtle_dataloaders(config, return_time=False):
    """
    准备龟类数据集的DataLoader

    Args:
        config: 配置字典
        return_time: 是否返回时间信息（用于时间加权triplet loss）

    Returns:
        train_loader: 训练集DataLoader
        test_loader: 测试集DataLoader
        num_identities: 身份数量
        train_dataset: 训练集Dataset实例
    """
    # 创建训练集
    train_dataset = TurtleDataset(
        config,
        split='train',
        transform=get_train_transform(config),
        return_time=return_time
    )

    # 创建测试集（共享train集的identity_map）
    test_dataset = TurtleDataset(
        config,
        split='test',
        transform=get_val_transform(config),
        identity_map=train_dataset.identity_map,
        return_time=return_time
    )
    
    num_identities = train_dataset.num_identities
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 0),
        pin_memory=torch.cuda.is_available(),
        drop_last=True  # 保证每个batch大小一致，最后不足batch_size的样本被丢弃
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 0),
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, test_loader, num_identities, train_dataset
