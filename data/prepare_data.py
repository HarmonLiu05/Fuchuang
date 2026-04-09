"""
数据预处理脚本：加载数据集，划分 train/val，打印统计信息
"""
import os
import sys
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.dataset import ChimpanzeeDataset
from data.dataset import get_train_transform, get_val_transform
from utils.utils import load_config


def prepare_dataloaders(config, seed=42):
    """
    准备 train/val DataLoader
    
    Returns:
        train_loader, val_loader, num_identities, dataset
    """
    full_dataset = ChimpanzeeDataset(
        root_dir=config['data']['root_dir'],
        annotation_file=config['data']['annotation_file'],
        image_dir=config['data']['image_dir'],
        min_samples_per_identity=config['data'].get('min_samples_per_identity', 20)
    )
    
    print(f"找到 {len(full_dataset.identities)} 个个体")
    print(f"总样本数: {len(full_dataset)}")
    print(f"个体列表: {full_dataset.identities}")
    
    # 按个体划分 train/val
    train_indices, val_indices = [], []
    
    for identity_idx in range(len(full_dataset.identities)):
        identity_samples = [
            i for i, sample in enumerate(full_dataset.samples)
            if sample['label'] == identity_idx
        ]
        
        train_idx, val_idx = train_test_split(
            identity_samples,
            test_size=1 - config['data']['train_split'],
            random_state=seed
        )
        
        train_indices.extend(train_idx)
        val_indices.extend(val_idx)
    
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    
    print(f"训练集: {len(train_subset)} 样本")
    print(f"验证集: {len(val_subset)} 样本")
    
    train_loader = DataLoader(
        train_subset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 2),
        pin_memory=torch.cuda.is_available()  # 仅 GPU 时启用
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 2),
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, len(full_dataset.identities), full_dataset


if __name__ == '__main__':
    config = load_config('configs/config_local.yaml')
    train_loader, val_loader, num_id, dataset = prepare_dataloaders(config)
    print(f"\n准备完成！个体数: {num_id}")
