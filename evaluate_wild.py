"""
野外数据集评估：评估模型在 C-Tai（野生环境）数据集上的泛化能力
"""
import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from data.dataset import ChimpanzeeDataset, get_val_transform
from utils.utils import load_config, set_seed
from models.backbone import ResNet50Backbone
from models.bottleneck import Bottleneck
from models.arcface import ArcFace
from utils.metrics import compute_accuracy, compute_tar_at_far, compute_all_metrics


def evaluate_wild(config_path, checkpoint):
    """在野外数据集上评估模型"""
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型权重（手动构建模型）
    backbone = ResNet50Backbone(pretrained=False)
    bottleneck = Bottleneck()
    # ArcFace 需要特殊处理，因为它是分类头
    checkpoint_data = torch.load(checkpoint, map_location=device)
    
    # 加载 backbone 和 bottleneck
    backbone.load_state_dict({k.replace('backbone.', ''): v for k, v in checkpoint_data.items() if 'backbone' in k})
    bottleneck.load_state_dict({k.replace('bottleneck.', ''): v for k, v in checkpoint_data.items() if 'bottleneck' in k})
    
    backbone = backbone.to(device)
    bottleneck = bottleneck.to(device)
    backbone.eval()
    bottleneck.eval()
    
    # 创建野外数据集
    wild_dataset = ChimpanzeeDataset(
        root_dir=config['data']['root_dir'],
        annotation_file=config['data'].get('wild_annotation_file', 'data_CTai/annotations_ctai.txt'),
        image_dir=config['data'].get('wild_image_dir', 'data_CTai'),
        min_samples_per_identity=config['data'].get('min_samples_per_identity', 10)
    )
    
    print(f"野外数据集: {len(wild_dataset.identities)} 个个体, {len(wild_dataset)} 张图片")
    print(f"个体列表: {wild_dataset.identities}")
    
    wild_loader = DataLoader(
        wild_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=torch.cuda.is_available()
    )
    
    # 特征提取
    features, labels = [], []
    pbar = tqdm(wild_loader, desc="提取特征")
    with torch.no_grad():
        for images, lbls in pbar:
            images = images.to(device, non_blocking=True)
            feats = backbone(images)
            feats = bottleneck(feats)
            features.append(feats.cpu())
            labels.append(lbls)
    
    features = torch.cat(features)
    labels = torch.cat(labels)
    
    # 计算指标
    accuracy = compute_accuracy(features, labels)
    tar_far, threshold = compute_tar_at_far(features, labels)
    
    print(f"\n{'='*50}")
    print(f"野外数据集（C-Tai）评估结果")
    print(f"{'='*50}")
    print(f"Accuracy (Rank-1): {accuracy:.4f}")
    print(f"TAR@FAR=0.1%:      {tar_far:.4f}")
    print(f"Threshold:         {threshold:.4f}")
    print(f"{'='*50}")
    
    return accuracy, tar_far


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='野外数据集评估')
    parser.add_argument('--config', default='configs/config_vast.yaml', help='配置文件路径')
    parser.add_argument('--checkpoint', required=True, help='模型权重路径')
    args = parser.parse_args()
    
    evaluate_wild(args.config, args.checkpoint)
