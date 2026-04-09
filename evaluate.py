"""
模型评估脚本
加载训练好的模型，在验证集上计算 Accuracy, TAR@FAR, Rank-1
"""
import os
import sys
import argparse
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from train import ChimpFaceModel
from data.prepare_data import prepare_dataloaders
from utils.metrics import compute_all_metrics
from utils.utils import load_config, get_device


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()
    features_list = []
    labels_list = []
    
    print("提取特征...")
    for images, labels in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        feats = model.backbone(images)
        feats = model.bottleneck(feats)
        features_list.append(feats.cpu())
        labels_list.append(labels)
    
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    print(f"\n特征矩阵: {features.shape}")
    print(f"标签数量: {labels.shape[0]}")
    
    print("\n计算评估指标...")
    metrics = compute_all_metrics(features, labels)
    
    print("\n" + "="*50)
    print("评估结果")
    print("="*50)
    print(f"Accuracy (Rank-1): {metrics['accuracy']:.4f}")
    print(f"TAR@FAR=0.1%:      {metrics['tar_at_far_0.1']:.4f}")
    print(f"Threshold:         {metrics['threshold']:.4f}")
    print("="*50)
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_local.yaml')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = get_device()
    
    print(f"加载检查点: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    arcface_weight = checkpoint['model_state_dict']['arcface.weight']
    num_identities = arcface_weight.shape[0]
    
    model = ChimpFaceModel(config, num_identities)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"模型包含 {num_identities} 个个体")
    
    train_loader, val_loader, _, dataset = prepare_dataloaders(config)
    
    metrics = evaluate_model(model, val_loader, device)


if __name__ == '__main__':
    main()
