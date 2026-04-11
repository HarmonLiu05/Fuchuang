"""
模型评估脚本
加载训练好的模型，在验证集上计算 Accuracy, TAR@FAR, Rank-1
支持普通数据集和龟类COCO数据集
"""
import os
import sys
import argparse
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from train import ChimpFaceModel
from utils.metrics import compute_all_metrics, compute_pred_accuracy
from utils.utils import load_config, get_device


def prepare_dataloaders(config):
    """根据配置自动选择数据加载器"""
    # 检测是否为龟类数据集配置
    if 'splits_dir' in config.get('data', {}):
        # 龟类COCO数据集
        from data.prepare_turtle_data import prepare_turtle_dataloaders
        return prepare_turtle_dataloaders(config)
    else:
        # 普通数据集
        from data.prepare_data import prepare_dataloaders as prepare_normal_dataloaders
        return prepare_normal_dataloaders(config)


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()
    features_list = []
    labels_list = []
    ids_list = []

    print("提取特征...")
    for images, labels in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device)
        feats = model.backbone(images)
        feats = model.bottleneck(feats)
        logits = model.arcface(feats, labels)
        features_list.append(feats.cpu())
        labels_list.append(labels)
        pred = torch.argmax(logits, dim=1)
        ids_list.append(pred.cpu())

    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    ids = torch.cat(ids_list, dim=0)

    print(f"\n特征矩阵: {features.shape}")
    print(f"标签数量: {labels.shape[0]}")

    print("\n计算评估指标...")
    metrics = compute_all_metrics(features, labels)
    metrics['accuracy0'] = compute_pred_accuracy(ids, labels)

    print("\n" + "="*50)
    print("评估结果")
    print("="*50)
    print(f"Accuracy (Rank-1):     {metrics['accuracy']:.4f}")
    print(f"Accuracy0 (Direct):    {metrics['accuracy0']:.4f}")
    print(f"TAR@FAR=0.1%:          {metrics['tar_at_far_0.1']:.4f}")
    print(f"Threshold:             {metrics['threshold']:.4f}")
    print("="*50)

    return metrics


def create_model(config, num_identities):
    """根据配置创建模型（支持ChimpFaceModel和TurtleFaceModel）"""
    # 尝试导入龟类模型
    try:
        from train_turtle import TurtleFaceModel
        return TurtleFaceModel(config, num_identities)
    except ImportError:
        # 回退到黑猩猩模型
        return ChimpFaceModel(config, num_identities)


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

    model = create_model(config, num_identities)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"模型包含 {num_identities} 个个体")

    train_loader, val_loader, _, dataset = prepare_dataloaders(config)

    metrics = evaluate_model(model, val_loader, device)


if __name__ == '__main__':
    main()
