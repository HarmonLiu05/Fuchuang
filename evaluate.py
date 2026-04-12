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
        labels_list.append(labels.cpu())
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
    """根据配置创建模型（支持TurtleFaceModel）"""
    from train_turtle import TurtleFaceModel
    return TurtleFaceModel(config, num_identities)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_turtle.yaml')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()

    print(f"加载检查点: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # 优先使用 checkpoint 中的配置（模型结构），但数据路径使用当前配置
    if 'config' in checkpoint:
        ckpt_config = checkpoint['config']
        # 合并配置，保留 checkpoint 中的关键字段
        config['model'].update({
            'backbone': ckpt_config['model'].get('backbone', config['model'].get('backbone', 'resnet50')),
            'bottleneck_dim': ckpt_config['model'].get('bottleneck_dim', config['model'].get('bottleneck_dim', 512)),
            'use_mlp_bottleneck': ckpt_config['model'].get('use_mlp_bottleneck', False),
            'use_se_block': ckpt_config['model'].get('use_se_block', False),
            'freeze_until_layer': ckpt_config['model'].get('freeze_until_layer', 3),
        })
        # 保留当前配置的数据路径，避免checkpoint中的路径覆盖
        current_data_config = config['data'].copy()
        config['data'] = ckpt_config.get('data', config.get('data', {}))
        # 用当前配置覆盖数据路径关键字段
        for key in ['root_dir', 'splits_dir', 'dataset_name', 'train_json', 'test_json']:
            if key in current_data_config:
                config['data'][key] = current_data_config[key]
        print(f"使用 checkpoint 中的配置: backbone={config['model']['backbone']}, bottleneck_dim={config['model']['bottleneck_dim']}")

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
