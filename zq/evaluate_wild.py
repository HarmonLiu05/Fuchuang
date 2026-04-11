"""
野外数据集评估：评估模型在 C-Tai（野生环境）数据集上的泛化能力
使用 C-Zoo 全量数据作为底库 (Gallery)，C-Tai 作为查询集 (Query)。
"""
import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from data.dataset import ChimpanzeeDataset
from utils.utils import load_config
from utils.metrics import compute_rank1_accuracy, compute_tar_at_far


@torch.no_grad()
def extract_features(model, dataloader, device):
    features, labels = [], []
    for images, lbls in tqdm(dataloader, desc="提取特征", leave=False):
        images = images.to(device, non_blocking=True)
        feats = model.backbone(images)
        feats = model.bottleneck(feats)
        features.append(feats.cpu())
        labels.append(lbls)
    return torch.cat(features), torch.cat(labels)


def evaluate_wild(config_path, checkpoint):
    """在野外数据集上评估模型"""
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载模型
    from train import ChimpFaceModel
    model = ChimpFaceModel(config, num_identities=24)
    model = model.to(device)
    
    checkpoint_data = torch.load(checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint_data:
        model.load_state_dict(checkpoint_data['model_state_dict'])
    else:
        model.load_state_dict(checkpoint_data)
    model.eval()

    # 2. 准备底库 Gallery (C-Zoo 全量数据)
    print("=> 正在加载底库数据 (C-Zoo)...")
    gallery_dataset = ChimpanzeeDataset(
        root_dir=config['data']['root_dir'],
        annotation_file='data_CZoo/annotations_czoo.txt',
        image_dir='data_CZoo',
        min_samples_per_identity=1
    )
    gallery_loader = DataLoader(gallery_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    gallery_features, gallery_labels = extract_features(model, gallery_loader, device)
    identity_map = {name: i for i, name in enumerate(gallery_dataset.identities)}
    print(f"底库加载完成：{len(gallery_dataset)} 样本, {len(gallery_dataset.identities)} 个个体")

    # 3. 准备查询集 Query (C-Tai 数据)
    print("=> 正在加载查询数据 (C-Tai)...")
    wild_dataset = ChimpanzeeDataset(
        root_dir=config['data']['root_dir'],
        annotation_file='data_CTai/annotations_ctai.txt',
        image_dir='data_CTai',
        min_samples_per_identity=1
    )
    
    # 转换标签：将 C-Tai 的标签映射到 C-Zoo 的标签 ID
    valid_indices = []
    query_labels = []
    for i in range(len(wild_dataset)):
        name = wild_dataset.samples[i]['identity']
        if name in identity_map:
            valid_indices.append(i)
            query_labels.append(identity_map[name])

    from torch.utils.data import Subset
    query_subset = Subset(wild_dataset, valid_indices)
    query_loader = DataLoader(query_subset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    if len(query_subset) == 0:
        print("Error: No matching identities between C-Tai and C-Zoo.")
        return

    query_features, _ = extract_features(model, query_loader, device)
    query_labels = torch.tensor(query_labels)
    print(f"查询集加载完成：{len(query_subset)} 样本（已对齐到底库 ID）")

    # 4. 计算指标
    accuracy = compute_rank1_accuracy(query_features, query_labels, gallery_features, gallery_labels)
    
    # 计算 TAR@FAR (Query 内部一致性)
    tar_at_far, threshold = compute_tar_at_far(query_features, query_labels)

    print(f"\n{'='*50}")
    print(f"野外数据集（C-Tai）识别评估结果")
    print(f"{'='*50}")
    print(f"Rank-1 Accuracy (vs 训练集): {accuracy:.4f}")
    print(f"TAR@FAR=0.1% (Query 内部):  {tar_at_far:.4f}")
    print(f"Threshold:                  {threshold:.4f}")
    print(f"{'='*50}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config_vast.yaml')
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    evaluate_wild(args.config, args.checkpoint)
