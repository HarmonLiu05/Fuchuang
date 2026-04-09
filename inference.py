"""
分类式推理脚本：直接通过模型内部的分类头输出身份预测
无需外部检索，一次前向传播即可得到结果
"""
import os
import sys
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from data.dataset import get_val_transform
from utils.utils import load_config, set_seed, get_device


def load_model(config_path, checkpoint, device):
    """加载训练好的模型（含分类头）"""
    from train import ChimpFaceModel
    config = load_config(config_path)
    
    # 创建模型
    model = ChimpFaceModel(config, num_identities=24)
    
    # 加载权重
    ckpt = torch.load(checkpoint, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    
    model = model.to(device)
    model.eval()
    return model, config


def predict_single(model, image_path, config, device, identity_names):
    """
    单张图像推理
    返回: (预测名字, 置信度, Top5 预测)
    """
    transform = get_val_transform(config)
    
    # 预处理
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        features = model.backbone(img_tensor)
        features = model.bottleneck(features)
        
        # 分类头推理：直接用 ArcFace 的 weight 矩阵
        logits = F.linear(features, model.arcface.weight)  # (1, 24)
        probs = F.softmax(logits, dim=1)
    
    # 解析结果
    pred_idx = torch.argmax(probs, dim=1).item()
    confidence = probs[0, pred_idx].item()
    
    # Top 5
    top5_probs, top5_indices = torch.topk(probs, 5, dim=1)
    top5 = [(identity_names[i], p.item()) for i, p in zip(top5_indices[0], top5_probs[0])]
    
    return identity_names[pred_idx], confidence, top5


def predict_batch(model, dataloader, device, identity_names):
    """
    批量推理（用于计算 Accuracy）
    返回: 预测列表 [(真实标签, 预测标签, 置信度), ...]
    """
    results = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="批量推理"):
            images = images.to(device)
            
            features = model.backbone(images)
            features = model.bottleneck(features)
            
            # 分类头推理
            logits = F.linear(features, model.arcface.weight)
            probs = F.softmax(logits, dim=1)
            
            pred_indices = torch.argmax(probs, dim=1)
            confidences = torch.max(probs, dim=1).values
            
            for i in range(len(labels)):
                results.append({
                    'true_label': labels[i].item(),
                    'true_name': identity_names[labels[i].item()],
                    'pred_label': pred_indices[i].item(),
                    'pred_name': identity_names[pred_indices[i].item()],
                    'confidence': confidences[i].item()
                })
    
    return results


def compute_accuracy(results):
    """计算分类准确率"""
    if not results:
        return 0.0
    correct = sum(1 for r in results if r['true_label'] == r['pred_label'])
    return correct / len(results)


def main():
    parser = argparse.ArgumentParser(description='分类式推理')
    parser.add_argument('--config', default='configs/config_local.yaml')
    parser.add_argument('--checkpoint', required=True, help='模型权重路径')
    parser.add_argument('--image', type=str, help='单张图像路径')
    parser.add_argument('--dataset', type=str, help='评估数据集 (C-Zoo 或 C-Tai)')
    args = parser.parse_args()
    
    device = get_device()
    model, config = load_model(args.config, args.checkpoint, device)
    
    # 加载身份名称列表
    from data.dataset import ChimpanzeeDataset
    gallery_ds = ChimpanzeeDataset(
        root_dir=config['data']['root_dir'],
        annotation_file='data_CZoo/annotations_czoo.txt',
        image_dir='data_CZoo',
        min_samples_per_identity=1
    )
    identity_names = gallery_ds.identities
    
    # === 模式 1: 单张推理 ===
    if args.image:
        pred_name, confidence, top5 = predict_single(model, args.image, config, device, identity_names)
        print(f"\n{'='*50}")
        print(f"图像: {args.image}")
        print(f"预测: {pred_name} (置信度: {confidence:.2%})")
        print(f"\nTop 5 预测:")
        for i, (name, prob) in enumerate(top5, 1):
            print(f"  {i}. {name}: {prob:.2%}")
        print(f"{'='*50}")
        return
    
    # === 模式 2: 数据集评估 ===
    if args.dataset:
        data_root = config['data']['root_dir']
        
        if args.dataset.lower() in ['czoo_train', 'czoo-train', 'train']:
            data_path = os.path.join(data_root, 'data_CZoo_split', 'train')
        elif args.dataset.lower() in ['czoo_test', 'czoo-test', 'test']:
            data_path = os.path.join(data_root, 'data_CZoo_split', 'test')
        elif args.dataset.lower() in ['ctai', 'c-tai', 'tai']:
            data_path = os.path.join(data_root, 'data_CTai', 'face_images')
        else:
            data_path = args.dataset
        
        if not os.path.exists(data_path):
            print(f"错误：路径不存在 → {data_path}")
            return
        
        print(f"=> 正在加载数据集 {args.dataset}...")
        
        from torchvision.datasets import ImageFolder
        transform = get_val_transform(config)
        dataset = ImageFolder(root=data_path, transform=transform)
        
        # ImageFolder 的类别名就是文件夹名（按字母排序）
        folder_classes = dataset.classes  # ['Alex', 'Alexandra', 'Annett', ...]
        
        # 但模型训练时用的是这个顺序:
        train_identities = identity_names  # ['Alex', 'Jahaga', 'Alexandra', ...]
        
        # 创建文件夹名 → 训练时索引 的映射
        folder_to_train_idx = {name: train_identities.index(name) for name in folder_classes if name in train_identities}
        
        print(f"找到 {len(folder_classes)} 个文件夹类别")
        print(f"映射到 {len(folder_to_train_idx)} 个训练身份")

        # 重新构造带正确标签的 DataLoader
        class CorrectLabelDataset(torch.utils.data.Dataset):
            def __init__(self, base_dataset, folder_to_train_idx):
                self.base = base_dataset
                self.mapping = folder_to_train_idx
            def __len__(self):
                return len(self.base)
            def __getitem__(self, idx):
                img, folder_label = self.base[idx]
                correct_label = self.mapping[folder_classes[folder_label]]
                return img, torch.tensor(correct_label)
        
        corrected_dataset = CorrectLabelDataset(dataset, folder_to_train_idx)
        loader = DataLoader(corrected_dataset, batch_size=64, shuffle=False, num_workers=0)

        print(f"=> 开始推理...")
        results = predict_batch(model, loader, device, identity_names)
        accuracy = compute_accuracy(results)
        
        # 打印逐张结果
        print(f"\n{'='*60}")
        print(f"数据集 {args.dataset} 分类推理结果")
        print(f"{'='*60}")
        print(f"\n总样本数: {len(results)}")
        print(f"正确预测: {sum(1 for r in results if r['true_label'] == r['pred_label'])}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy:.2%})")
        print(f"\n{'='*60}")
        
        # 打印错误样本
        errors = [r for r in results if r['true_label'] != r['pred_label']]
        if errors:
            print(f"\n错误预测样本 (前 10 个):")
            for r in errors[:10]:
                print(f"  实际: {r['true_name']:12s} → 预测: {r['pred_name']:12s} (置信度: {r['confidence']:.2%})")
        
        print(f"{'='*60}")
        return
    
    print("请指定 --image 或 --dataset 参数")


if __name__ == '__main__':
    main()
