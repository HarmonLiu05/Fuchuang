"""
推理脚本：输入一张大猩猩面部图片，输出预测的个体 ID

支持两种模式：
1. 闭集识别：已知个体 → 直接输出预测 ID
2. 开集识别：未知个体 → 输出 "Unknown" (置信度低于阈值)
"""
import os
import sys
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(__file__))

from train import ChimpFaceModel
from utils.utils import load_config, get_device


def predict_single_image(model, image_path, config, device, 
                        identity_names=None, threshold=None):
    """
    预测单张图片
    
    Args:
        model: 训练好的模型
        image_path: 图片路径
        config: 配置字典
        device: 计算设备
        identity_names: 个体名称列表（用于输出名称而非索引）
        threshold: 开集识别阈值（None 表示闭集识别）
    
    Returns:
        pred_identity: 预测的个体名称或 "Unknown"
        confidence: 置信度
    """
    # 加载和预处理图像
    transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # 提取特征
    model.eval()
    with torch.no_grad():
        features = model.backbone(input_tensor)
        features = model.bottleneck(features)  # 已 L2 归一化
        
        # 计算与 ArcFace 权重的相似度
        weight_norm = F.normalize(model.arcface.weight, p=2, dim=1)
        similarity = torch.mm(features, weight_norm.t()) * model.arcface.s
    
    # 获取预测
    pred_idx = similarity.argmax(dim=1).item()
    confidence = F.softmax(similarity, dim=1)[0, pred_idx].item()
    
    if identity_names:
        pred_identity = identity_names[pred_idx]
    else:
        pred_identity = f"identity_{pred_idx}"
    
    # 开集识别：如果置信度低于阈值，标记为 Unknown
    if threshold is not None and confidence < threshold:
        pred_identity = "Unknown"
    
    return pred_identity, confidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_local.yaml')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--image', type=str, required=True, help='输入图片路径')
    parser.add_argument('--open_set', action='store_true', help='启用开集识别')
    parser.add_argument('--threshold', type=float, default=0.5, help='开集识别阈值')
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = get_device()
    
    print(f"加载检查点: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # 重建模型
    arcface_weight = checkpoint['model_state_dict']['arcface.weight']
    num_identities = arcface_weight.shape[0]
    
    model = ChimpFaceModel(config, num_identities)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # 预测
    pred_identity, confidence = predict_single_image(
        model, args.image, config, device,
        threshold=args.threshold if args.open_set else None
    )
    
    print(f"\n图片: {args.image}")
    print(f"预测: {pred_identity}")
    print(f"置信度: {confidence:.4f}")


if __name__ == '__main__':
    main()
