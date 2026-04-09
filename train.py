"""
ResNet50 + ArcFace 训练脚本
支持混合精度训练、梯度累积、分阶段训练
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from models.backbone import ResNet50Backbone
from models.bottleneck import Bottleneck
from models.arcface import ArcFace
from data.prepare_data import prepare_dataloaders
from data.dataset import get_train_transform, get_val_transform
from utils.metrics import compute_all_metrics
from utils.utils import load_config, set_seed, get_device, ensure_dir


class ChimpFaceModel(nn.Module):
    """完整的大猩猩识别模型"""
    def __init__(self, config, num_identities):
        super().__init__()
        self.config = config
        model_cfg = config['model']
        
        # 阶段 1: 冻结 layer1-3
        self.backbone = ResNet50Backbone(
            pretrained=model_cfg['pretrained'],
            freeze_until_layer=3
        )
        
        self.bottleneck = Bottleneck(
            in_features=2048,
            bottleneck_dim=model_cfg['bottleneck_dim'],
            dropout=model_cfg['dropout']
        )
        
        self.arcface = ArcFace(
            in_features=model_cfg['bottleneck_dim'],
            num_classes=num_identities,
            s=model_cfg['arcface_s'],
            m=model_cfg['arcface_m']
        )
    
    def forward(self, images, labels):
        features = self.backbone(images)
        features = self.bottleneck(features)
        output = self.arcface(features, labels)
        return output, features
    
    def unfreeze_backbone(self):
        """阶段 2: 解冻 backbone"""
        self.backbone.unfreeze_all()


def train_one_epoch(model, dataloader, criterion, optimizer, 
                    accumulation_steps, scaler, device, epoch, config):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with autocast(enabled=config['training']['precision'] == 16):
            output, _ = model(images, labels)
            loss = criterion(output, labels) / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item() * accumulation_steps:.4f}"})
    
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, device):
    """验证模型"""
    model.eval()
    features_list = []
    labels_list = []
    
    for images, labels in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        
        feats = model.backbone(images)
        feats = model.bottleneck(feats)
        
        features_list.append(feats.cpu())
        labels_list.append(labels)
    
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    metrics = compute_all_metrics(features, labels)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_local.yaml')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    set_seed(args.seed)
    device = get_device()
    
    print(f"使用设备: {device}")
    print(f"配置文件: {args.config}")
    
    # 准备数据
    train_loader, val_loader, num_identities, dataset = prepare_dataloaders(config)
    
    # 创建模型
    model = ChimpFaceModel(config, num_identities)
    model = model.to(device)
    
    # 优化器（包含 backbone.layer4）
    optimizer = torch.optim.Adam([
        {'params': model.backbone.layer4.parameters(), 'lr': config['training']['backbone_lr']},
        {'params': model.bottleneck.parameters(), 'lr': config['training']['base_lr']},
        {'params': model.arcface.parameters(), 'lr': config['training']['base_lr']}
    ], weight_decay=config['training']['weight_decay'])
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=config['training']['eta_min']
    )
    
    # 损失函数（带 Label Smoothing）
    criterion = nn.CrossEntropyLoss(label_smoothing=config['model']['label_smoothing'])
    
    # 混合精度
    scaler = GradScaler(enabled=config['training']['precision'] == 16)
    
    # 检查点目录
    checkpoint_dir = ensure_dir(config['training']['checkpoint_dir'])
    
    # 训练循环
    best_accuracy = 0.0
    freeze_until_epoch = config['training']['freeze_until_epoch']
    
    for epoch in range(config['training']['epochs']):
        # 阶段 2: 解冻 backbone
        if epoch == freeze_until_epoch:
            print(f"\n--- Epoch {epoch}: 解冻 backbone ---\n")
            model.unfreeze_backbone()
            
            # 更新优化器（添加全部 backbone 参数）
            optimizer = torch.optim.Adam([
                {'params': model.backbone.parameters(), 'lr': config['training']['backbone_lr']},
                {'params': model.bottleneck.parameters(), 'lr': config['training']['base_lr']},
                {'params': model.arcface.parameters(), 'lr': config['training']['base_lr']}
            ], weight_decay=config['training']['weight_decay'])
        
        # 训练
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer,
            config['training']['accumulation_steps'], scaler, device, epoch, config
        )
        
        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # 验证
        metrics = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{config['training']['epochs']}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  LR: {current_lr:.6f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  TAR@FAR=0.1%: {metrics['tar_at_far_0.1']:.4f}")
        print(f"  Threshold: {metrics['threshold']:.4f}")
        
        # 保存最佳模型
        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': metrics['accuracy'],
                'config': config
            }, checkpoint_path)
            print(f"  *** 保存最佳模型 (Accuracy: {best_accuracy:.4f}) ***")
    
    print(f"\n训练完成！最佳 Accuracy: {best_accuracy:.4f}")


if __name__ == '__main__':
    main()
