"""
龟类个体识别训练脚本
基于 ResNet50 + ArcFace，使用COCO格式标注
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from models.backbone import ResNetBackbone
from models.bottleneck import Bottleneck
from models.arcface import ArcFace
from models.se_block import add_se_to_resnet_layer
from data.prepare_turtle_data import prepare_turtle_dataloaders
from data.dataset import get_train_transform, get_val_transform
from utils.utils import load_config, set_seed, get_device, ensure_dir
from utils.metrics import compute_all_metrics, compute_pred_accuracy


class TurtleFaceModel(nn.Module):
    def __init__(self, config, num_identities):
        super().__init__()
        model_cfg = config['model']
        
        # 支持多种 backbone
        backbone_name = model_cfg.get('backbone', 'resnet50')
        if backbone_name.startswith('resnet'):
            self.backbone = ResNetBackbone(
                pretrained=model_cfg['pretrained'],
                model_name=backbone_name,
                freeze_until_layer=model_cfg.get('freeze_until_layer', 3)
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # 可选：添加 SE-Block
        if model_cfg.get('use_se_block', False):
            add_se_to_resnet_layer(self.backbone.layer4, reduction=16)
        
        self.bottleneck = Bottleneck(
            in_features=2048,
            bottleneck_dim=model_cfg['bottleneck_dim'],
            dropout=model_cfg['dropout'],
            use_mlp=model_cfg.get('use_mlp_bottleneck', True)
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
        self.backbone.unfreeze_all()


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    features_list, labels_list, ids_list = [], [], []
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

    metrics = compute_all_metrics(features, labels)
    metrics['accuracy0'] = compute_pred_accuracy(ids, labels)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config_turtle.yaml')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)
    device = get_device()
    print(f"使用设备: {device}")
    print(f"配置文件: {args.config}")

    # 准备数据
    train_loader, test_loader, num_identities, train_dataset = prepare_turtle_dataloaders(config)
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"测试集: {len(test_loader.dataset)} 样本")

    # 模型、优化器、调度器
    model = TurtleFaceModel(config, num_identities).to(device)
    optimizer = torch.optim.Adam([
        {'params': model.backbone.layer4.parameters(), 'lr': config['training']['backbone_lr']},
        {'params': model.bottleneck.parameters(), 'lr': config['training']['base_lr']},
        {'params': model.arcface.parameters(), 'lr': config['training']['base_lr']}
    ], weight_decay=config['training']['weight_decay'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['epochs'],
        eta_min=config['training']['eta_min']
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=config['model']['label_smoothing'])
    scaler = torch.amp.GradScaler('cuda')
    checkpoint_dir = config['training']['checkpoint_dir']
    ensure_dir(checkpoint_dir)

    best_acc = 0.0
    best_acc0 = 0.0
    accumulation_steps = config['training']['accumulation_steps']

    print(f"\n开始训练: {config['training']['epochs']} epochs")
    print(f"Batch size: {config['training']['batch_size']}, Accumulation: {accumulation_steps}")

    for epoch in range(config['training']['epochs']):
        # === 训练 ===
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=(config['training']['precision'] == 16)):
                output, features = model(images, labels)
                loss = criterion(output, labels) / accumulation_steps

            scaler.scale(loss).backward()
            total_loss += loss.item() * accumulation_steps

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            pbar.set_postfix({'loss': f'{total_loss / (batch_idx + 1):.4f}'})

        # === 验证 ===
        metrics = evaluate(model, test_loader, device)
        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        acc = metrics['accuracy']
        acc0 = metrics['accuracy0']
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={acc:.4f}, Acc0={acc0:.4f}, "
              f"LR={scheduler.get_last_lr()[0]:.6f}")

        # === 保存最佳模型（基于 Accuracy0） ===
        if acc0 > best_acc0:
            best_acc0 = acc0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': acc,
                'accuracy0': acc0,
                'config': config
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"  ↳ 保存最佳模型! Acc0={best_acc0:.4f}, Acc={acc:.4f}")

        # 阶段 2: 解冻 backbone
        if epoch + 1 == config['training'].get('freeze_until_epoch', 10):
            print(">>> 解冻 backbone layer4...")
            model.unfreeze_backbone()
            optimizer.param_groups[0]['lr'] = config['training']['base_lr'] * 0.1

    print(f"\n训练完成! 最佳 Accuracy0: {best_acc0:.4f}, Accuracy: {best_acc:.4f}")
    print(f"模型保存在: {os.path.join(checkpoint_dir, 'best_model.pth')}")


if __name__ == '__main__':
    main()
