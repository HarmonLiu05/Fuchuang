"""
龟类个体识别训练脚本
基于 ResNet101 + ArcFace，使用COCO格式标注
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
from losses.triplet import BatchHardTripletLoss
from losses.time_weighted_triplet import TimeWeightedTripletLoss
from losses.temporal_apn_triplet import TemporalAPNTripletLoss
from samplers.time_aware_sampler import _to_numeric_time


def build_optimizer(model, training_cfg):
    """构建分层学习率的优化器"""
    backbone_lr = training_cfg['backbone_lr']
    base_lr = training_cfg['base_lr']
    backbone_lr_scales = training_cfg.get('backbone_lr_scales', {
        'stem': 0.1,
        'layer1': 0.1,
        'layer2': 0.25,
        'layer3': 0.5,
        'layer4': 1.0,
    })

    stem_params = list(model.backbone.conv1.parameters()) + list(model.backbone.bn1.parameters())
    param_groups = [
        {'name': 'stem', 'params': stem_params, 'lr': backbone_lr * backbone_lr_scales.get('stem', 0.1)},
        {'name': 'layer1', 'params': model.backbone.layer1.parameters(), 'lr': backbone_lr * backbone_lr_scales.get('layer1', 0.1)},
        {'name': 'layer2', 'params': model.backbone.layer2.parameters(), 'lr': backbone_lr * backbone_lr_scales.get('layer2', 0.25)},
        {'name': 'layer3', 'params': model.backbone.layer3.parameters(), 'lr': backbone_lr * backbone_lr_scales.get('layer3', 0.5)},
        {'name': 'layer4', 'params': model.backbone.layer4.parameters(), 'lr': backbone_lr * backbone_lr_scales.get('layer4', 1.0)},
        {'name': 'bottleneck', 'params': model.bottleneck.parameters(), 'lr': base_lr},
        {'name': 'arcface', 'params': model.arcface.parameters(), 'lr': base_lr},
    ]
    return torch.optim.Adam(param_groups, weight_decay=training_cfg['weight_decay'])


def update_optimizer_backbone_lrs(optimizer, training_cfg):
    """解冻 backbone 后更新所有层的学习率"""
    backbone_lr = training_cfg['backbone_lr']
    backbone_lr_scales = training_cfg.get('backbone_lr_scales', {})
    for param_group in optimizer.param_groups:
        group_name = param_group.get('name')
        if group_name in {'stem', 'layer1', 'layer2', 'layer3', 'layer4'}:
            param_group['lr'] = backbone_lr * backbone_lr_scales.get(group_name, 1.0)


def compute_triplet_weight(epoch, training_cfg):
    """计算当前 epoch 的 triplet_weight（带 warmup）"""
    max_weight = training_cfg.get('triplet_weight', 0.2)
    start_epoch = training_cfg.get('triplet_start_epoch', 80)
    warmup_epochs = training_cfg.get('triplet_warmup_epochs', 20)

    if epoch < start_epoch:
        return 0.0
    if warmup_epochs <= 0:
        return max_weight

    progress = min(1.0, (epoch - start_epoch + 1) / warmup_epochs)
    return max_weight * progress


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
    # 评估时关闭 ArcFace margin，使用标准余弦相似度
    original_margin = model.arcface.m
    was_training = model.training
    model.arcface.update_margin(0.0)
    model.eval()

    try:
        features_list, labels_list, ids_list = [], [], []
        for batch in tqdm(dataloader, desc="Evaluating"):
            images, labels = batch[:2]
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
    finally:
        # 确保 margin 和模型状态始终恢复
        model.arcface.update_margin(original_margin)
        if was_training:
            model.train()

    metrics = compute_all_metrics(features, labels)
    metrics['acc_direct'] = compute_pred_accuracy(ids, labels)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config_turtle.yaml')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default=None, help='从检查点恢复训练')
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)
    device = get_device()
    print(f"使用设备: {device}")
    print(f"配置文件: {args.config}")

    # 准备数据
    use_time_weighted = config['training'].get('use_time_weighted_triplet', False)
    use_temporal_apn = config['training'].get('use_temporal_apn_triplet', False)
    use_time_info = use_time_weighted or use_temporal_apn
    
    # 时间APN会使用fallback机制（时间选不出时用距离选）
    # 是否使用 TimeAwareBatchSampler 由配置文件决定（默认为 False）
    use_sampler = config['training'].get('use_time_aware_sampler', False)
    train_loader, test_loader, num_identities, train_dataset = prepare_turtle_dataloaders(
        config, return_time=use_time_info, use_time_aware_sampler=use_sampler
    )
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"测试集: {len(test_loader.dataset)} 样本")

    # 模型、优化器、调度器
    model = TurtleFaceModel(config, num_identities).to(device)
    optimizer = build_optimizer(model, config['training'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['epochs'],
        eta_min=config['training']['eta_min']
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=config['model']['label_smoothing'])
    
    # 根据配置选择 Triplet Loss 类型
    use_time_weighted = config['training'].get('use_time_weighted_triplet', False)
    use_temporal_apn = config['training'].get('use_temporal_apn_triplet', False)
    
    if use_temporal_apn:
        print(">>> 启用时间跨度感知 APN Triplet Loss")
        triplet_criterion = TemporalAPNTripletLoss(
            margin=config['training'].get('triplet_margin', 0.3),
            normalize_embeddings=True
        )
        use_time_info = True  # ensure time data is loaded
    elif use_time_weighted:
        print(">>> 启用时间加权 Triplet Loss")
        triplet_criterion = TimeWeightedTripletLoss(
            margin=config['training'].get('triplet_margin', 0.3),
            alpha=config['training'].get('time_alpha', 0.3),
            max_time_gap=config['training'].get('max_time_gap', 10.0),
            normalize_embeddings=True
        )
    else:
        print(">>> 使用普通 Batch-Hard Triplet Loss")
        triplet_criterion = BatchHardTripletLoss(
            margin=config['training'].get('triplet_margin', 0.3),
            normalize_embeddings=True
        )
    scaler = torch.amp.GradScaler('cuda')
    checkpoint_dir = config['training']['checkpoint_dir']
    ensure_dir(checkpoint_dir)

    best_acc = 0.0
    best_acc_direct = 0.0
    start_epoch = 0
    accumulation_steps = config['training']['accumulation_steps']

    # === 恢复检查点 ===
    if args.resume:
        print(f"\n>>> 从检查点恢复: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        
        checkpoint_state = checkpoint['model_state_dict']
        current_state = model.state_dict()
        
        # 严格检查形状
        mismatched_layers = []
        for name, param in checkpoint_state.items():
            if name in current_state and param.shape != current_state[name].shape:
                mismatched_layers.append(f"{name} (checkpoint: {param.shape}, current: {current_state[name].shape})")
        
        if mismatched_layers:
            raise RuntimeError(
                f"❌ 检查点与当前模型结构不匹配，无法接续训练！\n"
                f"不匹配的层:\n" + "\n".join(f"  - {l}" for l in mismatched_layers) + "\n"
                f"请确保检查点与当前配置的个体数量（num_identities）一致。"
            )
        
        # 加载模型权重
        model.load_state_dict(checkpoint_state)
        
        # 关键修改 1：不加载旧的优化器状态，保持配置文件中的初始学习率
        # 这样可以避免旧的低学习率（如 1e-6）影响新任务
        print("   ℹ️   不加载旧优化器状态，使用配置文件的初始学习率")
        
        # 关键修改 2：根据加载权重的轮次，动态调整 Triplet 开始轮次
        # 逻辑：既然加载的是第 56 轮的权重（最佳模型），那就从第 57 轮开始引入 Triplet
        # 这样就不需要干等 80 轮，避免了中间的空窗期
        checkpoint_epoch = checkpoint.get('epoch', 0)
        start_epoch = checkpoint_epoch + 1
        
        # 强制设置 Triplet 从当前轮次开始，让代码立刻开始 ramp-up
        config['training']['triplet_start_epoch'] = checkpoint_epoch
        print(f"   🎯 动态调整: Triplet Loss 将从第 {start_epoch} 轮开始引入 (Warmup)")
        
        # 关键修改 3：重建 Scheduler 以对齐新的总轮次 (200)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['training']['epochs'],
            eta_min=config['training']['eta_min']
        )
        
        # 步进到指定的 start_epoch (让学习率平滑过渡到当前水平)
        for _ in range(start_epoch):
            scheduler.step()
            
        # 打印最大学习率（通常是 ArcFace 或 Bottleneck 的学习率）作为参考
        # 因为第一个组（Stem）的学习率通常很低，打印它会产生误导
        max_lr = max(group['lr'] for group in optimizer.param_groups)
        print(f"   ✅ 模拟接续训练: 已跳过前 {start_epoch} 轮")
        print(f"   🔄 已重置 Scheduler (T_max={config['training']['epochs']})")
        print(f"   ✅ 当前最大学习率: {max_lr:.6f}")

    print(f"\n开始训练: {start_epoch} → {config['training']['epochs']} epochs")
    print(f"Batch size: {config['training']['batch_size']}, Accumulation: {accumulation_steps}")
    if use_temporal_apn:
        triplet_type = 'Temporal-APN'
    elif use_time_weighted:
        triplet_type = 'Time-Weighted'
    else:
        triplet_type = 'Batch-Hard'
    print(f"Triplet Loss: {triplet_type} (weight={config['training']['triplet_weight']}, alpha={config['training'].get('time_alpha', 0)})")

    for epoch in range(start_epoch, config['training']['epochs']):
        # === 训练 ===
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")

        current_triplet_weight = compute_triplet_weight(epoch, config['training'])

        for batch_idx, batch in enumerate(pbar):
            images, labels = batch[:2]
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # 如果使用需要时间信息的 Triplet Loss，提取时间信息
            if use_time_info:
                times_list = [_to_numeric_time(batch[2][i] if len(batch) > 2 else None)
                              for i in range(len(images))]
                times = torch.tensor([t if t is not None else float('nan') for t in times_list],
                                    dtype=torch.float32).to(device)

            amp_enabled = torch.cuda.is_available() and (config['training']['precision'] == 16)
            with torch.amp.autocast('cuda', enabled=amp_enabled):
                output, features = model(images, labels)
                cls_loss = criterion(output, labels)

                if current_triplet_weight > 0:
                    if use_time_info:
                        tri_loss = triplet_criterion(features, labels, times)
                    else:
                        tri_loss = triplet_criterion(features, labels)
                else:
                    tri_loss = torch.zeros(1, device=device, dtype=cls_loss.dtype).squeeze(0)
                    
                total_batch_loss = cls_loss + current_triplet_weight * tri_loss
                loss = total_batch_loss / accumulation_steps

            scaler.scale(loss).backward()
            total_loss += loss.item() * accumulation_steps

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            pbar.set_postfix({
                'loss': f'{total_loss / (batch_idx + 1):.4f}',
                'tri_w': f'{current_triplet_weight:.3f}'
            })

        # === 验证 ===
        metrics = evaluate(model, test_loader, device)
        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        acc = metrics['accuracy']
        acc_direct = metrics['acc_direct']
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={acc:.4f}, Acc_direct={acc_direct:.4f}, "
              f"TripletW={current_triplet_weight:.3f}, LR={scheduler.get_last_lr()[0]:.6f}")

        # === 保存最佳模型（基于 Acc_direct） ===
        if acc_direct > best_acc_direct:
            best_acc_direct = acc_direct
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'accuracy': acc,
                'acc_direct': acc_direct,
                'config': config
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model1.pth'))
            print(f"  ↳ 保存最佳模型! Acc_direct={best_acc_direct:.4f}, Acc={acc:.4f}")

        # 阶段 2: 解冻 backbone
        if epoch + 1 == config['training'].get('freeze_until_epoch', 10):
            print(">>> 解冻整个 backbone，并启用所有 backbone 参数更新...")
            model.unfreeze_backbone()
            update_optimizer_backbone_lrs(optimizer, config['training'])

    print(f"\n训练完成! 最佳 Acc_direct: {best_acc_direct:.4f}, Acc: {best_acc:.4f}")
    print(f"模型保存在: {os.path.join(checkpoint_dir, 'best_model1.pth')}")


if __name__ == '__main__':
    main()
