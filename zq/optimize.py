"""
Optuna 超参数优化脚本
自动搜索最佳模型架构和超参数组合
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import optuna
from optuna.trial import TrialState
import yaml

sys.path.insert(0, os.path.dirname(__file__))

from models.backbone import ResNetBackbone
from models.bottleneck import Bottleneck
from models.arcface import ArcFace
from models.se_block import add_se_to_resnet_layer
from data.turtle_dataset import TurtleDataset
from data.dataset import get_train_transform, get_val_transform
from utils.utils import load_config, set_seed, get_device, ensure_dir
from utils.metrics import compute_all_metrics
from torch.utils.data import DataLoader


class TurtleFaceModel(nn.Module):
    def __init__(self, config, num_identities, trial_params=None):
        super().__init__()
        model_cfg = config['model'].copy()
        
        # 如果 trial 提供了参数，覆盖默认配置
        if trial_params:
            model_cfg.update(trial_params)
        
        backbone_name = model_cfg.get('backbone', 'resnet50')
        freeze_until = model_cfg.get('freeze_until_layer', 3)
        
        self.backbone = ResNetBackbone(
            pretrained=model_cfg['pretrained'],
            model_name=backbone_name,
            freeze_until_layer=freeze_until
        )
        
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
def quick_evaluate(model, dataloader, device, max_batches=None):
    """快速评估（用于 Optuna pruning）"""
    model.eval()
    features_list, labels_list = [], []
    
    # 处理 DataParallel 包装
    backbone = model.module.backbone if hasattr(model, 'module') else model.backbone
    bottleneck = model.module.bottleneck if hasattr(model, 'module') else model.bottleneck
    
    for i, (images, labels) in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
        if max_batches and i >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        feats = backbone(images)
        feats = bottleneck(feats)
        features_list.append(feats.cpu())
        labels_list.append(labels)
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    metrics = compute_all_metrics(features, labels)
    return metrics['accuracy']


def train_one_epoch(model, train_loader, optimizer, device, epoch, config, accumulation_steps):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        output, _ = model(images, labels)
        loss = nn.CrossEntropyLoss(label_smoothing=config['model'].get('label_smoothing', 0.1))(
            output, labels
        )
        loss = loss / accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
    
    return total_loss / len(train_loader)


def objective(trial, config, device):
    """Optuna 目标函数"""
    data_cfg = config['data']
    train_cfg = config['training']
    
    # 1. 从 trial 采样超参数
    trial_params = {
        'backbone': trial.suggest_categorical('backbone', ['resnet50', 'resnet101']),
        'base_lr': trial.suggest_float('base_lr', 1e-4, 1e-2, log=True),
        'backbone_lr': trial.suggest_float('backbone_lr', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'dropout': trial.suggest_float('dropout', 0.2, 0.6),
        'arcface_m': trial.suggest_float('arcface_m', 0.3, 0.5),
        'arcface_s': trial.suggest_float('arcface_s', 25.0, 35.0),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-4, log=True),
        'freeze_until_layer': trial.suggest_int('freeze_until_layer', 0, 15),
        'use_se_block': trial.suggest_categorical('use_se_block', [True, False]),
    }
    
    # 更新配置的 batch_size
    data_cfg['batch_size'] = trial_params['batch_size']
    
    # 2. 准备数据
    # 创建包含 image_size 的完整配置用于 transform
    transform_config = config.copy()
    transform_config['data'] = transform_config['data'].copy()
    transform_config['data']['image_size'] = data_cfg['image_size']
    transform_config['data']['batch_size'] = trial_params['batch_size']
    
    # 创建训练集
    train_dataset = TurtleDataset(
        config,
        split='train',
        transform=get_train_transform(transform_config)
    )
    
    # 创建测试集（共享 identity_map）
    val_dataset = TurtleDataset(
        config,
        split='test',
        transform=get_val_transform(transform_config),
        identity_map=train_dataset.identity_map
    )
    
    num_identities = train_dataset.num_identities
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=trial_params['batch_size'],
        shuffle=True,
        num_workers=data_cfg.get('num_workers', 0),
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=trial_params['batch_size'],
        shuffle=False,
        num_workers=data_cfg.get('num_workers', 0),
        pin_memory=torch.cuda.is_available()
    )
    
    # 3. 创建模型
    model = TurtleFaceModel(config, num_identities, trial_params)
    model = model.to(device)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
    
    # 4. 创建优化器（分组学习率）
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': trial_params['backbone_lr']},
        {'params': other_params, 'lr': trial_params['base_lr']}
    ], weight_decay=trial_params['weight_decay'])
    
    # 5. 训练循环（每个 epoch 后报告 accuracy 用于 pruning）
    epochs = train_cfg['epochs_per_trial']
    accumulation_steps = train_cfg.get('accumulation_steps', 2)
    
    for epoch in range(1, epochs + 1):
        # 解冻 backbone
        if epoch > trial_params['freeze_until_layer']:
            if hasattr(model, 'module'):
                model.module.unfreeze_backbone()
            else:
                model.unfreeze_backbone()
        
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, config, accumulation_steps)
        
        # 每个 epoch 后评估
        accuracy = quick_evaluate(model, val_loader, device, max_batches=20)
        
        # 报告给 Optuna（用于 pruning）
        trial.report(accuracy, epoch)
        
        # 检查是否应该 pruning
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return accuracy


def create_study(config):
    """创建 Optuna Study"""
    optuna_cfg = config['optuna']
    output_dir = config['output']['dir']
    ensure_dir(output_dir)
    
    # Pruner
    if optuna_cfg['pruner'] == 'median':
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=optuna_cfg['pruner_n_min_trials'],
            n_warmup_steps=optuna_cfg['pruner_n_warmup_steps']
        )
    elif optuna_cfg['pruner'] == 'hyperband':
        pruner = optuna.pruners.HyperbandPruner()
    else:
        pruner = optuna.pruners.MedianPruner()
    
    # Sampler
    sampler = optuna.samplers.TPESampler(n_startup_trials=optuna_cfg['n_startup_trials'])
    
    # 创建 Study
    study = optuna.create_study(
        direction=optuna_cfg['direction'],
        sampler=sampler,
        pruner=pruner,
        study_name=optuna_cfg['study_name'],
        storage=optuna_cfg['storage'],
        load_if_exists=True
    )
    
    return study


def save_results(study, config):
    """保存优化结果可视化"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate, plot_slice
    
    output_dir = config['output']['dir']
    
    # 最佳参数
    best_params = study.best_params
    best_value = study.best_value
    
    print(f"\n{'='*60}")
    print(f"优化完成!")
    print(f"最佳 Test Accuracy: {best_value:.4f}")
    print(f"最佳参数:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}\n")
    
    # 保存最佳参数
    if config['output']['save_best_params']:
        best_params_yaml = os.path.join(output_dir, 'best_params.yaml')
        with open(best_params_yaml, 'w') as f:
            yaml.dump({
                'accuracy': best_value,
                'params': best_params
            }, f, default_flow_style=False)
        print(f"最佳参数已保存到: {best_params_yaml}")
    
    # 可视化
    try:
        # 优化历史
        fig = plot_optimization_history(study)
        fig.write_image(os.path.join(output_dir, 'optimization_history.png'))
        
        if config['output']['plot_importance']:
            fig = plot_param_importances(study)
            fig.write_image(os.path.join(output_dir, 'importance.png'))
            print(f"参数重要性图已保存到: {os.path.join(output_dir, 'importance.png')}")
        
        if config['output']['plot_parallel_coordinate']:
            fig = plot_parallel_coordinate(study)
            fig.write_image(os.path.join(output_dir, 'parallel_coordinate.png'))
        
        if config['output']['plot_slice']:
            fig = plot_slice(study)
            fig.write_image(os.path.join(output_dir, 'slice.png'))
        
        print(f"所有可视化结果已保存到: {output_dir}")
    except Exception as e:
        print(f"可视化保存失败: {e}")
        print(f"但最佳参数已保存到: {os.path.join(output_dir, 'best_params.yaml')}")


def main():
    parser = argparse.ArgumentParser(description='Optuna 超参数优化')
    parser.add_argument('--config', default='configs/config_optuna.yaml', help='Optuna 配置文件')
    parser.add_argument('--n-trials', type=int, default=None, help='覆盖配置文件中的 n_trials')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true', help='继续之前的优化')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    set_seed(args.seed)
    
    # 覆盖 n_trials
    if args.n_trials:
        config['optuna']['n_trials'] = args.n_trials
    
    device = get_device()
    print(f"使用设备: {device}")
    print(f"数据集: {config['data']['dataset_name']}")
    print(f"优化 Trials: {config['optuna']['n_trials']}")
    print(f"每 Trial Epochs: {config['training']['epochs_per_trial']}")
    
    # 创建 Study
    study = create_study(config)
    
    # 运行优化
    n_trials = config['optuna']['n_trials']
    print(f"\n开始优化... (总 Trials: {n_trials})")
    
    study.optimize(
        lambda trial: objective(trial, config, device),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # 保存结果
    save_results(study, config)


if __name__ == '__main__':
    main()
