# 龟类个体识别模型优化设计

## 概述

使用 Optuna 贝叶斯优化框架，对 ResNet + ArcFace 架构进行自动化超参数搜索和模型架构迭代，目标是在 dataset_D_ge3years_count 上达到 88%+ 的测试集 Accuracy。

## 目标

- **主要目标**: 最大化 Test Set Accuracy
- **次要目标**: 找到泛化能力强的超参数组合
- **数据集**: `dataset_D_ge3years_count`（108个体，2620训练图，1194测试图）

## 架构升级

### 1. 输入分辨率提升

| 项目 | 旧值 | 新值 |
|------|------|------|
| 输入尺寸 | 112×112 | 224×224 |
| 优势 | - | 保留更多龟类头部纹理细节 |

### 2. Backbone 多深度支持

支持 ResNet50/101 作为搜索维度：
- ResNet50: 更快，适合快速验证
- ResNet101: 更深，特征表达能力更强

### 3. Bottleneck 升级为多层 MLP

```
旧: Linear(2048→512) → BN → Dropout → L2 Norm
新: Linear(2048→1024) → BN → ReLU → Linear(1024→512) → BN → Dropout → L2 Norm
```

### 4. SE-Block 注意力（可选）

在 ResNet layer4 后添加 Squeeze-and-Excitation 模块，自动学习通道重要性。

## 最终架构

```
输入(224×224×3)
    ↓
ResNet50/101 + SE-Block(可选)
    ↓
AdaptiveAvgPool(7×7) → 2048维
    ↓
MLP Bottleneck (2048→1024→512) + BN + ReLU + Dropout
    ↓
L2 Normalization
    ↓
ArcFace Loss (s, m 可调)
    ↓
分类输出
```

## Optuna 搜索空间

| 参数 | 类型 | 搜索范围 | 说明 |
|------|------|---------|------|
| backbone | Categorical | [resnet50, resnet101] | 模型深度 |
| base_lr | Float (log) | 1e-4 ~ 1e-2 | 主学习率 |
| backbone_lr | Float (log) | 1e-5 ~ 1e-3 | Backbone 学习率 |
| batch_size | Categorical | [16, 32, 64] | 批次大小 |
| dropout | Float | 0.2 ~ 0.6 | 正则化 |
| arcface_m | Float | 0.3 ~ 0.5 | ArcFace margin |
| arcface_s | Float | 25.0 ~ 35.0 | ArcFace scale |
| weight_decay | Float (log) | 1e-5 ~ 1e-3 | 权重衰减 |
| freeze_until_epoch | Int | 0 ~ 15 | Backbone 冻结轮数 |
| use_se_block | Categorical | [true, false] | SE 注意力 |

## 训练策略

- **Time Budget**: 每个 trial 固定 5 epochs（快速评估）
- **Pruning**: 使用 MedianPruner，如果某个 trial 在 epoch 2 的 accuracy 低于历史中位数，提前终止
- **Trials**: 50 次（预计 4090 上 7-10 小时）

## 文件结构

```
chimpanzee_arcface/
├── optimize.py                      # Optuna 优化入口
├── configs/config_optuna.yaml       # Optuna 配置
├── configs/config_turtle.yaml       # 更新: 支持 224 分辨率
├── models/
│   ├── backbone.py                  # 升级: 支持 ResNet101
│   ├── bottleneck.py                # 升级: 多层 MLP
│   └── se_block.py                  # 新增: SE 注意力模块
└── results/optuna_study/            # 优化结果（自动创建）
    ├── study.db                     # Optuna 数据库
    ├── best_params.yaml             # 最佳参数
    └── importance.png               # 参数重要性图
```

## 运行方式

```bash
# 安装依赖
pip install optuna

# 启动优化
python optimize.py --config configs/config_optuna.yaml --n-trials 50

# 用最佳参数重新训练
python train_turtle.py --config configs/config_turtle.yaml --best-params results/optuna_study/best_params.yaml
```

## 预期结果

| 指标 | 当前 | 预期 |
|------|------|------|
| Test Accuracy | 79.58% | 88-93% |
| Backbone | ResNet50 | ResNet101 |
| 输入分辨率 | 112×112 | 224×224 |
