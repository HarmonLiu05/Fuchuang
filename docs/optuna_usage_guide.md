# Optuna 超参数优化使用指南

## 概述

本项目使用 Optuna 贝叶斯优化框架自动搜索龟类个体识别模型的最佳超参数组合。

## 架构升级

相比原版配置，做了以下升级：

| 组件 | 原版 | 升级版 |
|------|------|--------|
| 输入分辨率 | 112×112 | 224×224 |
| Backbone | ResNet50 | ResNet50/101 可选 |
| Bottleneck | 单层 Linear | 多层 MLP (2048→1024→512) |
| 注意力机制 | 无 | SE-Block（可选） |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行 Optuna 优化

```bash
# 默认 50 trials（预计 4090 上 7-10 小时）
python optimize.py --config configs/config_optuna.yaml --n-trials 50

# 快速测试 1 个 trial
python optimize.py --config configs/config_optuna.yaml --n-trials 1

# 继续之前的优化
python optimize.py --config configs/config_optuna.yaml --resume
```

### 3. 查看优化结果

优化完成后，结果保存在 `results/optuna_study/` 目录：

```
results/optuna_study/
├── study.db                  # Optuna 数据库（可中断后继续）
├── best_params.yaml          # 最佳参数
├── optimization_history.png  # 优化历史图
├── importance.png            # 参数重要性图
├── parallel_coordinate.png   # 平行坐标图
└── slice.png                 # 单参数切片图
```

### 4. 用最佳参数训练

```bash
# 查看最佳参数
cat results/optuna_study/best_params.yaml

# 手动更新 configs/config_turtle.yaml 中的参数
# 或者使用脚本自动生成配置
python train_turtle.py --config configs/config_turtle.yaml
```

## 云端运行（4090）

### 推荐参数

```bash
# SSH 到云端服务器后：
cd /path/to/chimpanzee_arcface

# 安装依赖
pip install -r requirements.txt

# 启动优化（建议使用 screen 或 tmux 防止断连中断）
screen -S optuna
python optimize.py --config configs/config_optuna.yaml --n-trials 50

# 按 Ctrl+A 然后 D 退出 screen，优化会继续运行
# 重新连接: screen -r optuna
```

### 预计时间

| 配置 | 单 Trial 时间 | 50 Trials |
|------|-------------|-----------|
| 4090 (24GB), batch=32 | ~8-12 分钟 | ~7-10 小时 |
| V100 (16GB), batch=16 | ~15-20 分钟 | ~12-17 小时 |

## 配置文件说明

### configs/config_optuna.yaml

| 字段 | 说明 |
|------|------|
| `data.image_size` | 输入分辨率（224 推荐） |
| `data.dataset_name` | 使用的数据集（推荐 dataset_D_ge3years_count） |
| `model.backbone` | 默认 backbone（Optuna 会覆盖） |
| `training.epochs_per_trial` | 每个 trial 跑多少 epoch（5 用于快速评估） |
| `optuna.n_trials` | 总试验次数 |
| `optuna.sampler` | TPE 贝叶斯优化器 |
| `optuna.pruner` | MedianPruner 自动终止差的 trial |

### 搜索空间

Optuna 会自动搜索以下超参数组合：

| 参数 | 搜索范围 |
|------|---------|
| backbone | resnet50, resnet101 |
| base_lr | 1e-4 ~ 1e-2 |
| backbone_lr | 1e-5 ~ 1e-3 |
| batch_size | 16, 32, 64 |
| dropout | 0.2 ~ 0.6 |
| arcface_m | 0.3 ~ 0.5 |
| arcface_s | 25.0 ~ 35.0 |
| weight_decay | 1e-5 ~ 1e-4 |
| freeze_until_layer | 0 ~ 15 |
| use_se_block | true, false |

## 常见问题

### Q: 优化中途可以停止吗？

可以。Optuna 使用 SQLite 数据库保存进度，下次运行 `--resume` 会继续。

### Q: 如何切换数据集？

修改 `configs/config_optuna.yaml` 中的 `data.dataset_name` 即可。

### Q: 显存不足怎么办？

降低 `batch_size` 搜索范围（在 optimize.py 中修改），或使用 112 分辨率。

### Q: 如何查看优化进度？

```bash
# 使用 optuna dashboard 可视化
pip install optuna-dashboard
optuna-dashboard sqlite:///results/optuna_study/study.db
# 浏览器打开 http://localhost:8080
```

## 预期结果

| 指标 | 当前最佳 | 预期优化后 |
|------|---------|-----------|
| Test Accuracy | 79.58% | 88-93% |
| Backbone | ResNet50 | ResNet101 |
| 分辨率 | 112×112 | 224×224 |
