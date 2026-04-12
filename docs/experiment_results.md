# 实验记录

## 实验 1: 默认配置基线

| 项目 | 详情 |
|------|------|
| **日期** | 2026-04-10 |
| **数据集** | dataset_D_ge3years_count (107个体, 2620训练图, 1194测试图) |
| **配置** | configs/config_turtle.yaml |

### 模型架构

| 组件 | 配置 |
|------|------|
| **输入分辨率** | 224×224 |
| **Backbone** | ResNet101 |
| **Bottleneck** | 多层 MLP (2048→1024→512) |
| **SE-Block** | 关闭 |
| **Dropout** | 0.4 |

### 训练参数

| 参数 | 值 |
|------|-----|
| batch_size | 16 |
| accumulation_steps | 4 |
| base_lr | 0.001 |
| backbone_lr | 0.0001 |
| weight_decay | 0.0005 |
| epochs | 30 |
| freeze_until_epoch | 10 |
| arcface_s | 30.0 |
| arcface_m | 0.4 |
| label_smoothing | 0.1 |

### 数据增强

| 增强 | 值 |
|------|-----|
| random_horizontal_flip | 0.5 |
| random_affine_degrees | 15 |
| color_jitter | brightness:0.3, contrast:0.3, saturation:0.2, hue:0.1 |
| gaussian_blur_kernel | 3 |
| random_erasing_p | 0.5 |

### 结果

| 指标 | 值 |
|------|-----|
| **Test Accuracy** | **86.61%** |

---

## 实验 2: GPU 优化配置（batch=128, 精简增强）

| 项目 | 详情 |
|------|------|
| **日期** | 2026-04-10 |
| **数据集** | dataset_D_ge3years_count |
| **目标** | 最大化 GPU 利用率 (100%) |

### 配置对比（vs 实验1）

| 参数 | 实验1 | 实验2 |
|------|-------|-------|
| batch_size | 16 | **128** |
| accumulation_steps | 4 | 1 |
| num_workers | 8 | 4 |
| color_jitter | 全开 | **仅 brightness** |
| random_affine | 15 | **0（关闭）** |
| random_erasing_p | 0.5 | **0（关闭）** |

### GPU 指标

| 指标 | 实验1 | 实验2 |
|------|-------|-------|
| GPU-Util | ~59% | **100%** |
| 显存 | ~6GB | **~11GB** |

### 结果

| 指标 | 实验1 | 实验2 | 变化 |
|------|-------|-------|------|
| **Test Accuracy** | 86.61% | **83.94%** | **↓ 2.67%** |

### 分析

- **下降原因**：过度减少数据增强导致模型泛化能力不足
- **教训**：GPU 利用率 100% 不代表效果好，需要在速度和增强之间平衡

---

## 实验 3: 平衡配置（batch=128, 适中增强）

| 项目 | 详情 |
|------|------|
| **日期** | 2026-04-10 |
| **数据集** | dataset_D_ge3years_count |
| **目标** | 平衡 GPU 利用率和数据增强强度 |

### 配置对比

| 参数 | 实验2 | 实验3 |
|------|-------|-------|
| batch_size | 128 | 128 |
| accumulation_steps | 1 | 1 |
| random_affine_degrees | 0 | **5** |
| color_jitter | 仅 brightness | **全开（轻量）** |
| random_erasing_p | 0 | **0.2** |

### 结果

| 指标 | 实验1 | 实验2 | 实验3 | 对比实验1 |
|------|-------|-------|-------|---------|
| **Test Accuracy** | **86.61%** | 83.94% | **84.63%** | **↓ 1.98%** |

### 分析

- 相比实验2（83.94%）**提升 0.69%**，说明恢复部分增强有效果
- 仍低于实验1（86.61%），说明 batch=128 可能对泛化有一定负面影响
- batch 越大，每个 epoch 的更新次数越少，模型学习步调更粗

### 结论

- batch=128 虽然 GPU 跑满，但泛化不如 batch=16
- **实验1 的配置仍然是最佳基线**

---

## 后续优化计划

1. 使用 Optuna 搜索超参数组合（学习率、batch_size、dropout、arcface_m 等）
2. 预期目标：Test Accuracy ≥ 88%
3. Optuna 配置：`configs/config_optuna.yaml`
4. 运行方式：`python optimize.py --config configs/config_optuna.yaml --n-trials 50`
