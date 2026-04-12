# 🐢 龟类个体识别系统 (Turtle ArcFace)

基于 ResNet101 + ArcFace 的龟类头部个体识别模型，支持 **时间跨度感知三元组损失 (Temporal APN)** 和 **时间加权 Triplet Loss**。

## 功能

- ✅ **个体识别**：输入龟类头部图像，输出个体 ID
- ✅ **时间跨度感知 Triplet Loss (Temporal APN)**：同类选时间跨度最大的样本作为 Positive，异类选时间跨度最小的作为 Negative
- ✅ **时间加权 Triplet Loss**：根据时间差距动态调整 Triplet Loss 权重
- ✅ **混合精度训练**：支持 FP16，节省显存
- ✅ **分阶段训练**：先冻结 backbone，后微调整体
- ✅ **数据增强**：翻转、仿射、颜色抖动、高斯模糊、随机擦除
- ✅ **评估指标**：Accuracy、TAR@FAR、Rank-1
- ✅ **多 Backbone 支持**：ResNet50 / ResNet101 / ResNet152
- ✅ **多层 MLP Bottleneck**：增强非线性表达能力
- ✅ **SE-Block 注意力**：可选的特征通道注意力模块
- ✅ **分层学习率**：不同 backbone 层使用不同学习率

## 项目结构

```
Fuchuang/
├── configs/
│   ├── config_turtle.yaml                  # 标准训练配置（时间加权 Triplet）
│   ├── config_baseline_ce_only.yaml        # Baseline：纯交叉熵
│   ├── config_baseline_dist_triplet_72ids.yaml  # Baseline：距离三元组
│   ├── config_best_apn_temporal.yaml       # 时间 APN 三元组
│   ├── config_exp3_temporal_apn.yaml       # 实验3：时间 APN
│   └── config_exp3_temporal_apn_72ids.yaml # 实验3：时间 APN（72个体）
├── data/
│   ├── dataset.py               # 数据增强函数
│   ├── turtle_dataset.py        # COCO 格式数据集加载器
│   └── prepare_turtle_data.py   # 数据准备（支持 TimeAwareBatchSampler）
├── models/
│   ├── backbone.py              # ResNet50/101/152 backbone
│   ├── bottleneck.py            # 多层 MLP 瓶颈层
│   ├── arcface.py               # ArcFace Head
│   └── se_block.py              # SE-Block 注意力模块
├── losses/
│   ├── triplet.py               # Batch-Hard Triplet Loss
│   ├── time_weighted_triplet.py # 时间加权 Triplet Loss
│   └── temporal_apn_triplet.py  # 时间跨度感知 APN Triplet Loss
├── samplers/
│   └── time_aware_sampler.py    # 时间感知 Batch 采样器
├── utils/
│   ├── metrics.py               # 评估指标计算
│   └── utils.py                 # 工具函数
├── train_turtle.py              # 训练脚本（支持 --resume）
├── evaluate.py                  # 评估脚本
├── inference.py                 # 推理脚本
├── requirements.txt             # 依赖
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据集

龟类数据集已独立仓库：[turtlehead-dataset](https://github.com/HarmonLiu05/turtlehead-dataset)

```bash
git clone https://github.com/HarmonLiu05/turtlehead-dataset.git
```

### 3. 下载预训练权重

预训练权重已独立仓库：[Weight](https://github.com/HarmonLiu05/Weight)

```bash
git clone https://github.com/HarmonLiu05/Weight.git
```

### 4. 训练模型（手动配置）

```bash
# 标准训练（时间加权 Triplet Loss）
python train_turtle.py --config configs/config_turtle.yaml

# 从 checkpoint 恢复训练
python train_turtle.py --config configs/config_turtle.yaml \
    --resume /path/to/checkpoint.pth
```

### 5. Optuna 自动优化（推荐）

```bash
# 自动搜索 50 组超参数组合
python optimize.py --config configs/config_optuna.yaml --n-trials 50

# 查看最佳参数
cat results/optuna_study/best_params.yaml

# 可视化优化历史
pip install optuna-dashboard
optuna-dashboard sqlite:///results/optuna_study/study.db
# 浏览器打开 http://localhost:8080
```

### 6. 评估模型

```bash
python evaluate.py --checkpoint checkpoints_turtle/best_model.pth \
    --config configs/config_turtle.yaml
```

## 实验结果

### 数据集

| 数据集 | 个体数 | 训练图 | 测试图 | 时间跨度 |
|--------|--------|--------|--------|---------|
| **dataset_D_ge3years_count** | 107 | 2,616 | 1,158 | ≥3 年 |
| dataset_E_ge5years_count | 38 | 966 | 438 | ≥5 年 |
| dataset_F_ge7years_count | 10 | 326 | 147 | ≥7 年 |

### 模型评估对比（dataset_D, 107 个体）

| 模型 | Epoch | Triplet Loss | Acc (直接预测) | Rank-1 (检索) | TAR@FAR=0.1% | Threshold | 说明 |
|------|-------|--------------|----------------|---------------|--------------|-----------|------|
| **epoch200_resnet-101_triloss_83%.pth** | 200 | ✅ 启用 | 85.84% | **94.13%** | **79.77%** | 0.6496 | 🥇 最佳模型 |
| best_model101_80epoch.pth | 80 | ❌ | 64.77% | 91.02% | 64.06% | 0.6016 | Baseline |

> 💡 训练 200 轮 + Triplet Loss 的模型明显优于 80 轮 baseline，Rank-1 提升 **3.11%**，TAR@FAR=0.1% 提升 **15.71%**

### 72 个体消融实验（min_samples_per_identity=15）

**实验设置**：
- 所有实验基于 **72 个个体**（过滤照片<15 的个体，训练集 2244 张，测试集 490 张）
- 均从相同 Baseline 权重（纯交叉熵 56 轮，Acc_direct=0.64）接续训练
- 总训练轮次：200 Epoch（从第 57 轮开始引入新损失函数）
- 学习率：base_lr=0.001, backbone_lr=0.0001（CosineAnnealingLR, T_max=200）
- Triplet Loss 权重：0.2，Warmup 10 轮

| 实验 | 损失函数配置 | Epoch | Acc (直接预测) | Acc (检索) | 说明 |
|------|-------------|-------|----------------|------------|------|
| **1. Baseline (纯交叉熵)** | CE only (80 epoch 完成) | 80 | **60.82%** | 90.61% | 对照基准 |
| **2. 距离三元组** | CE + Batch-Hard Triplet | 200 (从 57 开始) | **71.43%** | 0.00%* | +10.61% |
| **3. 时间 APN** | CE + Temporal-APN Triplet | 200 (从 57 开始) | **73.27%** | 0.00%* | **+12.45%** |

**关键结论**：
- 距离三元组使 Acc_direct 提升 **+10.61%**
- 时间 APN 比距离三元组再高 **+1.84%**
- **时间信息有助于模型学习时间鲁棒的特征**

> *Acc=0.00% 为临时显示异常，实际 Rank-1 检索能力待 evaluate.py 验证

## 🚧 下一步目标

### 1. 在 100+ 个体上验证 APN 有效性

当前 72 个体的消融实验已证明时间 APN 优于纯距离三元组（+1.84%），但样本量有限。下一步将在 **107 个体的 dataset_D**（min_samples≥5）上验证：

- [ ] 使用完整 107 个体重新训练 Baseline / 距离三元组 / 时间 APN
- [ ] 对比 APN 在更大个体量级下的增益是否依然显著
- [ ] 验证时间跨度选择策略是否在更多个体间具有泛化能力

### 2. 迁移到人脸数据集

将当前训练框架迁移到**人脸个体识别**任务，验证时间 APN 三元组在更广泛场景下的有效性：

- [ ] 适配人脸数据集格式（如 VGGFace2、CASIA-WebFace、MS1M 等）
- [ ] 将"拍摄时间"映射为"采集时间"，保留时间跨度选择逻辑
- [ ] 训练并对比：人脸场景下 APN vs 距离三元组的性能差异
- [ ] 探索跨物种泛化：龟类 → 人脸的特征学习是否可以互相迁移

## 损失函数对比

| 损失类型 | Positive 选择 | Negative 选择 | 适用场景 |
|---------|--------------|--------------|---------|
| **Batch-Hard Triplet** | 同类中距离最远 | 异类中距离最近 | 通用度量学习 |
| **Time-Weighted Triplet** | 同类中距离最远 | 异类中距离最近 + 时间加权 | 有时间信息的数据集 |
| **Temporal APN Triplet** | 同类中**时间跨度最大** | 异类中**时间跨度最小** | 强调时间鲁棒性 |

## 模型架构

```
输入图像 (224×224×3)
    ↓
ResNet101 Backbone (ImageNet 预训练)
    ↓
SE-Block (可选)
    ↓
AdaptiveAvgPool(7×7) → 2048 维
    ↓
多层 MLP Bottleneck: 2048→1024→512 + BN + ReLU + Dropout + L2 归一化
    ↓
ArcFace Head: s=30.0, m=0.35, 107 分类
    ↓
输出个体 ID
```

## 训练策略

| 阶段 | Epoch | 训练内容 | 学习率 |
|------|-------|---------|--------|
| 阶段 1 | 1-10 | backbone.layer4+ + bottleneck + arcface | backbone: 1e-4, 其他: 1e-3 |
| 阶段 2 | 11-200 | 全部参数微调 | 余弦衰减 → 1e-6 |

### Triplet Loss 调度

| Epoch | Triplet 状态 | 权重 |
|-------|-------------|------|
| 0-79 | ❌ 未启用 | 0 |
| 80-99 | ⏳ Warmup | 0 → 0.2 |
| 100-200 | ✅ 启用 | 0.2 |

## 依赖

```
torch>=2.0.0
torchvision>=0.15.0
pyyaml>=6.0
numpy>=1.24.0
Pillow>=9.0.0
tqdm>=4.65.0
matplotlib>=3.7.0
scikit-learn>=1.2.0
faiss-cpu>=1.7.0
optuna>=3.0.0
kaleido>=0.2.1
```

## 相关仓库

- 📦 [数据集](https://github.com/HarmonLiu05/turtlehead-dataset)
- ⚖️ [预训练权重](https://github.com/HarmonLiu05/Weight)

## License

MIT
