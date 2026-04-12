# 🐢 龟类个体识别系统 (Turtle ArcFace)

基于 ResNet101 + ArcFace 的龟类头部个体识别模型，支持 Optuna 超参数自动优化。

## 功能

- ✅ **个体识别**：输入龟类头部图像，输出个体 ID
- ✅ **Optuna 超参数优化**：自动搜索最佳学习率、batch size、dropout 等
- ✅ **混合精度训练**：支持 FP16，节省显存
- ✅ **分阶段训练**：先冻结 backbone，后微调整体
- ✅ **数据增强**：翻转、仿射、颜色抖动、高斯模糊、随机擦除
- ✅ **评估指标**：Accuracy、TAR@FAR、Rank-1
- ✅ **多 Backbone 支持**：ResNet50 / ResNet101
- ✅ **多层 MLP Bottleneck**：增强非线性表达能力

## 项目结构

```
chimpanzee_arcface/
├── configs/
│   ├── config_turtle.yaml       # 龟类训练配置
│   └── config_optuna.yaml       # Optuna 优化配置
├── data/
│   ├── dataset.py               # 数据增强函数
│   ├── turtle_dataset.py        # COCO 格式数据集加载器
│   └── prepare_turtle_data.py   # 数据准备函数
├── models/
│   ├── backbone.py              # ResNet50/101 backbone
│   ├── bottleneck.py            # 多层 MLP 瓶颈层
│   ├── arcface.py               # ArcFace Head
│   └── se_block.py              # SE-Block 注意力模块
├── utils/
│   ├── metrics.py               # 评估指标计算
│   └── utils.py                 # 工具函数
├── optimize.py                  # Optuna 超参数优化脚本
├── train_turtle.py              # 龟类训练脚本
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

### 3. 训练模型（手动配置）

```bash
# 编辑 configs/config_turtle.yaml 修改参数后运行
python train_turtle.py --config configs/config_turtle.yaml
```

### 4. Optuna 自动优化（推荐）

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

### 5. 评估模型

```bash
python evaluate.py --checkpoint checkpoints_turtle/best_model.pth --config configs/config_turtle.yaml
```

## 实验结果

### 数据集

| 数据集 | 个体数 | 训练图 | 测试图 | 时间跨度 |
|--------|--------|--------|--------|---------|
| **dataset_D_ge3years_count** | 107 | 2,616 | 1,158 | ≥3 年 |
| dataset_E_ge5years_count | 38 | 966 | 438 | ≥5 年 |
| dataset_F_ge7years_count | 10 | 326 | 147 | ≥7 年 |

### 模型评估对比（dataset_D）

| 模型 | Epoch | Triplet Loss | Acc (直接预测) | Rank-1 (检索) | TAR@FAR=0.1% | Threshold | 说明 |
|------|-------|--------------|----------------|---------------|--------------|-----------|------|
| **epoch200_resnet-101_triloss_83%.pth** | 200 | ✅ 启用 | 85.84% | **94.13%** | **79.77%** | 0.6496 | 🥇 最佳模型 |
| best_model101_80epoch.pth | 80 | ❌ | 64.77% | 91.02% | 64.06% | 0.6016 | Baseline |
| best_model1.pth | - | - | - | - | - | - | 待评估 |

**指标说明**：
- **Acc (Accuracy)**: 直接预测准确率，ArcFace head 输出的 argmax 与真实标签比较
- **Rank-1**: 检索第一正确率，基于特征向量相似度（余弦/欧氏距离）检索，最相似样本的标签是否正确
- **TAR@FAR=0.1%**: 误识率 0.1% 时的真接受率

> 💡 训练 200 轮 + Triplet Loss 的模型明显优于 80 轮 baseline，Rank-1 提升 **3.11%**，TAR@FAR=0.1% 提升 **15.71%**

### 72 个体对比实验（min_samples_per_identity=15）

**实验设置说明**：
- 所有实验均基于 **72 个个体**（过滤照片<15 的个体，训练集 2244 张，测试集 490 张）
- 均从相同的 Baseline 权重（纯交叉熵 56 轮，Acc_direct=0.64）接续训练
- 总训练轮次：200 Epoch（从第 57 轮开始引入新损失函数）
- 学习率：base_lr=0.001, backbone_lr=0.0001（CosineAnnealingLR, T_max=200）
- Triplet Loss 权重：0.2，Warmup 10 轮
- 数据加载：`num_workers=8`，普通 Shuffle（35 个 batch/epoch）

| 实验 | 损失函数配置 | Epoch | Acc (直接预测) | Acc (检索) | 权重路径 |
|------|-------------|-------|----------------|------------|---------|
| **1. Baseline (纯交叉熵)** | CE only (80 epoch 完成) | 80 | **60.82%** | 90.61% | `/workspace/experiments-checkpoints/baseline_ce_only/best_model1.pth` |
| **2. 距离三元组** | CE + Batch-Hard Triplet | 200 (从 57 开始) | **71.43%** | 0.00%* | `/workspace/experiments-checkpoints/baseline_dist_triplet_72ids/best_model1.pth` |
| **3. 时间 APN** | CE + Temporal-APN Triplet | 200 (从 57 开始) | **73.27%** | 0.00%* | `/workspace/experiments-checkpoints/best_apn_temporal/best_model1.pth` |

**实验 2 关键说明**：
- **Triplet Loss 类型**：Batch-Hard（普通距离三元组，不考虑时间）
- **Positive 选择**：同类中特征距离最远的样本
- **Negative 选择**：异类中特征距离最近的样本
- **Batch 采样策略**：普通 Shuffle（35 batch/epoch），不强制同个体多张照片
- **效果**：Acc_direct 从 Baseline 的 60.82% 提升至 **71.43%**（+10.61%），说明距离三元组损失显著增强了特征判别力

**实验 3 关键说明**（已完成 ✅）：
- **Triplet Loss 类型**：Temporal-APN（时间跨度感知三元组，精确到月份）
- **APN 选择逻辑**：Positive=同类中【时间跨度最大】的样本（平局用距离最远）；Negative=异类中【时间跨度最近】的样本（平局用距离最近）
- **Batch 采样策略**：普通 Shuffle（35 batch/epoch），不强制同个体多张照片
- **效果**：Acc_direct 从 Baseline 的 60.82% 提升至 **73.27%**（+12.45%），**比距离三元组高 1.84%**
- **结论**：时间跨度选择策略在特征判别力上**优于**纯距离选择，说明时间信息有助于模型学习时间鲁棒的特征
- **注意**：Acc=0.00% 为临时显示异常，实际 Rank-1 检索能力待 evaluate.py 验证

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
ArcFace Head: s=30.0, m=0.4, 107 分类
    ↓
输出个体 ID
```

## 训练策略

| 阶段 | Epoch | 训练内容 | 学习率 |
|------|-------|---------|--------|
| 阶段 1 | 1-10 | backbone.layer4+ + bottleneck + arcface | backbone: 1e-4, 其他: 1e-3 |
| 阶段 2 | 11-30 | 全部参数微调 | 余弦衰减 |

## Optuna 搜索空间

| 参数 | 搜索范围 |
|------|---------|
| backbone | resnet50, resnet101 |
| base_lr | 1e-4 ~ 1e-2 |
| backbone_lr | 1e-5 ~ 1e-3 |
| batch_size | 64, 128, 256 |
| dropout | 0.2 ~ 0.6 |
| arcface_m | 0.3 ~ 0.5 |
| arcface_s | 25.0 ~ 35.0 |
| weight_decay | 1e-5 ~ 1e-4 |
| freeze_until_layer | 0 ~ 5 |
| use_se_block | true, false |

## 数据集

- **名称**: Turtle Head Dataset
- **个体数**: 最高 107 个（dataset_D）
- **图片数**: 最高 3,814 张
- **格式**: COCO JSON 标注
- **来源**: [turtlehead-dataset](https://github.com/HarmonLiu05/turtlehead-dataset)

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

## License

MIT
