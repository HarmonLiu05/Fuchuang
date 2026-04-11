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
| **dataset_D_ge3years_count** | 107 | 2,620 | 1,194 | ≥3 年 |
| dataset_E_ge5years_count | 38 | 966 | 438 | ≥5 年 |
| dataset_F_ge7years_count | 10 | 326 | 147 | ≥7 年 |

### 实验对比（dataset_D）

| 实验 | batch | 数据增强 | GPU-Util | Test Accuracy | 说明 |
|------|-------|---------|----------|---------------|------|
| **实验 1** | 16 | 全开 | 59% | **86.61%** | 🥇 最佳基线 |
| **实验 3** | 128 | 适中 | 100% | **84.63%** | GPU 跑满但精度下降 |
| **实验 2** | 128 | 精简 | 100% | **83.94%** | 增强太少导致过拟合 |

> 📝 Optuna 优化进行中，目标 ≥ 88%

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
