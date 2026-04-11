# 🐢 龟类个体识别系统 (Turtle ArcFace)

基于 ResNet101 + ArcFace 的龟类头部个体识别模型。

## 功能

- ✅ **个体识别**：输入龟类头部图像，输出个体 ID
- ✅ **混合精度训练**：支持 FP16，节省显存
- ✅ **分阶段训练**：先冻结 backbone，后微调整体
- ✅ **数据增强**：翻转、仿射、颜色抖动、高斯模糊、随机擦除
- ✅ **评估指标**：Accuracy（最近邻）、Accuracy0（直接分类）
- ✅ **多 Backbone 支持**：ResNet50 / ResNet101
- ✅ **多层 MLP Bottleneck**：增强非线性表达能力

## 项目结构

```
chimpanzee_arcface/
├── configs/
│   └── config_turtle.yaml       # 龟类训练配置
├── data/
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
├── train_turtle.py              # 龟类训练脚本
├── evaluate.py                  # 评估脚本
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

### 3. 训练模型

```bash
# 编辑 configs/config_turtle.yaml 修改参数后运行
python train_turtle.py --config configs/config_turtle.yaml
```

### 4. 评估模型

```bash
python evaluate.py --checkpoint checkpoints_turtle/best_model.pth
```

## 实验结果

### 数据集

| 数据集 | 个体数 | 训练图 | 测试图 | 时间跨度 |
|--------|--------|--------|--------|---------|
| **dataset_D_ge3years_count** | 107 | 2,616 | 1,158 | ≥3 年 |

### 实验 1: 200 Epoch 长训练

| 指标 | 值 |
|------|-----|
| **最佳 Acc0** | **64.68%** |
| **最终 Acc0** | 63.82% |
| **Acc (最近邻)** | ~85% |
| 训练轮次 | 222 epochs |

### 配置

```yaml
model:
  backbone: resnet101
  bottleneck_dim: 512
  dropout: 0.4
  arcface_s: 30.0
  arcface_m: 0.35

training:
  batch_size: 128
  base_lr: 0.002
  backbone_lr: 0.0002
  epochs: 200
  freeze_until_epoch: 10
```

## 模型架构

```
输入图像 (224×224×3)
    ↓
ResNet101 Backbone (ImageNet 预训练)
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
| 阶段 1 | 1-10 | backbone.layer4+ + bottleneck + arcface | backbone: 2e-4, 其他: 2e-3 |
| 阶段 2 | 11-200 | 全部参数微调 | 余弦衰减 |

## 数据集

- **名称**: Turtle Head Dataset
- **个体数**: 107（dataset_D）
- **图片数**: 3,774 张
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
```

## License

MIT
